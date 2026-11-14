"""Stage 7 detector physics, accounting, reconstruction, and visualizations."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PySide6.QtWidgets import QApplication

from petsim.acquisition_controller import AcquisitionController
from petsim.experiments.config import ExperimentConfig
from petsim.experiments.export import metadata_lines
from petsim.experiments.runner import ExperimentResult, collect_coincidences, run_experiment
from petsim.main_window import MainWindow
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer, AcquisitionConfig, AcquisitionStats
from petsim.physics.comparison import reconstruct_ideal_and_degraded
from petsim.physics.detector_effects import (
    DetectorPhysicsConfig,
    account_events,
    compton_energy_kev,
    fwhm_to_sigma,
    timing_accepted,
)
from petsim.physics.pet_engine import PHOTON_ENERGY_KEV, PETEngine
from petsim.physics_comparison_worker import PhysicsComparisonRunner
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.widgets.detector_physics_panel import DetectorPhysicsPanel


def _engine(seed: int = 1) -> PETEngine:
    return PETEngine(BrainPhantom(), DetectorRing(48), seed=seed)


def _on_ring(points: np.ndarray, radius: float = 0.92) -> bool:
    if points.size == 0:
        return True
    radii = np.hypot(points[:, 0], points[:, 1])
    return bool(np.allclose(radii, radius, atol=1e-6))


def test_compton_endpoints_and_fwhm_conversion():
    assert abs(compton_energy_kev(511.0, 0.0) - 511.0) < 1e-9
    assert abs(compton_energy_kev(511.0, np.pi / 2.0) - 255.5) < 1e-6
    assert abs(compton_energy_kev(511.0, np.pi) - 511.0 / 3.0) < 1e-6
    assert abs(fwhm_to_sigma(2.0 * np.sqrt(2.0 * np.log(2.0))) - 1.0) < 1e-9


def test_timing_window_uses_measured_difference_not_residual():
    flight_ps = 800.0
    residual_ps = 30.0
    measured_ps = flight_ps + residual_ps
    # The residual would pass a 400 ps window. The measured difference does not.
    assert abs(residual_ps) <= 400.0
    assert timing_accepted(measured_ps, 400.0) is False
    assert timing_accepted(flight_ps, 400.0) is False
    assert timing_accepted(residual_ps, 400.0) is True


def test_zero_timing_resolution_rejects_on_flight_time_only():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=0.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20.0,
        random_fraction=0.0,
        scatter_probability=0.0,
    )
    events = _engine(2).simulate_batch(400, detector_efficiency=1.0, physics=physics)
    assert events
    assert all(abs(ev.timing_residual_ps) < 1e-9 for ev in events)
    assert all(
        abs(ev.measured_delta_t_ps - ev.flight_time_difference_ps) < 1e-6 for ev in events
    )
    rejected = [ev for ev in events if not ev.passed_timing]
    accepted = [ev for ev in events if ev.passed_timing]
    assert rejected and accepted
    assert all(abs(ev.measured_delta_t_ps) > 20.0 for ev in rejected)
    assert all(abs(ev.flight_time_difference_ps) <= 20.0 for ev in accepted)


def test_disabled_config_matches_ideal_events():
    ideal = _engine(4).simulate_batch(300, detector_efficiency=0.8)
    disabled = _engine(4).simulate_batch(
        300,
        detector_efficiency=0.8,
        physics=DetectorPhysicsConfig(
            enabled=False,
            random_fraction=1.0,
            scatter_probability=1.0,
            energy_fwhm_percent=30.0,
        ),
    )
    assert [ev.annihilation_xy for ev in ideal] == [ev.annihilation_xy for ev in disabled]
    assert [ev.hit_a_xy for ev in ideal] == [ev.hit_a_xy for ev in disabled]
    assert [ev.is_coincidence for ev in ideal] == [ev.is_coincidence for ev in disabled]
    assert all(not ev.is_random and not ev.is_scattered for ev in disabled)


def test_energy_window_requires_both_photons_and_is_reproducible():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=20.0,
        energy_window_low_kev=500.0,
        energy_window_high_kev=522.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20000.0,
        random_fraction=0.0,
        scatter_probability=0.0,
    )
    first = _engine(8).simulate_batch(600, detector_efficiency=1.0, physics=physics)
    second = _engine(8).simulate_batch(600, detector_efficiency=1.0, physics=physics)
    assert [round(ev.energy_a_kev, 6) for ev in first] == [round(ev.energy_a_kev, 6) for ev in second]
    assert [ev.passed_energy for ev in first] == [ev.passed_energy for ev in second]
    for ev in first:
        inside = (
            physics.energy_window_low_kev <= ev.energy_a_kev <= physics.energy_window_high_kev
            and physics.energy_window_low_kev <= ev.energy_b_kev <= physics.energy_window_high_kev
        )
        assert ev.passed_energy is inside
        if not inside:
            assert not ev.is_coincidence


def test_scatter_changes_energy_and_zero_angle_preserves_the_hit():
    blurred = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=0.0,
        energy_window_low_kev=450.0,
        energy_window_high_kev=600.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20000.0,
        scatter_probability=1.0,
        scatter_angle_sigma_rad=1.1,
    )
    events = _engine(11).simulate_batch(150, detector_efficiency=1.0, physics=blurred)
    assert all(ev.is_scattered for ev in events)
    assert any(min(ev.energy_a_kev, ev.energy_b_kev) < 450.0 for ev in events)
    assert any(ev.is_coincidence for ev in events)
    straight = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=0.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20000.0,
        scatter_probability=1.0,
        scatter_angle_sigma_rad=0.0,
    )
    untouched = _engine(12).simulate_batch(40, detector_efficiency=1.0, physics=straight)
    for ev in untouched:
        assert abs(ev.energy_a_kev - PHOTON_ENERGY_KEV) < 1e-9
        assert ev.ideal_hit_a_xy is not None
        assert np.allclose(ev.hit_a_xy, ev.ideal_hit_a_xy, atol=1e-8)
        assert np.allclose(ev.hit_b_xy, ev.ideal_hit_b_xy, atol=1e-8)


def test_random_pairs_are_extra_crystal_center_coincidences():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=0.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20000.0,
        random_fraction=1.0,
        scatter_probability=0.0,
    )
    events = _engine(9).simulate_batch(40, detector_efficiency=1.0, physics=physics)
    randoms = [ev for ev in events if ev.is_random]
    annihilations = [ev for ev in events if not ev.is_random]
    assert len(annihilations) == 40
    assert len(randoms) == 40
    radius = DetectorRing(48).radius
    for ev in randoms:
        assert ev.is_coincidence
        assert ev.annihilation_xy == (0.0, 0.0)
        assert ev.detector_a != ev.detector_b
        assert abs(np.hypot(*ev.hit_a_xy) - radius) < 1e-8
    accounting = account_events(events)
    assert accounting.annihilation_partition_holds()
    assert accounting.accepted_random == 40
    assert accounting.accepted_coincidences == 80


def test_energy_and_timing_rejections_are_not_double_counted():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=30.0,
        energy_window_low_kev=505.0,
        energy_window_high_kev=517.0,
        timing_fwhm_ps=4000.0,
        timing_window_ps=50.0,
        random_fraction=0.25,
        scatter_probability=0.2,
    )
    events = _engine(21).simulate_batch(500, detector_efficiency=0.7, physics=physics)
    second = _engine(21).simulate_batch(500, detector_efficiency=0.7, physics=physics)
    assert [ev.is_coincidence for ev in events] == [ev.is_coincidence for ev in second]
    assert [ev.is_random for ev in events] == [ev.is_random for ev in second]
    accounting = account_events(events)
    assert accounting.annihilation_partition_holds()
    energy_ids = {
        id(ev)
        for ev in events
        if not ev.is_random and ev.detected_a and ev.detected_b and not ev.passed_energy
    }
    timing_ids = {
        id(ev)
        for ev in events
        if (
            not ev.is_random
            and ev.detected_a
            and ev.detected_b
            and ev.passed_energy
            and not ev.passed_timing
        )
    }
    assert energy_ids.isdisjoint(timing_ids)
    assert len(energy_ids) == accounting.energy_rejected
    assert len(timing_ids) == accounting.timing_rejected
    assert accounting.energy_rejected > 0
    assert accounting.timing_rejected > 0


def test_degraded_endpoints_reconstruct_with_fbp_and_mlem_without_truth():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=10.0,
        timing_fwhm_ps=300.0,
        timing_window_ps=6000.0,
        random_fraction=0.3,
        scatter_probability=0.25,
    )
    events = _engine(6).simulate_batch(700, detector_efficiency=0.95, physics=physics)
    buffer = AcquisitionBuffer()
    buffer.extend(events)
    degraded_a, degraded_b = buffer.coincidence_endpoints()
    ideal_a, ideal_b = buffer.ideal_coincidence_endpoints()
    assert degraded_a.shape[0] > 0
    assert ideal_a.shape[0] > 0
    assert _on_ring(degraded_a) and _on_ring(degraded_b)
    assert _on_ring(ideal_a) and _on_ring(ideal_b)
    # Truth positions sit inside the brain, not on the detector ring.
    truths = np.asarray([ev.annihilation_xy for ev in events if not ev.is_random])
    assert np.mean(np.hypot(truths[:, 0], truths[:, 1])) < 0.7
    fbp_cfg = ReconstructionConfig(method="fbp", image_size=32, n_theta=24, n_s=32).clamp()
    mlem_cfg = ReconstructionConfig(
        method="mlem", n_iterations=2, image_size=32, n_theta=24, n_s=32
    ).clamp()
    fbp = reconstruct_from_endpoints(degraded_a, degraded_b, fbp_cfg)
    mlem = reconstruct_from_endpoints(degraded_a, degraded_b, mlem_cfg)
    assert fbp.image.shape == (32, 32)
    assert mlem.image.shape == (32, 32)
    assert np.isfinite(fbp.image).all()
    assert np.isfinite(mlem.image).all()
    assert fbp.n_coincidences == degraded_a.shape[0]
    assert mlem.history is not None


def test_comparison_can_be_cancelled_before_the_degraded_image():
    physics = DetectorPhysicsConfig(
        enabled=True,
        energy_fwhm_percent=0.0,
        timing_fwhm_ps=0.0,
        timing_window_ps=20000.0,
        random_fraction=0.5,
    )
    events = _engine(3).simulate_batch(80, detector_efficiency=1.0, physics=physics)
    buffer = AcquisitionBuffer()
    buffer.extend(events)
    ideal_a, ideal_b = buffer.ideal_coincidence_endpoints()
    degraded_a, degraded_b = buffer.coincidence_endpoints()
    config = ReconstructionConfig(method="fbp", image_size=32, n_theta=16, n_s=32).clamp()
    ideal, degraded = reconstruct_ideal_and_degraded(
        ideal_a,
        ideal_b,
        degraded_a,
        degraded_b,
        config,
        cancel=lambda: True,
    )
    assert ideal.cancelled
    assert degraded.cancelled
    assert degraded.n_coincidences == 0


def test_controller_accounting_and_reset(qapp):
    del qapp
    controller = AcquisitionController()
    controller.apply_config(
        AcquisitionConfig(n_events=100, detector_efficiency=1.0, animation_speed="Fast", use_seed=True, seed=5)
    )
    controller.apply_detector_physics(
        DetectorPhysicsConfig(
            enabled=True,
            energy_fwhm_percent=0.0,
            timing_fwhm_ps=0.0,
            timing_window_ps=20000.0,
            random_fraction=1.0,
            scatter_probability=0.0,
        )
    )
    controller.engine.set_seed(5)
    controller._stats.running = True
    controller._on_tick()
    stats = controller.stats
    assert stats.total_events == 25
    assert stats.accepted_unscattered == 25
    assert stats.random_coincidences == 25
    assert stats.coincidences == 50
    assert stats.energy_rejected == 0
    assert stats.timing_rejected == 0
    assert stats.detected_singles == 0
    assert stats.total_events == (
        stats.accepted_unscattered
        + stats.scattered_coincidences
        + stats.energy_rejected
        + stats.timing_rejected
        + stats.detected_singles
        + stats.undetected
    )
    controller.reset()
    assert controller.stats.total_events == 0
    assert controller.stats.coincidences == 0
    assert controller.stats.random_coincidences == 0
    assert controller.buffer.n_annihilations() == 0
    assert controller.measured_photon_energies_kev().size == 0
    assert controller.coincidence_endpoints()[0].shape[0] == 0


def test_collect_cancel_and_experiment_metadata(tmp_path: Path):
    engine = _engine(1)
    physics = DetectorPhysicsConfig(enabled=True, random_fraction=0.2, energy_fwhm_percent=5.0)
    assert collect_coincidences(
        engine,
        20,
        detector_efficiency=1.0,
        physics=physics,
        cancel=lambda: True,
    ) is None
    config = ExperimentConfig(
        phantom=PhantomState(),
        methods=("fbp",),
        count_targets=(20,),
        n_trials=1,
        seed=4,
        image_size=32,
        n_theta=16,
        n_s=32,
        detector_efficiency=1.0,
        detector_physics=physics,
    )
    result = run_experiment(config)
    assert result.trials
    assert result.trials[0].status == "ok"
    text = "\n".join(metadata_lines(result))
    assert "detector_effects_applied=true" in text
    assert "energy_fwhm_percent_at_511keV=5.0" in text
    assert "timing_window_quantity=abs_measured_delta_t_ps" in text
    assert "random_fraction=0.2" in text
    ideal = ExperimentResult(config=ExperimentConfig(phantom=PhantomState()).clamp())
    ideal_text = "\n".join(metadata_lines(ideal))
    assert "detector_effects_applied=false" in ideal_text
    destination = tmp_path / "trials.csv"
    from petsim.experiments.export import export_trials_csv

    export_trials_csv(result, destination)
    saved = destination.read_text(encoding="utf-8")
    assert "detector_effects_applied=true" in saved


def test_visualizations_are_independent_and_reset(qapp):
    del qapp
    window = MainWindow()
    panel = window.detector_physics
    foreign = {
        window.reconstruction.canvas.figure,
        window.reconstruction.convergence_canvas.figure,
        window.reconstruction.count_canvas.figure,
        window.experiments.plot_localization.figure,
        window.experiments.plot_tbr.figure,
        window.experiments.plot_cv.figure,
        window.experiments.plot_runtime.figure,
    }
    own = {
        panel.energy_canvas.figure,
        panel.class_canvas.figure,
        panel.comparison_canvas.figure,
    }
    assert own.isdisjoint(foreign)
    assert len(own) == 3
    panel.energy_canvas.show_spectrum(np.array([480.0, 511.0, 540.0, 620.0]), 435.0, 650.0)
    assert len(panel.energy_canvas._ax.lines) == 2
    panel.class_canvas.show_counts(
        AcquisitionStats(accepted_unscattered=4, scattered_coincidences=2, random_coincidences=3)
    )
    assert len(panel.class_canvas._ax.patches) == 3
    panel.clear_visuals()
    assert panel._compare_status.text() == "Comparison cleared."
    assert not window.experiments._use_physics.isChecked()
    built = window.experiments.build_config(PhantomState())
    assert built is not None
    assert built.detector_physics is None
    window.experiments._use_physics.setChecked(True)
    window.experiments.set_detector_physics(
        DetectorPhysicsConfig(enabled=True, energy_fwhm_percent=15.0, random_fraction=0.1)
    )
    opted = window.experiments.build_config(PhantomState())
    assert opted is not None and opted.detector_physics is not None
    assert opted.detector_physics.enabled
    assert opted.detector_physics.energy_fwhm_percent == 15.0
    window.close()


def test_comparison_worker_drops_a_result_after_reset(qapp):
    app = qapp
    runner = PhysicsComparisonRunner()
    delivered: list[object] = []
    runner.finished.connect(delivered.append)
    config = ReconstructionConfig(method="fbp", image_size=16, n_theta=8, n_s=16).clamp()
    hits = np.array([[0.92, 0.0], [-0.92, 0.0], [0.0, 0.92]], dtype=np.float64)
    other = np.array([[-0.92, 0.0], [0.92, 0.0], [0.0, -0.92]], dtype=np.float64)
    assert runner.start(hits, other, hits, other, config)
    runner.invalidate_pending()
    assert runner.shutdown(timeout_ms=10_000)
    app.processEvents()
    assert delivered == []
    QApplication.processEvents()
