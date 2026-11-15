"""Stage 8 correction estimates, metrics, studies, and GUI."""

from __future__ import annotations

import inspect
import time

import numpy as np

from petsim.corrections.energy import (
    count_side_window_coincidences,
    scatter_transfer_coefficient,
)
from petsim.corrections.estimates import (
    apply_sinogram_corrections,
    estimate_random_sinogram,
    estimate_scatter_sinogram,
)
from petsim.corrections.metrics import scale_normalized_rmse, score_reconstruction
from petsim.corrections.reconstruct import (
    MODE_BOTH,
    MODE_RANDOM,
    MODE_SCATTER,
    MODE_UNCORRECTED,
    reconstruct_corrected,
)
from petsim.corrections.studies import CorrectionStudyConfig, acquire_matched, run_correction_study
from petsim.main_window import MainWindow
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.detector_effects import DetectorPhysicsConfig
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig
from petsim.reconstruction.sinogram import build_sinogram


def test_random_estimate_matches_the_configured_rate_and_is_reproducible():
    first = estimate_random_sinogram(1000, 0.2, n_theta=32, n_s=48, s_max=0.92)
    second = estimate_random_sinogram(1000, 0.2, n_theta=32, n_s=48, s_max=0.92)
    assert first.shape == (32, 48)
    assert abs(first.sum() - 200.0) < 1e-6
    assert np.array_equal(first, second)
    assert np.all(first >= 0.0)
    signature = inspect.signature(estimate_random_sinogram)
    assert "is_random" not in signature.parameters
    assert "tumor_x" not in signature.parameters


def test_random_correction_reduces_a_pure_random_sinogram():
    ring = DetectorRing(48)
    crystals = ring.crystal_positions()
    rng = np.random.default_rng(0)
    pairs = np.array([rng.choice(48, size=2, replace=False) for _ in range(2500)])
    measured = build_sinogram(
        crystals[pairs[:, 0]],
        crystals[pairs[:, 1]],
        n_theta=32,
        n_s=48,
        s_max=0.92,
    )
    estimate = estimate_random_sinogram(2500, 1.0, n_theta=32, n_s=48, s_max=0.92)
    residual = np.maximum(measured.data - estimate, 0.0)
    assert residual.sum() < measured.data.sum() * 0.75


def test_side_window_count_uses_energy_not_labels():
    energy_a = [511.0, 360.0, 500.0, 340.0]
    energy_b = [511.0, 500.0, 350.0, 330.0]
    detected = [True, True, True, True]
    timing = [True, True, True, False]
    count = count_side_window_coincidences(
        energy_a,
        energy_b,
        detected,
        detected,
        timing,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    )
    # Second and third pairs are witnesses. The fourth fails timing. 511 keV pairs are not.
    assert count == 2
    assert count_side_window_coincidences(
        energy_a,
        energy_b,
        detected,
        detected,
        timing,
        photo_low_kev=250.0,
        photo_high_kev=700.0,
    ) == 0
    assert scatter_transfer_coefficient(
        0.0,
        scatter_angle_sigma_rad=0.6,
        energy_fwhm_percent=12.0,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    ) == 0.0
    coefficient = scatter_transfer_coefficient(
        0.4,
        scatter_angle_sigma_rad=0.6,
        energy_fwhm_percent=12.0,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    )
    assert coefficient > 1.0
    peaked = np.full((8, 16), 4.0)
    estimate = estimate_scatter_sinogram(
        peaked,
        n_side_window=10,
        scatter_probability=0.4,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    )
    assert np.all(estimate >= 0.0)
    assert abs(estimate.sum() - min(coefficient * 10.0, peaked.sum())) < 1e-6
    assert estimate_scatter_sinogram(
        peaked,
        n_side_window=0,
        scatter_probability=0.4,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    ).sum() == 0.0
    source = inspect.getsource(estimate_scatter_sinogram) + inspect.getsource(count_side_window_coincidences)
    assert "is_scattered" not in source
    assert "tumor" not in source


def test_combined_correction_clips_prompts_and_hides_labels():
    measured = np.full((16, 32), 0.1, dtype=np.float64)
    edges = np.linspace(-0.92, 0.92, 33)
    corrected, random_est, scatter_est = apply_sinogram_corrections(
        measured,
        correct_random=True,
        correct_scatter=True,
        n_annihilations=5000,
        random_fraction=1.0,
        s_edges=edges,
        s_max=0.92,
    )
    assert np.all(corrected >= 0.0)
    assert np.all(np.isfinite(corrected))
    assert random_est.sum() > scatter_est.sum()
    signature = inspect.signature(reconstruct_corrected)
    joined = " ".join(signature.parameters)
    assert "is_random" not in joined
    assert "annihilation_xy" not in joined
    assert "tumor" not in joined


def test_fbp_and_mlem_accept_corrected_prompts():
    ring = DetectorRing(48)
    crystals = ring.crystal_positions()
    angles = np.linspace(0, 2 * np.pi, 36, endpoint=False)
    hits_a = np.column_stack((0.92 * np.cos(angles), 0.92 * np.sin(angles)))
    hits_b = -hits_a
    # A few random chords mixed into the measured list.
    rng = np.random.default_rng(1)
    extra = np.array([rng.choice(48, size=2, replace=False) for _ in range(12)])
    hits_a = np.vstack((hits_a, crystals[extra[:, 0]]))
    hits_b = np.vstack((hits_b, crystals[extra[:, 1]]))
    shared = dict(image_size=32, n_theta=16, n_s=32, smooth_sigma_s=0.0, smooth_sigma_theta=0.0)
    measured = build_sinogram(hits_a, hits_b, n_theta=16, n_s=32, s_max=0.92)
    by_method = {}
    for method in ("fbp", "mlem"):
        by_method[method] = {}
        for mode in (MODE_UNCORRECTED, MODE_RANDOM, MODE_SCATTER, MODE_BOTH):
            result = reconstruct_corrected(
                hits_a,
                hits_b,
                ReconstructionConfig(method=method, n_iterations=2, **shared).clamp(),
                mode=mode,
                n_annihilations=80,
                random_fraction=0.4,
                n_side_window=12,
                scatter_probability=0.35,
                photo_low_kev=435.0,
                photo_high_kev=650.0,
            )
            by_method[method][mode] = result
            assert result.image.shape == (32, 32)
            assert np.isfinite(result.image).all()
            assert result.config.method == method
            assert np.all(result.sinogram.data >= -1e-9)
        assert by_method[method][MODE_RANDOM].random_estimate_counts > 0.0
        assert by_method[method][MODE_RANDOM].scatter_estimate_counts == 0.0
        assert by_method[method][MODE_SCATTER].random_estimate_counts == 0.0
        assert by_method[method][MODE_SCATTER].scatter_estimate_counts > 0.0
        assert by_method[method][MODE_BOTH].random_estimate_counts > 0.0
        assert by_method[method][MODE_BOTH].scatter_estimate_counts > 0.0
    for mode in (MODE_UNCORRECTED, MODE_RANDOM, MODE_SCATTER, MODE_BOTH):
        assert np.allclose(
            by_method["fbp"][mode].sinogram.data,
            by_method["mlem"][mode].sinogram.data,
        )
    positive = measured.data > 0
    scatter_ratio = by_method["fbp"][MODE_SCATTER].sinogram.data[positive] / measured.data[positive]
    assert np.allclose(scatter_ratio, scatter_ratio.flat[0])
    assert float(scatter_ratio.flat[0]) < 1.0
    random_ratio = by_method["fbp"][MODE_RANDOM].sinogram.data[positive] / measured.data[positive]
    assert float(np.std(random_ratio)) > 1e-4


def test_scale_normalized_rmse_and_crc_on_the_phantom():
    phantom = PhantomState()
    activity = BrainPhantom(state=phantom, grid_size=48).rasterize()
    rmse, nrmse, alpha = scale_normalized_rmse(activity, activity)
    assert rmse is not None and rmse < 1e-6
    assert nrmse is not None and nrmse < 1e-6
    assert abs(alpha - 1.0) < 1e-6
    blank = np.zeros_like(activity)
    _rmse, blank_nrmse, _alpha = scale_normalized_rmse(blank, activity)
    assert blank_nrmse is not None and blank_nrmse > 0.5


def test_correction_study_is_reproducible_and_cancellable():
    phantom = PhantomState()
    config = CorrectionStudyConfig(
        phantom=phantom,
        methods=("fbp",),
        count_targets=(80,),
        random_fractions=(0.3,),
        scatter_probabilities=(0.2,),
        modes=(MODE_UNCORRECTED, MODE_RANDOM, MODE_SCATTER, MODE_BOTH),
        n_trials=1,
        seed=5,
        image_size=32,
        n_theta=16,
        n_s=32,
        n_iterations=1,
    )
    first = run_correction_study(config)
    second = run_correction_study(config)
    assert len(first.trials) == 4
    assert [trial.mode for trial in first.trials] == [MODE_UNCORRECTED, MODE_RANDOM, MODE_SCATTER, MODE_BOTH]
    assert [trial.seed for trial in first.trials] == [trial.seed for trial in second.trials]
    assert [trial.metrics.nrmse for trial in first.trials] == [trial.metrics.nrmse for trial in second.trials]
    # The same acquisition feeds every mode: truth counts match and seeds match.
    assert len({trial.truth_random for trial in first.trials}) == 1
    assert first.trials[0].truth_random > 0
    cancelled = run_correction_study(config, cancel=lambda: True)
    assert cancelled.cancelled
    assert cancelled.trials == []


def test_energy_scatter_tracks_compton_counts_not_randoms():
    """Side-window transfer follows photopeak scatter. Labels are assertions only.

    Scale-free scores stay near the uncorrected or random-corrected values
    because small-angle photopeak scatter is not a separate sinogram shape.
    """
    phantom = PhantomState()
    config = ReconstructionConfig(
        method="fbp",
        image_size=48,
        n_theta=32,
        n_s=48,
        filter_name="hann",
        smooth_sigma_s=1.0,
        smooth_sigma_theta=0.5,
    ).clamp()

    def acquire(seed: int, scatter: float, random: float):
        engine = PETEngine(BrainPhantom(phantom), DetectorRing(48), seed=seed)
        physics = DetectorPhysicsConfig(
            enabled=True,
            random_fraction=random,
            scatter_probability=scatter,
            energy_fwhm_percent=12.0,
            timing_fwhm_ps=200.0,
            timing_window_ps=8000.0,
        )
        acquired = acquire_matched(engine, 1800, physics, detector_efficiency=1.0)
        assert acquired is not None
        return acquired, physics

    def estimates(acquired, physics, *, correct_random: bool, correct_scatter: bool):
        measured = build_sinogram(
            acquired.hits_a,
            acquired.hits_b,
            n_theta=config.n_theta,
            n_s=config.n_s,
            s_max=config.s_max,
        )
        scale = acquired.hits_a.shape[0] / max(acquired.n_accepted, 1)
        return apply_sinogram_corrections(
            measured.data,
            correct_random=correct_random,
            correct_scatter=correct_scatter,
            n_annihilations=acquired.n_annihilations,
            random_fraction=physics.random_fraction * scale,
            s_edges=measured.s_edges,
            s_max=config.s_max,
            n_side_window=acquired.n_side_window,
            scatter_probability=physics.scatter_probability,
            photo_low_kev=physics.energy_window_low_kev,
            photo_high_kev=physics.energy_window_high_kev,
            energy_fwhm_percent=physics.energy_fwhm_percent,
            scatter_angle_sigma_rad=physics.scatter_angle_sigma_rad,
        )

    def metrics_for(acquired, physics, mode: str):
        scale = acquired.hits_a.shape[0] / max(acquired.n_accepted, 1)
        result = reconstruct_corrected(
            acquired.hits_a,
            acquired.hits_b,
            config,
            mode=mode,
            n_annihilations=acquired.n_annihilations,
            random_fraction=physics.random_fraction * scale,
            n_side_window=acquired.n_side_window,
            scatter_probability=physics.scatter_probability,
            photo_low_kev=physics.energy_window_low_kev,
            photo_high_kev=physics.energy_window_high_kev,
            energy_fwhm_percent=physics.energy_fwhm_percent,
            scatter_angle_sigma_rad=physics.scatter_angle_sigma_rad,
        )
        return score_reconstruction(result, phantom), result

    # Scatter only: the estimate follows the Compton count and ignores a zero random rate.
    for seed, scatter in ((11, 0.25), (11, 0.55), (29, 0.45)):
        acquired, physics = acquire(seed, scatter, 0.0)
        assert acquired.truth_random == 0
        assert acquired.truth_scattered > 300
        assert acquired.n_side_window > 20
        _corrected, random_est, scatter_est = estimates(
            acquired, physics, correct_random=True, correct_scatter=True
        )
        assert random_est.sum() == 0.0
        assert abs(scatter_est.sum() - acquired.truth_scattered) / acquired.truth_scattered < 0.30
        plain, _plain_result = metrics_for(acquired, physics, MODE_UNCORRECTED)
        scattered, _scatter_result = metrics_for(acquired, physics, MODE_SCATTER)
        assert abs(scattered.nrmse - plain.nrmse) < 0.02
        assert abs((scattered.crc or 0.0) - (plain.crc or 0.0)) < 0.02
        assert abs((scattered.background_cv or 0.0) - (plain.background_cv or 0.0)) < 0.02
        if plain.localization_error is not None and scattered.localization_error is not None:
            assert abs(scattered.localization_error - plain.localization_error) < 0.02

    # Randoms only: witnesses stay empty, so scatter correction does not copy the random estimate.
    for seed, random in ((11, 0.35), (29, 0.55)):
        acquired, physics = acquire(seed, 0.0, random)
        assert acquired.truth_scattered == 0
        assert acquired.truth_random > 200
        assert acquired.n_side_window <= 8
        _corrected, random_est, scatter_est = estimates(
            acquired, physics, correct_random=False, correct_scatter=True
        )
        _, random_only, _scatter_off = estimates(
            acquired, physics, correct_random=True, correct_scatter=False
        )
        assert scatter_est.sum() < 0.05 * max(acquired.truth_random, 1)
        assert abs(random_only.sum() - acquired.truth_random) / acquired.truth_random < 0.25

    # Both sources: scatter estimate tracks Compton counts, random estimate tracks randoms.
    for seed in (11, 29):
        acquired, physics = acquire(seed, 0.45, 0.35)
        assert acquired.truth_scattered > 400
        assert acquired.truth_random > 400
        _corrected, random_est, scatter_est = estimates(
            acquired, physics, correct_random=True, correct_scatter=True
        )
        assert abs(random_est.sum() - acquired.truth_random) / acquired.truth_random < 0.25
        assert abs(scatter_est.sum() - acquired.truth_scattered) / acquired.truth_scattered < 0.35
        uncorrected, _u = metrics_for(acquired, physics, MODE_UNCORRECTED)
        random_metrics, _r = metrics_for(acquired, physics, MODE_RANDOM)
        scatter_metrics, scatter_result = metrics_for(acquired, physics, MODE_SCATTER)
        both, both_result = metrics_for(acquired, physics, MODE_BOTH)
        assert scatter_result.scatter_estimate_counts > 0.0
        assert both_result.random_estimate_counts > 0.0
        assert both_result.scatter_estimate_counts > 0.0
        for metrics in (uncorrected, random_metrics, scatter_metrics, both):
            assert metrics.nrmse is not None and np.isfinite(metrics.nrmse)
            assert metrics.background_cv is None or np.isfinite(metrics.background_cv)
            assert metrics.crc is None or np.isfinite(metrics.crc)
        # Scatter-only does not become the random correction.
        assert abs(scatter_metrics.nrmse - uncorrected.nrmse) < 0.02
        assert abs(both.nrmse - random_metrics.nrmse) < 0.05


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    a = np.asarray(left, dtype=np.float64).ravel()
    b = np.asarray(right, dtype=np.float64).ravel()
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= 0.0:
        return 0.0
    return float(np.dot(a, b) / denom)


def test_scatter_estimate_is_a_uniform_scale_of_the_prompts():
    """Photopeak scatter has the prompt shape. The side window only sets the count.

    Ground-truth labels build the comparison sinograms. They are not passed
    into ``estimate_scatter_sinogram``.
    """
    uneven = np.tile(np.linspace(0.2, 3.0, 12), (6, 1))
    estimate = estimate_scatter_sinogram(
        uneven,
        n_side_window=8,
        scatter_probability=0.35,
        photo_low_kev=435.0,
        photo_high_kev=650.0,
    )
    fraction = float(estimate.sum() / uneven.sum())
    assert 0.0 < fraction < 1.0
    assert np.allclose(estimate, uneven * fraction)

    def photo(event) -> bool:
        return (
            event.detected_a
            and event.detected_b
            and event.passed_timing
            and 435.0 <= event.energy_a_kev <= 650.0
            and 435.0 <= event.energy_b_kev <= 650.0
        )

    def witness(event) -> bool:
        if not (event.detected_a and event.detected_b and event.passed_timing):
            return False
        energies = (event.energy_a_kev, event.energy_b_kev)
        photo_flags = [435.0 <= energy <= 650.0 for energy in energies]
        side_flags = [300.0 <= energy <= 420.0 for energy in energies]
        return photo_flags[0] and side_flags[1] or photo_flags[1] and side_flags[0]

    def sinogram_of(events) -> np.ndarray:
        if not events:
            return np.zeros((24, 32))
        hits_a = np.array([event.hit_a_xy for event in events], dtype=np.float64)
        hits_b = np.array([event.hit_b_xy for event in events], dtype=np.float64)
        return build_sinogram(hits_a, hits_b, n_theta=24, n_s=32, s_max=0.92).data

    for seed, scatter in ((11, 0.25), (29, 0.55), (7, 0.7)):
        engine = PETEngine(BrainPhantom(PhantomState()), DetectorRing(48), seed=seed)
        physics = DetectorPhysicsConfig(
            enabled=True,
            scatter_probability=scatter,
            random_fraction=0.0,
            energy_fwhm_percent=12.0,
            timing_fwhm_ps=200.0,
            timing_window_ps=8000.0,
        )
        events = []
        while sum(1 for event in events if not event.is_random) < 4000:
            events.extend(engine.simulate_batch(400, detector_efficiency=1.0, physics=physics))
        prompts = [event for event in events if photo(event) and not event.is_random]
        scattered = [event for event in prompts if event.is_scattered]
        witnesses = [event for event in events if witness(event)]
        prompt_sino = sinogram_of(prompts)
        scatter_sino = sinogram_of(scattered)
        witness_sino = sinogram_of(witnesses)
        assert _cosine(scatter_sino, prompt_sino) > _cosine(scatter_sino, witness_sino) + 0.10
        corrected, _random_est, scatter_est = apply_sinogram_corrections(
            prompt_sino,
            correct_random=False,
            correct_scatter=True,
            n_annihilations=4000,
            random_fraction=0.0,
            s_edges=np.linspace(-0.92, 0.92, prompt_sino.shape[1] + 1),
            n_side_window=len(witnesses),
            scatter_probability=scatter,
            photo_low_kev=435.0,
            photo_high_kev=650.0,
        )
        positive = prompt_sino > 0
        ratio = corrected[positive] / prompt_sino[positive]
        assert np.allclose(ratio, ratio.flat[0])
        assert scatter_est.sum() > 0.0


def test_scored_reconstruction_does_not_require_labels():
    phantom = PhantomState()
    angles = np.linspace(0, 2 * np.pi, 20, endpoint=False)
    hits_a = np.column_stack((0.92 * np.cos(angles), 0.92 * np.sin(angles)))
    hits_b = -hits_a
    result = reconstruct_corrected(
        hits_a,
        hits_b,
        ReconstructionConfig(method="fbp", image_size=32, n_theta=16, n_s=32, smooth_sigma_s=0.0, smooth_sigma_theta=0.0).clamp(),
        mode=MODE_UNCORRECTED,
        n_annihilations=20,
        random_fraction=0.0,
    )
    metrics = score_reconstruction(result, phantom)
    assert metrics.runtime_s >= 0.0
    assert metrics.nrmse is not None


def test_corrections_tab_comparison_and_reset(qapp):
    window = MainWindow()
    assert window.corrections.comparison_method() == "fbp"
    assert not window.detector_physics.current_config().enabled
    window.acquisition.apply_config(
        window.acquisition.config.__class__(
            n_events=1000,
            detector_efficiency=1.0,
            animation_speed="Fast",
            use_seed=True,
            seed=4,
        )
    )
    window.acquisition.engine.set_seed(4)
    window.acquisition.start()
    window.acquisition.pause()
    window.acquisition._stats.running = True
    window.acquisition._stats.paused = False
    for _ in range(2):
        window.acquisition._on_tick()
    window.acquisition.pause()
    window.corrections._iterations.setCurrentText("1")
    window._on_compare_corrections()
    deadline = time.time() + 40.0
    while window.correction_runner.busy and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    qapp.processEvents()
    assert not window.correction_runner.busy
    status = window.corrections._compare_status.text()
    assert "AttributeError" not in status
    assert "Comparison finished." in status
    titles = [ax.get_title() for ax in window.corrections.comparison_canvas._axes]
    assert any(title.startswith("Ideal") for title in titles)
    assert "Uncorrected" in titles
    assert "Random correction" in titles
    assert "Scatter scaling" in titles
    assert window.corrections._table.rowCount() == 5
    assert window.corrections._table.horizontalHeaderItem(9).text() == "Random counts"
    assert window.corrections._table.horizontalHeaderItem(10).text() == "Scatter counts"
    assert "spatial template" in window.corrections._table.horizontalHeaderItem(10).toolTip()
    window.corrections.reset_view()
    assert window.corrections._table.rowCount() == 0
    assert window.corrections._compare.isEnabled()
    window.close()
