"""Stage 8 correction estimates, metrics, studies, and GUI."""

from __future__ import annotations

import inspect
import time

import numpy as np

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
from petsim.corrections.studies import CorrectionStudyConfig, run_correction_study
from petsim.main_window import MainWindow
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
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


def test_scatter_estimate_is_nonnegative_and_fits_a_flat_tail():
    edges = np.linspace(-0.92, 0.92, 49)
    centers = 0.5 * (edges[:-1] + edges[1:])
    profile = np.where(np.abs(centers) < 0.4, 8.0, 1.0)
    measured = np.tile(profile, (16, 1))
    estimate = estimate_scatter_sinogram(measured, edges, sigma_s_bins=1.5, sigma_theta_bins=0.0)
    assert estimate.shape == measured.shape
    assert np.all(estimate >= 0.0)
    tail = np.abs(centers) >= 0.78
    assert abs(estimate[:, tail].sum() - measured[:, tail].sum()) / measured[:, tail].sum() < 0.05
    assert estimate_scatter_sinogram(np.zeros_like(measured), edges).sum() == 0.0


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
    for method in ("fbp", "mlem"):
        result = reconstruct_corrected(
            hits_a,
            hits_b,
            ReconstructionConfig(method=method, n_iterations=2, **shared).clamp(),
            mode=MODE_BOTH,
            n_annihilations=40,
            random_fraction=0.3,
        )
        assert result.image.shape == (32, 32)
        assert np.isfinite(result.image).all()
        assert result.config.method == method
        assert np.all(result.sinogram.data >= -1e-9)


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
    assert any("uncorrected" in title for title in titles)
    assert window.corrections._table.rowCount() == 5
    window.corrections.reset_view()
    assert window.corrections._table.rowCount() == 0
    assert window.corrections._compare.isEnabled()
    window.close()
