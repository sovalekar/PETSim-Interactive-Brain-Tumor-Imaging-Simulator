"""Stage 5 MLEM, projector adjoint, and reconstruction-panel state tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.count_comparison import CountComparisonResult, CountLevelMetrics
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.mlem import mlem_reconstruct
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.reconstruction.projectors import backproject, forward_project, sensitivity_map
from petsim.reconstruction.sinogram import Sinogram
from petsim.reconstruction_worker import ReconstructionRunner
from petsim.widgets.reconstruction_panel import ReconstructionPanel


def _sinogram(data: np.ndarray, *, s_max: float = 1.0) -> Sinogram:
    n_theta, n_s = data.shape
    return Sinogram(
        data=np.asarray(data, dtype=np.float64),
        theta_edges=np.linspace(0.0, np.pi, n_theta + 1),
        s_edges=np.linspace(-s_max, s_max, n_s + 1),
        n_events=int(np.sum(data)),
    )


def _disk(n: int, center: tuple[float, float], radius: float, fov: float = 1.0) -> np.ndarray:
    axis = np.linspace(-fov, fov, n)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return ((xx - center[0]) ** 2 + (yy - center[1]) ** 2 <= radius**2).astype(np.float64)


def test_adjoint_consistency():
    rng = np.random.default_rng(1)
    n, n_theta, n_s = 12, 8, 16
    sino = _sinogram(np.zeros((n_theta, n_s)))
    image = rng.random((n, n))
    projection = rng.random((n_theta, n_s))
    forward = forward_project(image, sino.theta_centers, sino.s_centers, fov=1.0)
    backward = backproject(
        projection, sino.theta_centers, sino.s_centers, image_size=n, fov=1.0
    )
    lhs = float(np.sum(forward * projection))
    rhs = float(np.sum(image * backward))
    assert abs(lhs - rhs) <= 1e-8 * max(abs(lhs), 1.0)


def test_mlem_stays_nonnegative_and_finite():
    sino = _sinogram(np.ones((6, 12)))
    image, history, cancelled = mlem_reconstruct(
        sino, n_iterations=4, image_size=24, fov=1.0
    )
    assert not cancelled
    assert image.shape == (24, 24)
    assert np.all(np.isfinite(image))
    assert np.min(image) >= 0.0
    assert len(history.iterations) == 4


def test_zero_count_bins_do_not_produce_nans():
    data = np.zeros((8, 18))
    data[:, 9] = 5.0
    image, history, _ = mlem_reconstruct(
        _sinogram(data), n_iterations=3, image_size=20, fov=1.0
    )
    assert np.all(np.isfinite(image))
    assert np.min(image) >= 0.0
    assert np.all(np.isfinite(history.log_likelihood) | np.isneginf(history.log_likelihood))


def test_zero_sensitivity_pixels_stay_zero():
    # A single near-horizontal projection whose s bins lie on +x does not see
    # the far corner x = y = -1, so that pixel has zero sensitivity.
    data = np.zeros((1, 8))
    data[0, 3] = 6.0
    sino = Sinogram(
        data=data,
        theta_edges=np.array([0.0, 1e-3]),
        s_edges=np.linspace(0.45, 0.9, 9),
        n_events=6,
    )
    sens = sensitivity_map(
        sino.theta_centers, sino.s_centers, image_size=24, fov=1.0
    )
    assert sens[0, 0] == 0.0
    assert np.any(sens > 0.0)
    image, _, _ = mlem_reconstruct(sino, n_iterations=4, image_size=24, fov=1.0)
    assert image[0, 0] == 0.0
    assert np.all(np.isfinite(image))
    assert np.min(image) >= 0.0


def test_mlem_output_grid_matches_fov_coordinates():
    n = 21
    true_xy = (0.2, -0.1)
    phantom = _disk(n, true_xy, 0.18)
    blank = _sinogram(np.zeros((10, 24)), s_max=1.0)
    measured = forward_project(phantom, blank.theta_centers, blank.s_centers, fov=1.0)
    sino = _sinogram(measured, s_max=1.0)
    image, _, _ = mlem_reconstruct(sino, n_iterations=8, image_size=n, fov=1.0)
    assert image.shape == (n, n)
    axis = np.linspace(-1.0, 1.0, n)
    peak = np.unravel_index(int(np.argmax(image)), image.shape)
    assert abs(axis[peak[1]] - true_xy[0]) < 0.15
    assert abs(axis[peak[0]] - true_xy[1]) < 0.15


def test_mlem_is_deterministic():
    sino = _sinogram(np.linspace(0.0, 3.0, 6 * 14).reshape(6, 14))
    first, hist_a, _ = mlem_reconstruct(sino, n_iterations=3, image_size=16)
    second, hist_b, _ = mlem_reconstruct(sino, n_iterations=3, image_size=16)
    np.testing.assert_allclose(first, second)
    np.testing.assert_allclose(hist_a.log_likelihood, hist_b.log_likelihood)


def test_mlem_likelihood_increases_on_consistent_disk():
    phantom = _disk(20, (0.05, 0.1), 0.22)
    blank = _sinogram(np.zeros((8, 20)))
    measured = forward_project(phantom, blank.theta_centers, blank.s_centers, fov=1.0)
    _, history, cancelled = mlem_reconstruct(
        _sinogram(measured), n_iterations=6, image_size=20, fov=1.0
    )
    assert not cancelled
    ll = history.log_likelihood
    assert np.all(np.isfinite(ll))
    assert ll[-1] > ll[0]


def test_cancel_stops_between_iterations():
    sino = _sinogram(np.ones((4, 10)))
    seen = {"n": 0}

    def cancel() -> bool:
        seen["n"] += 1
        return seen["n"] > 2

    image, history, cancelled = mlem_reconstruct(
        sino, n_iterations=12, image_size=16, cancel=cancel
    )
    assert cancelled
    assert len(history.iterations) == 2
    assert np.min(image) >= 0.0


def test_fbp_and_mlem_use_the_same_measured_counts_when_unsmoothed():
    state = PhantomState(tumor_enabled=True, tumor_x=0.15, tumor_y=-0.05).clamp()
    engine = PETEngine(BrainPhantom(state), DetectorRing(48))
    engine.set_seed(4)
    buf = AcquisitionBuffer()
    buf.extend(engine.simulate_batch(2500, detector_efficiency=0.8))
    hits_a, hits_b = buf.coincidence_endpoints()
    assert hits_a.shape[0] > 100
    cfg = ReconstructionConfig(
        n_theta=24,
        n_s=32,
        image_size=32,
        n_iterations=3,
        smooth_sigma_s=0.0,
        smooth_sigma_theta=0.0,
    )
    fbp = reconstruct_from_endpoints(hits_a, hits_b, cfg.as_method("fbp"))
    mlem = reconstruct_from_endpoints(hits_a, hits_b, cfg.as_method("mlem"))
    np.testing.assert_allclose(fbp.sinogram.data, mlem.sinogram.data)
    assert fbp.image.shape == mlem.image.shape == (32, 32)
    assert np.min(mlem.image) >= 0.0
    assert mlem.history is not None and len(mlem.history.iterations) == 3
    assert fbp.runtime_s >= 0.0 and mlem.runtime_s >= 0.0


def test_runner_drops_stale_job(qapp):
    runner = ReconstructionRunner()
    runner._generation = 5
    delivered = []
    runner.finished.connect(delivered.append)
    from petsim.reconstruction.pipeline import ReconstructionResult

    stale = ReconstructionResult(
        sinogram=_sinogram(np.ones((2, 4))),
        image=np.ones((8, 8)),
        config=ReconstructionConfig(image_size=8),
        n_coincidences=3,
        job_id=4,
    )
    runner._on_finished(stale)
    assert delivered == []
    current = ReconstructionResult(
        sinogram=stale.sinogram,
        image=stale.image,
        config=stale.config,
        n_coincidences=3,
        job_id=5,
    )
    runner._on_finished(current)
    assert delivered == [current]


def test_gui_method_switch_and_independent_canvases(qapp):
    panel = ReconstructionPanel()
    # isHidden follows setVisible even when the parent window is not shown.
    assert not panel._filter_row.isHidden()
    assert panel._iter_row.isHidden()
    assert panel.current_config().method == "fbp"
    panel._method.setCurrentIndex(1)
    assert not panel._iter_row.isHidden()
    assert panel._filter_row.isHidden()
    assert panel.current_config().method == "mlem"
    assert panel.current_config().n_iterations == 10
    assert panel._iterations.currentText() == "10"
    figures = {
        panel.canvas.figure,
        panel.count_canvas.figure,
        panel.convergence_canvas.figure,
    }
    assert len(figures) == 3

    from petsim.reconstruction.pipeline import ReconstructionResult

    image = np.ones((16, 16))
    sino = _sinogram(np.ones((4, 8)))
    fbp = ReconstructionResult(
        sinogram=sino,
        image=image,
        config=ReconstructionConfig(image_size=16, method="fbp"),
        n_coincidences=4,
        message="fbp",
    )
    panel.show_result(fbp)
    recon_title = panel.canvas.figure.axes[0].get_title()
    count_title = panel.count_canvas.figure.axes[0].get_title()
    panel.show_mlem_progress(1, 10, -12.0, 0.4)
    assert panel.canvas.figure.axes[0].get_title() == recon_title
    assert panel.count_canvas.figure.axes[0].get_title() == count_title
    assert "convergence" in panel.convergence_canvas.figure.axes[0].get_title().lower()

    panel.show_count_comparison(
        CountComparisonResult(
            levels=[CountLevelMetrics(1000, 1000, True, 0.1, 2.0, 0.4, "OK")],
            total_available=1000,
            message="counts",
        )
    )
    assert panel.canvas.figure.axes[0].get_title() == recon_title
    assert "convergence" in panel.convergence_canvas.figure.axes[0].get_title().lower()

    panel.invalidate("Acquisition data cleared.")
    assert panel._fbp_result is None
    assert panel._mlem_result is None
    cleared = " ".join(t.get_text() for t in panel.convergence_canvas.figure.axes[0].texts)
    assert "cleared" in cleared.lower()
