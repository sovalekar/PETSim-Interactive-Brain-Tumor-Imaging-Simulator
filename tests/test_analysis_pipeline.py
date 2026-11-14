"""End-to-end Stage 4 analysis + GUI invalidation tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.pipeline import analyze_reconstruction
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_from_endpoints,
)
from petsim.widgets.reconstruction_panel import ReconstructionPanel


def _acquire_and_reconstruct(seed: int = 21, n_events: int = 12000):
    state = PhantomState(
        tumor_enabled=True,
        tumor_x=0.18,
        tumor_y=0.12,
        tumor_radius=0.10,
        tumor_intensity=1.0,
        brain_radius=0.55,
        brain_intensity=0.35,
    ).clamp()
    phantom = BrainPhantom(state)
    engine = PETEngine(phantom, DetectorRing(48))
    engine.set_seed(seed)
    events = engine.simulate_batch(n_events, detector_efficiency=0.9)
    buf = AcquisitionBuffer()
    buf.extend(events)
    hits_a, hits_b = buf.coincidence_endpoints()
    result = reconstruct_from_endpoints(
        hits_a,
        hits_b,
        ReconstructionConfig(
            filter_name="hann",
            n_theta=64,
            n_s=96,
            image_size=96,
        ),
    )
    return result, state, hits_a.shape[0]


def test_analyze_reconstruction_reports_true_center():
    result, state, n = _acquire_and_reconstruct()
    assert n > 500
    analysis = analyze_reconstruction(result, state)
    assert analysis.localization.true_xy == (state.tumor_x, state.tumor_y)
    assert analysis.localization.reliable
    assert analysis.localization.estimated_xy is not None
    assert analysis.localization.error is not None
    # With ~thousands of coincidences, localization should be in the right quadrant.
    assert analysis.localization.error < 0.35
    assert analysis.contrast.valid
    assert analysis.contrast.tbr is not None and analysis.contrast.tbr > 0
    assert analysis.rois is not None
    assert analysis.rois.tumor_n_pixels > 0


def test_analyze_disabled_tumor():
    result, state, _ = _acquire_and_reconstruct(seed=5, n_events=8000)
    state = PhantomState(
        tumor_enabled=False,
        brain_radius=state.brain_radius,
        brain_intensity=state.brain_intensity,
    ).clamp()
    analysis = analyze_reconstruction(result, state)
    assert not analysis.localization.reliable
    assert not analysis.contrast.valid
    assert "disabled" in analysis.message.lower() or "disabled" in analysis.localization.message.lower()


def test_analyze_missing_phantom_state():
    result, _, _ = _acquire_and_reconstruct(seed=9, n_events=6000)
    analysis = analyze_reconstruction(result, None)
    assert not analysis.localization.reliable
    assert not analysis.contrast.valid


def test_panel_invalidation_clears_analysis(qapp):
    panel = ReconstructionPanel()
    result, state, _ = _acquire_and_reconstruct(seed=4, n_events=6000)
    panel.show_result(result)
    analysis = analyze_reconstruction(result, state)
    panel.show_analysis(analysis)
    assert panel._analysis is not None
    assert panel._loc_true.text() != "—"

    panel.invalidate("Acquisition data cleared.")
    assert panel._result is None
    assert panel._analysis is None
    assert panel._loc_true.text() == "—"
    assert panel._c_tbr.text() == "—"
    assert "cleared" in panel._analysis_status.text().lower() or "cleared" in panel._status.text().lower()


def test_panel_clear_output_resets_metrics(qapp):
    panel = ReconstructionPanel()
    result, state, _ = _acquire_and_reconstruct(seed=8, n_events=5000)
    panel.show_result(result)
    panel.show_analysis(analyze_reconstruction(result, state))
    panel.clear_output()
    assert panel._analysis is None
    assert panel._comparison is None


def _synthetic_result():
    from petsim.reconstruction.sinogram import Sinogram

    sino = Sinogram(
        data=np.ones((4, 8), dtype=np.float64),
        theta_edges=np.linspace(0.0, np.pi, 5),
        s_edges=np.linspace(-1.0, 1.0, 9),
        n_events=10,
    )
    image = np.linspace(0.0, 1.0, 32 * 32, dtype=np.float64).reshape(32, 32)
    return ReconstructionResult(
        sinogram=sino,
        image=image,
        config=ReconstructionConfig(image_size=32),
        n_coincidences=10,
        message="Synthetic FBP",
    )


def _synthetic_comparison():
    from petsim.analysis.count_comparison import CountComparisonResult, CountLevelMetrics

    levels = [
        CountLevelMetrics(1000, 1000, True, 0.12, 2.1, 0.80, "OK"),
        CountLevelMetrics(3000, 3000, True, 0.08, 2.4, 0.55, "OK"),
        CountLevelMetrics(5000, 5000, True, 0.05, 2.6, 0.40, "OK"),
        CountLevelMetrics(10000, 4000, False, None, None, None, "Insufficient coincidences."),
    ]
    return CountComparisonResult(
        levels=levels,
        total_available=4000,
        message="Compared nested prefixes.",
    )


def test_reconstruction_and_count_canvases_are_independent(qapp):
    panel = ReconstructionPanel()
    assert panel.canvas is not panel.count_canvas
    assert panel.canvas.figure is not panel.count_canvas.figure
    labels = [panel._display.itemText(i) for i in range(panel._display.count())]
    assert "Count Comparison Plot" not in labels
    assert "Reconstructed Image" in labels

    result = _synthetic_result()
    panel.show_result(result)
    recon_title = panel.canvas.figure.axes[0].get_title()
    assert "FBP" in recon_title

    panel.show_count_comparison(_synthetic_comparison())
    assert panel.canvas.figure.axes[0].get_title() == recon_title
    count_title = panel.count_canvas.figure.axes[0].get_title()
    assert "coefficient of variation" in count_title.lower()
    assert any(
        abs(line.get_xdata()[0] - 1000) < 1e-6
        for line in panel.count_canvas.figure.axes[0].get_lines()
    )

    panel._cmap.setCurrentText("gray")
    assert panel.canvas.figure.axes[0].get_title() == recon_title
    assert panel.count_canvas.figure.axes[0].get_title() == count_title

    panel.show_result(result)
    assert "FBP" in panel.canvas.figure.axes[0].get_title()
    assert panel.count_canvas.figure.axes[0].get_title() == count_title
    assert panel._comparison is not None

    panel.invalidate("Acquisition data cleared.")
    assert panel._comparison is None
    assert panel._result is None
    placeholder = " ".join(
        text.get_text() for text in panel.count_canvas.figure.axes[0].texts
    )
    assert "Count Comparison" in placeholder
    recon_placeholder = " ".join(
        text.get_text() for text in panel.canvas.figure.axes[0].texts
    )
    assert "Acquisition data cleared" in recon_placeholder


def test_filter_change_clears_stale_metrics(qapp):
    panel = ReconstructionPanel()
    result, state, _ = _acquire_and_reconstruct(seed=2, n_events=5000)
    panel.show_result(result)
    panel.show_analysis(analyze_reconstruction(result, state))
    assert panel._analysis is not None
    panel._filter.setCurrentText("Ramp")
    assert panel._analysis is None
