"""End-to-end Stage 4 analysis + GUI invalidation tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.pipeline import analyze_reconstruction
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
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


def test_filter_change_clears_stale_metrics(qapp):
    panel = ReconstructionPanel()
    result, state, _ = _acquire_and_reconstruct(seed=2, n_events=5000)
    panel.show_result(result)
    panel.show_analysis(analyze_reconstruction(result, state))
    assert panel._analysis is not None
    panel._filter.setCurrentText("Ramp")
    assert panel._analysis is None
