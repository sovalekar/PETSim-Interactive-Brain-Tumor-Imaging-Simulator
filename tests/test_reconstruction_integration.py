"""Integration: Stage 2 coincidences → Stage 3 FBP, tumor sensitivity, UI state."""

from __future__ import annotations

import time

import numpy as np
import pytest

pytest.importorskip("PySide6")

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionConfig
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from tests.qt_subprocess import run_qt_script


def _endpoints_from_engine(state: PhantomState, seed: int, n: int = 8000):
    eng = PETEngine(BrainPhantom(state=state), DetectorRing(), seed=seed)
    events = eng.simulate_batch(n, detector_efficiency=1.0)
    ha = np.array([e.hit_a_xy for e in events if e.is_coincidence], dtype=np.float64)
    hb = np.array([e.hit_b_xy for e in events if e.is_coincidence], dtype=np.float64)
    return ha, hb


def test_reconstruction_uses_stage2_coincidences_not_truth():
    state = PhantomState(tumor_enabled=True, tumor_x=0.2, tumor_y=0.1)
    ha, hb = _endpoints_from_engine(state, seed=7, n=6000)
    assert ha.shape[0] > 100
    result = reconstruct_from_endpoints(
        ha, hb, ReconstructionConfig(filter_name="hann", image_size=128)
    )
    assert result.has_data
    assert result.image.shape == (128, 128)
    assert np.all(np.isfinite(result.image))
    assert result.sinogram.data.sum() == pytest.approx(float(ha.shape[0]), rel=0, abs=1.0)


def test_tumor_location_sensitivity():
    """Moving the tumor should move the reconstructed hot spot."""
    left = PhantomState(
        brain_intensity=0.15,
        tumor_enabled=True,
        tumor_x=-0.25,
        tumor_y=0.0,
        tumor_radius=0.12,
        tumor_intensity=2.0,
    )
    right = PhantomState(
        brain_intensity=0.15,
        tumor_enabled=True,
        tumor_x=0.25,
        tumor_y=0.0,
        tumor_radius=0.12,
        tumor_intensity=2.0,
    )
    cfg = ReconstructionConfig(filter_name="hann", image_size=128, n_theta=180, n_s=128)
    r_left = reconstruct_from_endpoints(*_endpoints_from_engine(left, 11, 12000), cfg)
    r_right = reconstruct_from_endpoints(*_endpoints_from_engine(right, 11, 12000), cfg)

    axis = np.linspace(-1.0, 1.0, 128)
    # Restrict peak search to a band around y≈0 to reduce streak noise influence.
    band = slice(54, 74)
    ix_l = int(np.argmax(r_left.image_display[band, :].max(axis=0)))
    ix_r = int(np.argmax(r_right.image_display[band, :].max(axis=0)))
    assert axis[ix_l] < -0.05
    assert axis[ix_r] > 0.05
    assert axis[ix_r] > axis[ix_l]


def test_fixed_seed_pipeline_deterministic():
    state = PhantomState()
    a1, b1 = _endpoints_from_engine(state, seed=99, n=4000)
    a2, b2 = _endpoints_from_engine(state, seed=99, n=4000)
    assert np.allclose(a1, a2)
    r1 = reconstruct_from_endpoints(a1, b1, ReconstructionConfig(filter_name="ramp"))
    r2 = reconstruct_from_endpoints(a2, b2, ReconstructionConfig(filter_name="ramp"))
    assert np.allclose(r1.sinogram.data, r2.sinogram.data)
    assert np.allclose(r1.image, r2.image)


def test_controller_endpoints_exclude_non_coincidences(qapp, acquisition_controller):
    ctrl = acquisition_controller
    ctrl.set_phantom_state(PhantomState())
    ctrl.apply_config(
        AcquisitionConfig(
            n_events=1000,
            detector_efficiency=0.5,
            animation_speed="Fast",
            use_seed=True,
            seed=3,
        )
    )
    ctrl.start()
    deadline = time.time() + 5.0
    while not ctrl.stats.finished and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)
    ha, hb = ctrl.coincidence_endpoints()
    assert ha.shape[0] == ctrl.stats.coincidences
    assert ha.shape == hb.shape
    assert ctrl.acquired_activity is not None


def test_main_window_recon_state_management():
    """Stage 3 UI/state checks in a child process (macOS abort isolation)."""
    script = r"""
import time
import sys
from PySide6.QtWidgets import QApplication
app = QApplication([sys.argv[0] if sys.argv else 'petsim-pytest'])
app.setQuitOnLastWindowClosed(False)
from petsim.main_window import MainWindow
window = MainWindow()
app.processEvents()
assert window.reconstruction._run_btn.isEnabled() is False
window.controls._n_events.setValue(1000)
window.controls._efficiency.setValue(100)
window.controls._speed.setCurrentText('Fast')
window.controls._use_seed.setChecked(True)
window.controls._seed_edit.setText('5')
app.processEvents()
window.controls.start_acquisition.emit()
deadline = time.time() + 5.0
while not window.acquisition.stats.finished and time.time() < deadline:
    app.processEvents()
    time.sleep(0.005)
assert window.acquisition.stats.coincidences > 0
assert window.reconstruction._run_btn.isEnabled() is True
window.reconstruction._on_run()
deadline = time.time() + 10.0
while window.recon_runner.busy and time.time() < deadline:
    app.processEvents()
    time.sleep(0.01)
assert window.recon_runner.shutdown(timeout_ms=5000)
app.processEvents()
assert window.reconstruction._result is not None
assert window.reconstruction._result.image.shape == (128, 128)
window.controls._tumor_x.setValue(30)
app.processEvents()
assert window.acquisition.stats.total_events == 0
assert window.reconstruction._result is None
assert window.reconstruction._run_btn.isEnabled() is False
window.close()
app.processEvents()
"""
    result = run_qt_script(script)
    assert result.ok, (
        f"GUI subprocess failed (code={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
