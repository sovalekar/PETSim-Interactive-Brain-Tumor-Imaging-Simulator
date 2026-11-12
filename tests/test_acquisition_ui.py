"""UI smoke tests: acquisition controller + responsive offscreen window."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig

# ``qapp``, ``main_window``, and ``acquisition_controller`` come from conftest.py
# (session-scoped QApplication + macOS-safe teardown).


def test_controller_batches_and_stats(qapp, acquisition_controller):
    ctrl = acquisition_controller
    ctrl.set_phantom_state(PhantomState())
    ctrl.apply_config(
        AcquisitionConfig(
            n_events=1000,  # UI/config floor is 1,000 events
            detector_efficiency=1.0,
            animation_speed="Fast",
            use_seed=True,
            seed=0,
        )
    )

    seen = []
    ctrl.events_simulated.connect(lambda ev: seen.append(len(ev)))
    ctrl.start()

    # Pump the Qt event loop until finished (should be quick in Fast mode).
    deadline = time.time() + 5.0
    while not ctrl.stats.finished and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.01)

    assert ctrl.stats.finished
    assert ctrl.stats.total_events == 1000
    assert ctrl.stats.coincidences == 1000  # ε = 1
    assert sum(seen) == 1000


def test_main_window_stays_responsive_during_acquisition(qapp, main_window):
    window = main_window
    window.show()
    qapp.processEvents()

    window.controls._n_events.setValue(1000)
    window.controls._efficiency.setValue(100)
    window.controls._speed.setCurrentText("Fast")
    window.controls._use_seed.setChecked(True)
    window.controls._seed_edit.setText("7")
    qapp.processEvents()

    window.controls.start_acquisition.emit()

    # While acquiring, the event loop must keep processing (responsiveness).
    processed = 0
    deadline = time.time() + 5.0
    while not window.acquisition.stats.finished and time.time() < deadline:
        qapp.processEvents()
        processed += 1
        time.sleep(0.005)

    assert window.acquisition.stats.finished
    assert processed > 5
    assert window.acquisition.stats.total_events == 1000
    assert window.acquisition.stats.coincidences == 1000


def test_phantom_change_resets_acquisition(qapp, main_window):
    window = main_window
    qapp.processEvents()
    window.controls._n_events.setValue(5000)
    window.controls._speed.setCurrentText("Fast")
    window.controls._efficiency.setValue(100)
    window.controls.start_acquisition.emit()

    # Let a few events accumulate.
    deadline = time.time() + 2.0
    while window.acquisition.stats.total_events < 25 and time.time() < deadline:
        qapp.processEvents()
        time.sleep(0.005)

    assert window.acquisition.stats.total_events >= 25
    # Changing a phantom slider resets acquisition data.
    window.controls._brain_radius.setValue(60)
    qapp.processEvents()
    assert window.acquisition.stats.total_events == 0
    assert len(window.acquisition.buffer) == 0
    assert not window.acquisition.stats.running
