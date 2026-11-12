"""UI smoke tests: acquisition controller + responsive offscreen window."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("PySide6")

from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig
from tests.qt_subprocess import run_qt_script

# ``qapp`` / ``acquisition_controller`` come from conftest.py.
# ``MainWindow`` checks run in a subprocess (macOS native-abort isolation).


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


def test_main_window_stays_responsive_during_acquisition():
    """MainWindow + Matplotlib QtAgg isolated from the pytest process.

    Exact prior crash surface on macOS: fixture setup constructing MainWindow
    (FigureCanvasQTAgg) under QT_QPA_PLATFORM=offscreen → native abort of the
    whole pytest interpreter.  Child uses ``minimal`` on Darwin.
    """
    script = r"""
import time
import sys
from PySide6.QtWidgets import QApplication
app = QApplication([sys.argv[0] if sys.argv else 'petsim-pytest'])
app.setQuitOnLastWindowClosed(False)
from petsim.main_window import MainWindow
window = MainWindow()
app.processEvents()
window.controls._n_events.setValue(1000)
window.controls._efficiency.setValue(100)
window.controls._speed.setCurrentText('Fast')
window.controls._use_seed.setChecked(True)
window.controls._seed_edit.setText('7')
app.processEvents()
window.controls.start_acquisition.emit()
processed = 0
deadline = time.time() + 5.0
while not window.acquisition.stats.finished and time.time() < deadline:
    app.processEvents()
    processed += 1
    time.sleep(0.005)
assert window.acquisition.stats.finished, 'acquisition did not finish'
assert processed > 5, 'event loop did not pump'
assert window.acquisition.stats.total_events == 1000
assert window.acquisition.stats.coincidences == 1000
window.close()
app.processEvents()
"""
    result = run_qt_script(script)
    assert result.ok, (
        f"GUI subprocess failed (code={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )


def test_phantom_change_resets_acquisition():
    script = r"""
import time
import sys
from PySide6.QtWidgets import QApplication
app = QApplication([sys.argv[0] if sys.argv else 'petsim-pytest'])
app.setQuitOnLastWindowClosed(False)
from petsim.main_window import MainWindow
window = MainWindow()
app.processEvents()
window.controls._n_events.setValue(5000)
window.controls._speed.setCurrentText('Fast')
window.controls._efficiency.setValue(100)
window.controls.start_acquisition.emit()
deadline = time.time() + 2.0
while window.acquisition.stats.total_events < 25 and time.time() < deadline:
    app.processEvents()
    time.sleep(0.005)
assert window.acquisition.stats.total_events >= 25
window.controls._brain_radius.setValue(60)
app.processEvents()
assert window.acquisition.stats.total_events == 0
assert len(window.acquisition.buffer) == 0
assert not window.acquisition.stats.running
window.close()
app.processEvents()
"""
    result = run_qt_script(script)
    assert result.ok, (
        f"GUI subprocess failed (code={result.returncode}).\n"
        f"stdout:\n{result.stdout}\nstderr:\n{result.stderr}"
    )
