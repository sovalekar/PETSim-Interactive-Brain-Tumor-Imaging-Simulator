"""Pytest fixtures for PySide6 — macOS-safe Qt lifecycle.

Why macOS still aborted after the first fix
------------------------------------------
1. ``QT_QPA_PLATFORM=offscreen`` (as in the documented pytest command) is
   unstable on macOS with PySide6: ``QApplication`` / platform-plugin init
   can ``abort()`` natively during the first Qt fixture setup. Upstream CI
   practice is ``minimal`` on macOS and ``offscreen`` on Linux.

2. Destroying a still-running ``QThread`` calls ``abort()`` in Qt. GUI tests
   that touch ``ReconstructionRunner`` (or leave timers/threads alive) can
   therefore kill the interpreter at fixture teardown / GC / session exit —
   which looks like a crash "during pytest", not a failed assertion.

This conftest:
* remaps Darwin ``offscreen`` → ``minimal`` **before** importing PySide6
* creates one session ``QApplication`` with ``quitOnLastWindowClosed=False``
* stops every live ``QThread`` after each test
* tears down ``MainWindow`` / acquisition controllers explicitly
"""

from __future__ import annotations

import gc
import os
import sys
import warnings

# ---------------------------------------------------------------------------
# Platform selection MUST happen before any ``import PySide6``.
# ---------------------------------------------------------------------------


def _configure_qt_platform() -> str:
    if sys.platform == "darwin":
        requested = os.environ.get("QT_QPA_PLATFORM", "minimal")
        if requested == "offscreen":
            warnings.warn(
                "QT_QPA_PLATFORM=offscreen frequently native-aborts on macOS "
                "with PySide6 during QApplication fixture setup; "
                "PETSim pytest is using 'minimal' instead.",
                UserWarning,
                stacklevel=1,
            )
            os.environ["QT_QPA_PLATFORM"] = "minimal"
        else:
            os.environ.setdefault("QT_QPA_PLATFORM", "minimal")
    else:
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    return os.environ["QT_QPA_PLATFORM"]


_QT_PLATFORM = _configure_qt_platform()

import pytest

pytest.importorskip("PySide6")

from PySide6.QtCore import QThread  # noqa: E402
from PySide6.QtWidgets import QApplication  # noqa: E402


def pytest_configure(config):  # noqa: ARG001
    """Create the process-wide QApplication as early as possible."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0] if sys.argv else "petsim-pytest"])
    app.setQuitOnLastWindowClosed(False)
    # Expose for debugging local macOS runs.
    config._petsim_qt_platform = _QT_PLATFORM  # type: ignore[attr-defined]
    config._petsim_qt_plugin = app.platformName()  # type: ignore[attr-defined]


def pytest_report_header(config):
    platform = getattr(config, "_petsim_qt_platform", os.environ.get("QT_QPA_PLATFORM"))
    plugin = getattr(config, "_petsim_qt_plugin", "?")
    return [f"petsim Qt: QT_QPA_PLATFORM={platform} (plugin={plugin})"]


def _stop_all_qthreads(timeout_ms: int = 5_000) -> None:
    """Quit any QThread still running so ~QThread cannot abort() the process."""
    # Copy the list — quitting can mutate object graph during iteration.
    threads = []
    for obj in gc.get_objects():
        try:
            if isinstance(obj, QThread):
                threads.append(obj)
        except ReferenceError:
            continue
    for thread in threads:
        try:
            if thread.isRunning():
                thread.quit()
                thread.wait(timeout_ms)
        except RuntimeError:
            # Underlying C++ object already deleted.
            continue


@pytest.fixture(scope="session")
def qapp():
    """One process-wide QApplication for the entire pytest session."""
    app = QApplication.instance()
    if app is None:
        app = QApplication([sys.argv[0] if sys.argv else "petsim-pytest"])
        app.setQuitOnLastWindowClosed(False)
    yield app
    _stop_all_qthreads()
    app.processEvents()


@pytest.fixture(autouse=True)
def _qt_thread_cleanup():
    """After every test, ensure no QThread is left running (macOS abort guard)."""
    yield
    _stop_all_qthreads()
    app = QApplication.instance()
    if app is not None:
        app.processEvents()


@pytest.fixture
def main_window(qapp):
    """Create a MainWindow and tear it down cleanly after the test."""
    from petsim.main_window import MainWindow

    window = MainWindow()
    yield window
    _shutdown_main_window(window, qapp)


def _shutdown_main_window(window, qapp: QApplication) -> None:
    """Stop acquisition timers / recon threads, then destroy the window."""
    try:
        window.acquisition.reset()
    except Exception:  # noqa: BLE001
        pass
    try:
        window.recon_runner.shutdown(timeout_ms=10_000)
    except Exception:  # noqa: BLE001
        pass
    window.close()
    window.deleteLater()
    qapp.processEvents()
    _stop_all_qthreads()
    qapp.processEvents()


@pytest.fixture
def acquisition_controller(qapp):
    """Standalone AcquisitionController with guaranteed timer shutdown."""
    from petsim.acquisition_controller import AcquisitionController

    ctrl = AcquisitionController()
    yield ctrl
    try:
        ctrl.reset()
    except Exception:  # noqa: BLE001
        pass
    ctrl.deleteLater()
    qapp.processEvents()
