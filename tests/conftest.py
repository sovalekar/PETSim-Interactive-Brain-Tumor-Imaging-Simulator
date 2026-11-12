"""Pytest fixtures for PySide6 — platform-safe Qt lifecycle.

macOS platform policy
---------------------
``QT_QPA_PLATFORM=offscreen`` and ``minimal`` are unreliable with PySide6 on
many macOS installs:

* ``offscreen`` can native-abort during ``QApplication`` setup.
* ``minimal`` may be present on disk yet fail to load (process abort /
  exit 134), even when ``cocoa`` works and the PETSim GUI launches.

Because the desktop app already runs under **cocoa**, pytest on Darwin
defaults to ``cocoa`` and remaps ``offscreen`` / ``minimal`` → ``cocoa``.
An explicit ``QT_QPA_PLATFORM=cocoa`` is never overridden.

Linux keeps ``offscreen`` as the headless default.

Also: stop live ``QThread``s after each test so ``~QThread`` cannot abort()
at GC, and tear down acquisition controllers / windows explicitly.
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
    """Choose a Qt platform plugin that can actually initialize.

    Returns the value written to ``QT_QPA_PLATFORM``.
    """
    requested = os.environ.get("QT_QPA_PLATFORM")

    if sys.platform == "darwin":
        # Prefer cocoa: it matches a working PETSim GUI on macOS.
        if requested in (None, ""):
            os.environ["QT_QPA_PLATFORM"] = "cocoa"
        elif requested in {"offscreen", "minimal"}:
            warnings.warn(
                f"QT_QPA_PLATFORM={requested} is unreliable on macOS with "
                "PySide6 (abort / plugin load failure). PETSim pytest is "
                "using 'cocoa' instead. Set QT_QPA_PLATFORM=cocoa explicitly "
                "to silence this warning.",
                UserWarning,
                stacklevel=1,
            )
            os.environ["QT_QPA_PLATFORM"] = "cocoa"
        # else: honor cocoa or any other explicit choice
        return os.environ["QT_QPA_PLATFORM"]

    # Linux / other: headless default for CI.
    if not requested:
        os.environ["QT_QPA_PLATFORM"] = "offscreen"
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
    config._petsim_qt_platform = _QT_PLATFORM  # type: ignore[attr-defined]
    config._petsim_qt_plugin = app.platformName()  # type: ignore[attr-defined]


def pytest_report_header(config):
    platform = getattr(config, "_petsim_qt_platform", os.environ.get("QT_QPA_PLATFORM"))
    plugin = getattr(config, "_petsim_qt_plugin", "?")
    return [f"petsim Qt: QT_QPA_PLATFORM={platform} (plugin={plugin})"]


def _stop_all_qthreads(timeout_ms: int = 5_000) -> None:
    """Quit any QThread still running so ~QThread cannot abort() the process."""
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
    """After every test, ensure no QThread is left running (abort guard)."""
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
