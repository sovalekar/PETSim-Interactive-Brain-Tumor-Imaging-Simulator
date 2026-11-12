"""Pytest fixtures for PySide6 — macOS-safe Qt lifecycle.

Root cause this addresses
-------------------------
On macOS, PySide6/Qt can abort the Python process during pytest when:

1. ``QT_QPA_PLATFORM`` is still the native Cocoa plugin while tests only
   drive ``processEvents()`` (no real event loop).
2. More than one ``QApplication`` is created across test modules.
3. Closing/destroying the last window quits the app mid-suite
   (``quitOnLastWindowClosed`` defaults to True).
4. ``MainWindow`` / ``QTimer`` / ``QThread`` objects are still alive at
   interpreter shutdown.

This conftest forces the offscreen plugin **before** Qt is imported,
provides a single session-scoped ``QApplication``, disables quit-on-last-
window-closed, and supplies helpers that stop timers/threads and destroy
widgets after each GUI test.
"""

from __future__ import annotations

import os
import sys

# MUST run at import time — pytest loads conftest before test modules that
# ``import PySide6``. setdefault preserves an explicit caller override.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest

pytest.importorskip("PySide6")

from PySide6.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="session")
def qapp():
    """One process-wide QApplication for the entire pytest session."""
    app = QApplication.instance()
    if app is None:
        # argv[0] only — avoids pytest's own argv confusing Qt.
        app = QApplication([sys.argv[0] if sys.argv else "petsim-pytest"])

    # Critical on macOS: closing test windows must not quit the session app.
    app.setQuitOnLastWindowClosed(False)

    yield app

    # Drain deferred deleteLater() calls; do **not** destroy the QApplication
    # explicitly — tearing it down before process exit is a common macOS crash.
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
