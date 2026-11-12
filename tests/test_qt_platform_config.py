"""Verify macOS Qt platform remapping without constructing widgets."""

from __future__ import annotations

import os
import sys

import pytest


def test_darwin_offscreen_is_remapped_to_minimal(monkeypatch):
    """The documented pytest command sets offscreen; on Darwin we must not keep it."""
    import importlib
    import tests.conftest as conf

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    # Re-run the pure helper (does not re-import PySide6).
    chosen = conf._configure_qt_platform()
    assert chosen == "minimal"
    assert os.environ["QT_QPA_PLATFORM"] == "minimal"


def test_linux_keeps_offscreen_default(monkeypatch):
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    import tests.conftest as conf

    chosen = conf._configure_qt_platform()
    assert chosen == "offscreen"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only plugin expectation")
def test_session_plugin_is_minimal_on_darwin(qapp):
    assert qapp.platformName() == "minimal"
