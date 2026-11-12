"""Qt platform selection policy for pytest (no Stage 1–3 physics changes)."""

from __future__ import annotations

import os
import sys

import pytest


def test_darwin_offscreen_is_remapped_to_cocoa(monkeypatch):
    """Documented offscreen command must not force a broken minimal plugin."""
    import tests.conftest as conf

    monkeypatch.setenv("QT_QPA_PLATFORM", "offscreen")
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    chosen = conf._configure_qt_platform()
    assert chosen == "cocoa"
    assert os.environ["QT_QPA_PLATFORM"] == "cocoa"


def test_darwin_minimal_is_remapped_to_cocoa(monkeypatch):
    import tests.conftest as conf

    monkeypatch.setenv("QT_QPA_PLATFORM", "minimal")
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    chosen = conf._configure_qt_platform()
    assert chosen == "cocoa"


def test_darwin_preserves_explicit_cocoa(monkeypatch):
    import tests.conftest as conf

    monkeypatch.setenv("QT_QPA_PLATFORM", "cocoa")
    monkeypatch.setattr(sys, "platform", "darwin", raising=False)
    chosen = conf._configure_qt_platform()
    assert chosen == "cocoa"
    assert os.environ["QT_QPA_PLATFORM"] == "cocoa"


def test_linux_keeps_offscreen_default(monkeypatch):
    monkeypatch.delenv("QT_QPA_PLATFORM", raising=False)
    monkeypatch.setattr(sys, "platform", "linux", raising=False)
    import tests.conftest as conf

    chosen = conf._configure_qt_platform()
    assert chosen == "offscreen"


@pytest.mark.skipif(sys.platform != "darwin", reason="macOS-only plugin expectation")
def test_session_plugin_is_usable_on_darwin(qapp):
    """Accept cocoa (preferred). Do not require minimal when cocoa was chosen."""
    name = qapp.platformName()
    assert name in {"cocoa", "offscreen", "minimal"}, f"unexpected plugin: {name}"
    # If the user/env selected cocoa, the live plugin must match.
    requested = os.environ.get("QT_QPA_PLATFORM", "cocoa")
    if requested == "cocoa":
        assert name == "cocoa"
