"""Dark workstation theme for PETSim."""

from __future__ import annotations

from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import QApplication

# Palette tokens — cool charcoal / teal clinical workstation look
COLORS = {
    "bg": "#12161c",
    "bg_elevated": "#1a2028",
    "bg_panel": "#161b22",
    "border": "#2a3340",
    "text": "#e6edf3",
    "text_muted": "#8b9bb0",
    "accent": "#3db8a0",
    "accent_dim": "#2a8f7c",
    "danger": "#d46a6a",
    "input": "#0f1318",
    "highlight": "#243040",
}

STYLESHEET = f"""
QWidget {{
    background-color: {COLORS["bg"]};
    color: {COLORS["text"]};
    font-family: "Segoe UI", "Ubuntu", "Cantarell", sans-serif;
    font-size: 13px;
}}

QMainWindow, QDialog {{
    background-color: {COLORS["bg"]};
}}

QFrame#PanelFrame {{
    background-color: {COLORS["bg_panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
}}

QLabel#PanelTitle {{
    color: {COLORS["text"]};
    font-size: 14px;
    font-weight: 600;
    padding: 4px 0 8px 0;
}}

QLabel#MutedLabel {{
    color: {COLORS["text_muted"]};
}}

QLabel#ValueLabel {{
    color: {COLORS["accent"]};
    min-width: 48px;
}}

QGroupBox {{
    background-color: {COLORS["bg_elevated"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    margin-top: 12px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
    color: {COLORS["text_muted"]};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {COLORS["input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    width: 14px;
    margin: -5px 0;
    background: {COLORS["accent"]};
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS["accent_dim"]};
    border-radius: 3px;
}}

QDoubleSpinBox, QSpinBox, QComboBox, QLineEdit {{
    background-color: {COLORS["input"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 4px 6px;
    selection-background-color: {COLORS["accent_dim"]};
}}

QPushButton {{
    background-color: {COLORS["bg_elevated"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 6px 14px;
    color: {COLORS["text"]};
}}

QPushButton:hover {{
    border-color: {COLORS["accent"]};
    color: {COLORS["accent"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["highlight"]};
}}

QPushButton#PrimaryButton {{
    background-color: {COLORS["accent_dim"]};
    border-color: {COLORS["accent"]};
    color: {COLORS["text"]};
    font-weight: 600;
}}

QPushButton#PrimaryButton:hover {{
    background-color: {COLORS["accent"]};
    color: {COLORS["bg"]};
}}

QStatusBar {{
    background-color: {COLORS["bg_elevated"]};
    border-top: 1px solid {COLORS["border"]};
    color: {COLORS["text_muted"]};
}}

QSplitter::handle {{
    background-color: {COLORS["border"]};
    width: 2px;
    height: 2px;
}}

QScrollArea {{
    border: none;
    background-color: transparent;
}}
"""


def apply_dark_theme(app: QApplication) -> None:
    """Apply palette + stylesheet so the whole app stays dark, including Matplotlib hosts."""
    palette = QPalette()
    bg = QColor(COLORS["bg"])
    text = QColor(COLORS["text"])
    accent = QColor(COLORS["accent"])
    base = QColor(COLORS["input"])
    muted = QColor(COLORS["text_muted"])

    palette.setColor(QPalette.ColorRole.Window, bg)
    palette.setColor(QPalette.ColorRole.WindowText, text)
    palette.setColor(QPalette.ColorRole.Base, base)
    palette.setColor(QPalette.ColorRole.AlternateBase, QColor(COLORS["bg_elevated"]))
    palette.setColor(QPalette.ColorRole.Text, text)
    palette.setColor(QPalette.ColorRole.Button, QColor(COLORS["bg_elevated"]))
    palette.setColor(QPalette.ColorRole.ButtonText, text)
    palette.setColor(QPalette.ColorRole.Highlight, accent)
    palette.setColor(QPalette.ColorRole.HighlightedText, bg)
    palette.setColor(QPalette.ColorRole.ToolTipBase, QColor(COLORS["bg_elevated"]))
    palette.setColor(QPalette.ColorRole.ToolTipText, text)
    palette.setColor(QPalette.ColorRole.PlaceholderText, muted)
    palette.setColor(QPalette.ColorRole.Link, accent)

    app.setPalette(palette)
    app.setStyle("Fusion")
    app.setStyleSheet(STYLESHEET)
