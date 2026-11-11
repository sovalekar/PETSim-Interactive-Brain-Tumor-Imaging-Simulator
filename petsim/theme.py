"""Dark clinical workstation theme for PETSim."""

from __future__ import annotations

# Shared palette used by both Qt stylesheets and Matplotlib figures.
COLORS = {
    "bg": "#0e1117",
    "panel": "#161b22",
    "panel_alt": "#1c2330",
    "border": "#30363d",
    "text": "#e6edf3",
    "text_muted": "#8b949e",
    "accent": "#3fb950",
    "accent_dim": "#238636",
    "warn": "#d29922",
    "danger": "#f85149",
    "brain": "#4a6fa5",
    "brain_edge": "#7aa2d4",
    "tumor": "#e5534b",
    "tumor_hot": "#ff7b72",
    "detector": "#58a6ff",
    "detector_active": "#79c0ff",
    "fov": "#30363d",
    "grid": "#21262d",
}

DARK_STYLESHEET = f"""
QWidget {{
    background-color: {COLORS["bg"]};
    color: {COLORS["text"]};
    font-family: "Segoe UI", "Ubuntu", "Cantarell", sans-serif;
    font-size: 13px;
}}

QMainWindow {{
    background-color: {COLORS["bg"]};
}}

QFrame#Panel {{
    background-color: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
}}

QLabel#PanelTitle {{
    color: {COLORS["text"]};
    font-size: 14px;
    font-weight: 600;
    padding: 4px 0;
}}

QLabel#SectionLabel {{
    color: {COLORS["text_muted"]};
    font-size: 11px;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    padding-top: 8px;
}}

QLabel#HintLabel {{
    color: {COLORS["text_muted"]};
    font-size: 12px;
}}

QLabel#ValueLabel {{
    color: {COLORS["accent"]};
    font-family: "Cascadia Code", "Consolas", "Ubuntu Mono", monospace;
    font-size: 12px;
    min-width: 52px;
}}

QGroupBox {{
    background-color: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    margin-top: 12px;
    padding-top: 14px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 6px;
    color: {COLORS["text_muted"]};
}}

QSlider::groove:horizontal {{
    height: 6px;
    background: {COLORS["border"]};
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: {COLORS["accent"]};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS["accent_dim"]};
    border-radius: 3px;
}}

QCheckBox {{
    spacing: 8px;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS["border"]};
    border-radius: 3px;
    background: {COLORS["panel_alt"]};
}}

QCheckBox::indicator:checked {{
    background: {COLORS["accent_dim"]};
    border-color: {COLORS["accent"]};
}}

QPushButton {{
    background-color: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 7px 14px;
    color: {COLORS["text"]};
}}

QPushButton:hover {{
    background-color: {COLORS["border"]};
    border-color: {COLORS["text_muted"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["accent_dim"]};
}}

QPushButton#PrimaryButton {{
    background-color: {COLORS["accent_dim"]};
    border-color: {COLORS["accent"]};
    font-weight: 600;
}}

QPushButton#PrimaryButton:hover {{
    background-color: {COLORS["accent"]};
    color: {COLORS["bg"]};
}}

QPushButton:disabled,
QPushButton#PrimaryButton:disabled {{
    color: {COLORS["text_muted"]};
    background-color: {COLORS["panel"]};
    border-color: {COLORS["border"]};
}}

QComboBox {{
    background-color: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    padding: 5px 8px;
    min-width: 120px;
}}

QComboBox::drop-down {{
    border: none;
    width: 20px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    selection-background-color: {COLORS["accent_dim"]};
}}

QStatusBar {{
    background-color: {COLORS["panel"]};
    border-top: 1px solid {COLORS["border"]};
    color: {COLORS["text_muted"]};
}}

QSplitter::handle {{
    background-color: {COLORS["border"]};
    width: 2px;
}}

QScrollArea {{
    border: none;
    background: transparent;
}}
"""
