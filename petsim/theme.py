"""Dark scientific-workstation theme for PETSim.

Interface colors are separate from the scientific colors used for the
phantom, tumor, and detector events.
"""

from __future__ import annotations

COLORS = {
    "bg": "#090D15",
    "panel": "#151E2B",
    "panel_alt": "#1B2636",
    "elevated": "#101722",
    "border": "#293547",
    "text": "#F3F6FB",
    "text_muted": "#94A3B8",
    "accent": "#38BDF8",
    "accent_dim": "#0E7490",
    "accent_secondary": "#818CF8",
    "warn": "#F59E0B",
    "danger": "#EF4444",
    "success": "#22C55E",
    "brain": "#4a6fa5",
    "brain_edge": "#7aa2d4",
    "tumor": "#e5534b",
    "tumor_hot": "#ff7b72",
    "detector": "#58a6ff",
    "detector_active": "#79c0ff",
    "fov": "#293547",
    "grid": "#1B2636",
    "info": "#38BDF8",
    "estimated": "#F59E0B",
    "truth": "#818CF8",
}

_FONT = '"Helvetica Neue", "Segoe UI", "Ubuntu", sans-serif'
_MONO = '"SF Mono", "Menlo", "Cascadia Code", "Ubuntu Mono", monospace'


def style_axes(ax, *, grid: bool = False) -> None:
    """Apply the shared Matplotlib surface. Does not change plotted data."""
    ax.set_facecolor(COLORS["bg"])
    ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
    ax.title.set_color(COLORS["text"])
    ax.xaxis.label.set_color(COLORS["text_muted"])
    ax.yaxis.label.set_color(COLORS["text_muted"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])
    if grid:
        ax.grid(True, color=COLORS["border"], linewidth=0.4, alpha=0.55)
    figure = getattr(ax, "figure", None)
    if figure is not None:
        figure.set_facecolor(COLORS["panel"])


DARK_STYLESHEET = f"""
QWidget {{
    background-color: {COLORS["bg"]};
    color: {COLORS["text"]};
    font-family: {_FONT};
    font-size: 13px;
}}

QMainWindow, QMenuBar {{
    background-color: {COLORS["bg"]};
}}

QMenuBar {{
    border-bottom: 1px solid {COLORS["border"]};
    spacing: 4px;
    padding: 2px 8px;
}}

QMenuBar::item:selected, QMenu::item:selected {{
    background: {COLORS["panel_alt"]};
}}

QMenu {{
    background: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    padding: 4px;
}}

QFrame#Panel, QWidget#AppBar {{
    background-color: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 10px;
}}

QFrame#MetricCard {{
    background-color: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
}}

QLabel#AppTitle {{
    font-size: 20px;
    font-weight: 600;
    background: transparent;
    letter-spacing: 0.01em;
}}

QLabel#AppSubtitle, QLabel#HeaderStatus {{
    color: {COLORS["text_muted"]};
    font-size: 12px;
    background: transparent;
}}

QLabel#LogoMark {{
    color: {COLORS["accent"]};
    font-size: 22px;
    font-weight: 600;
    background: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 2px 8px;
}}

QLabel#PanelTitle {{
    color: {COLORS["text"]};
    font-size: 16px;
    font-weight: 600;
    background: transparent;
    padding: 0;
}}

QLabel#SectionLabel {{
    color: {COLORS["text"]};
    font-size: 13px;
    font-weight: 600;
    background: transparent;
    padding-top: 2px;
}}

QLabel#HintLabel {{
    color: {COLORS["text_muted"]};
    font-size: 12px;
    background: transparent;
}}

QLabel#ValueLabel, QLabel#MetricValue {{
    color: {COLORS["text"]};
    font-family: {_MONO};
    font-size: 18px;
    font-weight: 600;
    background: transparent;
}}

QLabel#MetricCaption {{
    color: {COLORS["text_muted"]};
    font-size: 11px;
    background: transparent;
}}

QLabel#KindParameter, QLabel#KindMeasured, QLabel#KindEstimated, QLabel#KindTruth {{
    font-size: 10px;
    font-weight: 600;
    padding: 1px 7px;
    border-radius: 8px;
    background: {COLORS["elevated"]};
}}

QLabel#KindParameter {{
    color: {COLORS["info"]};
    border: 1px solid {COLORS["info"]};
}}

QLabel#KindMeasured {{
    color: {COLORS["success"]};
    border: 1px solid {COLORS["success"]};
}}

QLabel#KindEstimated {{
    color: {COLORS["estimated"]};
    border: 1px solid {COLORS["estimated"]};
}}

QLabel#KindTruth {{
    color: {COLORS["truth"]};
    border: 1px solid {COLORS["truth"]};
}}

QGroupBox {{
    background-color: {COLORS["elevated"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    margin-top: 14px;
    padding: 12px 8px 8px 8px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    left: 12px;
    padding: 0 6px;
    color: {COLORS["text_muted"]};
}}

QToolButton#CollapseButton {{
    background: transparent;
    border: none;
    color: {COLORS["text"]};
    font-size: 13px;
    font-weight: 600;
    padding: 4px 2px;
    text-align: left;
}}

QToolButton#CollapseButton:hover {{
    color: {COLORS["accent"]};
}}

QPushButton {{
    background-color: {COLORS["panel_alt"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 8px;
    padding: 7px 14px;
    color: {COLORS["text"]};
}}

QPushButton:hover {{
    background-color: {COLORS["border"]};
}}

QPushButton:pressed {{
    background-color: {COLORS["accent_dim"]};
}}

QPushButton:disabled, QPushButton#PrimaryButton:disabled, QPushButton#DangerButton:disabled {{
    color: {COLORS["text_muted"]};
    background-color: {COLORS["elevated"]};
    border-color: {COLORS["border"]};
}}

QPushButton#PrimaryButton {{
    background-color: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
    color: #082033;
    font-weight: 600;
}}

QPushButton#PrimaryButton:hover {{
    background-color: #7DD3FC;
    color: #082033;
}}

QPushButton#DangerButton {{
    background: transparent;
    border-color: {COLORS["danger"]};
    color: {COLORS["danger"]};
}}

QPushButton#DangerButton:hover {{
    background: {COLORS["danger"]};
    color: {COLORS["bg"]};
}}

QPushButton#StepButton {{
    background: transparent;
    border: 1px solid transparent;
    color: {COLORS["text_muted"]};
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 600;
}}

QPushButton#StepButton:hover {{
    background: {COLORS["panel_alt"]};
    color: {COLORS["text"]};
}}

QPushButton#StepButton:checked {{
    background: {COLORS["panel_alt"]};
    border-color: {COLORS["accent"]};
    color: {COLORS["text"]};
}}

QSlider::groove:horizontal {{
    height: 4px;
    background: {COLORS["border"]};
    border-radius: 2px;
}}

QSlider::sub-page:horizontal {{
    background: {COLORS["accent"]};
    border-radius: 2px;
}}

QSlider::handle:horizontal {{
    background: {COLORS["text"]};
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
    border: 2px solid {COLORS["accent"]};
}}

QCheckBox {{
    spacing: 8px;
    background: transparent;
}}

QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {COLORS["border"]};
    border-radius: 4px;
    background: {COLORS["elevated"]};
}}

QCheckBox::indicator:checked {{
    background: {COLORS["accent"]};
    border-color: {COLORS["accent"]};
}}

QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox {{
    background-color: {COLORS["elevated"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    padding: 4px 8px;
    min-height: 22px;
}}

QComboBox:hover, QLineEdit:hover, QSpinBox:hover, QDoubleSpinBox:hover,
QComboBox:focus, QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLORS["accent"]};
}}

QComboBox::drop-down {{
    border: none;
    width: 22px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS["panel"]};
    border: 1px solid {COLORS["border"]};
    selection-background-color: {COLORS["accent_dim"]};
    selection-color: {COLORS["text"]};
}}

QStatusBar {{
    background-color: {COLORS["elevated"]};
    border-top: 1px solid {COLORS["border"]};
    color: {COLORS["text_muted"]};
    font-size: 12px;
}}

QSplitter::handle {{
    background-color: {COLORS["border"]};
}}

QSplitter::handle:hover {{
    background-color: {COLORS["accent"]};
}}

QScrollArea {{
    border: none;
    background: transparent;
}}

QProgressBar {{
    background: {COLORS["elevated"]};
    border: 1px solid {COLORS["border"]};
    border-radius: 6px;
    text-align: center;
    color: {COLORS["text"]};
    min-height: 16px;
    font-size: 11px;
}}

QProgressBar::chunk {{
    background: {COLORS["accent"]};
    border-radius: 5px;
}}

QHeaderView::section {{
    background: {COLORS["panel_alt"]};
    color: {COLORS["text"]};
    border: none;
    border-bottom: 1px solid {COLORS["border"]};
    border-right: 1px solid {COLORS["border"]};
    padding: 6px 8px;
    font-size: 11px;
    font-weight: 600;
}}

QTableWidget {{
    background: {COLORS["panel"]};
    alternate-background-color: {COLORS["elevated"]};
    gridline-color: {COLORS["border"]};
    selection-background-color: {COLORS["accent_dim"]};
    selection-color: {COLORS["text"]};
    font-family: {_MONO};
    font-size: 12px;
}}

QTableWidget::item {{
    padding: 4px 6px;
}}

QToolTip {{
    background: {COLORS["panel_alt"]};
    color: {COLORS["text"]};
    border: 1px solid {COLORS["border"]};
    padding: 6px 8px;
}}
"""
