"""Shared labels that separate parameters, measurements, estimates, and truth."""

from __future__ import annotations

from PySide6.QtWidgets import QHBoxLayout, QLabel, QWidget

KIND_HELP = {
    "parameter": "Configurable simulation or reconstruction setting. Changing it changes the next run.",
    "measured": "Quantity counted or formed from the current acquisition or reconstructed image.",
    "estimated": "Model-based estimate. It is not a ground-truth label and may not be a spatial map.",
    "truth": "Known phantom value used only for educational validation after reconstruction.",
}

_OBJECT_NAMES = {
    "parameter": "KindParameter",
    "measured": "KindMeasured",
    "estimated": "KindEstimated",
    "truth": "KindTruth",
}

_TITLES = {
    "parameter": "Parameter",
    "measured": "Measured",
    "estimated": "Estimated",
    "truth": "Ground truth",
}


def kind_chip(kind: str) -> QLabel:
    """Small badge. ``kind`` is parameter, measured, estimated, or truth."""
    label = QLabel(_TITLES[kind])
    label.setObjectName(_OBJECT_NAMES[kind])
    label.setToolTip(KIND_HELP[kind])
    return label


def section_row(title: str, kind: str | None = None) -> QWidget:
    """Section title with an optional kind badge."""
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 4, 0, 0)
    layout.setSpacing(8)
    heading = QLabel(title)
    heading.setObjectName("SectionLabel")
    layout.addWidget(heading)
    if kind is not None:
        layout.addWidget(kind_chip(kind))
    layout.addStretch(1)
    return row
