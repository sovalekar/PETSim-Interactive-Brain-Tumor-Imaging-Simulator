"""Shared workstation chrome: badges, sections, metric cards, and plot styling."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QLabel,
    QSizePolicy,
    QTableWidgetItem,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from petsim.theme import style_axes

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

GUIDE_TEXT = """
<p><b>Coincidence detection.</b> A PET coincidence is two 511 keV photons recorded
close enough in time to be treated as one annihilation. PETSim draws them as a
line between two crystals.</p>
<p><b>Random coincidences.</b> Two photons from different annihilations can fall
in the timing window. Stage 7 inserts those pairs uniformly among crystals.
Random correction subtracts that expected uniform background. It changes the
sinogram shape.</p>
<p><b>Compton scatter.</b> A photon can deflect once in the simplified in-plane
model and lose energy. The photopeak still accepts small deflections. Those
events sit on top of the unscattered lines.</p>
<p><b>Global scatter scaling.</b> A lower energy window estimates how many
photopeak prompts are scattered. Every remaining bin is multiplied by that one
fraction. This is a total, not a map of where scatter sits, and it does not
remove individual events.</p>
<p><b>Filtered backprojection.</b> The sinogram is filtered and smeared back
across the image. The ramp keeps high frequencies. Hann rolls them off.</p>
<p><b>MLEM.</b> Maximum-likelihood expectation maximization updates a nonnegative
image to raise a Poisson likelihood. More iterations can fit noise.</p>
<p><b>Tumor-to-background ratio.</b> Mean tumor-ROI intensity divided by mean
background-ROI intensity. It is not a calibrated SUV.</p>
<p><b>Localization error.</b> Distance, in field-of-view units, between the known
tumor center and the peak found in the reconstructed image.</p>
"""


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


class CollapsibleSection(QWidget):
    """Heading that shows or hides a body. The body layout is ``content``."""

    def __init__(self, title: str, kind: str | None = None, *, expanded: bool = True) -> None:
        super().__init__()
        root = QVBoxLayout(self)
        root.setContentsMargins(0, 2, 0, 2)
        root.setSpacing(4)
        header = QVBoxLayout()
        header.setSpacing(2)
        self._toggle = QToolButton()
        self._toggle.setObjectName("CollapseButton")
        self._toggle.setCheckable(True)
        self._toggle.setChecked(expanded)
        self._toggle.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextBesideIcon)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)
        self._toggle.setText(title)
        self._toggle.setToolTip(f"Show or hide {title}.")
        self._toggle.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
        self._toggle.toggled.connect(self._on_toggled)
        header.addWidget(self._toggle, alignment=Qt.AlignmentFlag.AlignLeft)
        if kind is not None:
            chip = kind_chip(kind)
            chip.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Fixed)
            header.addWidget(chip, alignment=Qt.AlignmentFlag.AlignLeft)
        root.addLayout(header)
        self.body = QWidget()
        self.content = QVBoxLayout(self.body)
        self.content.setContentsMargins(0, 0, 0, 4)
        self.content.setSpacing(6)
        self.body.setVisible(expanded)
        root.addWidget(self.body)

    def _on_toggled(self, expanded: bool) -> None:
        self.body.setVisible(expanded)
        self._toggle.setArrowType(Qt.ArrowType.DownArrow if expanded else Qt.ArrowType.RightArrow)


class MetricCard(QFrame):
    """Compact measured or estimated value. ``value`` accepts ``setText``."""

    def __init__(self, caption: str, kind: str | None = None) -> None:
        super().__init__()
        self.setObjectName("MetricCard")
        self.setMinimumWidth(0)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)
        top = QHBoxLayout()
        cap = QLabel(caption)
        cap.setObjectName("MetricCaption")
        cap.setWordWrap(True)
        top.addWidget(cap, stretch=1)
        if kind is not None:
            top.addWidget(kind_chip(kind))
        layout.addLayout(top)
        self.value = QLabel("—")
        self.value.setObjectName("MetricValue")
        self.value.setWordWrap(True)
        self.value.setMinimumWidth(0)
        layout.addWidget(self.value)


def numeric_item(text: str) -> QTableWidgetItem:
    """Right-aligned, read-only table cell for a quantitative value."""
    item = QTableWidgetItem(text)
    item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
    return item


def show_guide(parent: QWidget) -> None:
    """Undergraduate-level explanations of the terms used in the workstation."""
    from PySide6.QtWidgets import QMessageBox

    box = QMessageBox(parent)
    box.setWindowTitle("PETSim guide")
    box.setTextFormat(Qt.TextFormat.RichText)
    box.setText(GUIDE_TEXT)
    box.exec()


__all__ = [
    "GUIDE_TEXT",
    "KIND_HELP",
    "CollapsibleSection",
    "MetricCard",
    "kind_chip",
    "numeric_item",
    "section_row",
    "show_guide",
    "style_axes",
]
