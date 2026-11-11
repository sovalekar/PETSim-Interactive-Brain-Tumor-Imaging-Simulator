"""Center panel hosting the detector-ring + phantom Matplotlib canvas."""

from __future__ import annotations

from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from petsim.models.phantom import PhantomParameters
from petsim.viz.detector_canvas import DetectorRingCanvas


class DetectorPanel(QFrame):
    """FOV visualization: 48 detectors around a synthetic brain phantom."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel("Detector ring & phantom")
        title.setObjectName("PanelTitle")

        hint = QLabel(
            "48 discrete detector elements on a circular ring. "
            "The activity map is a synthetic phantom for UI preview only."
        )
        hint.setObjectName("MutedLabel")
        hint.setWordWrap(True)

        self.canvas = DetectorRingCanvas(self)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addWidget(hint)
        layout.addWidget(self.canvas, stretch=1)

    def update_phantom(self, params: PhantomParameters) -> None:
        self.canvas.redraw(params)
