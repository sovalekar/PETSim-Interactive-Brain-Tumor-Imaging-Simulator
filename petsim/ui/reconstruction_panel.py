"""Reserved reconstruction panel — intentionally empty / placeholder only."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget


class ReconstructionPanel(QFrame):
    """Right-hand panel reserved for future reconstruction UI.

    No reconstruction algorithm or fake reconstructed image is shown.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel("Reconstruction")
        title.setObjectName("PanelTitle")

        placeholder = QLabel(
            "Reserved for image reconstruction.\n\n"
            "This panel is intentionally empty — PET reconstruction "
            "algorithms are not implemented in this release."
        )
        placeholder.setObjectName("MutedLabel")
        placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        placeholder.setWordWrap(True)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(8)
        layout.addWidget(title)
        layout.addStretch(1)
        layout.addWidget(placeholder)
        layout.addStretch(1)
