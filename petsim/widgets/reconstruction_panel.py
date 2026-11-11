"""Right-hand panel reserved for future image reconstruction output."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QFrame,
    QLabel,
    QPushButton,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from petsim.theme import COLORS


class ReconstructionPanel(QFrame):
    """Placeholder for reconstructed PET images.

    Intentionally empty of physics or fake reconstructions — reserved for a
    future module that will receive sinograms / projections.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Reconstruction")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Reserved for tomographic reconstruction. No PET physics or "
            "image formation is implemented in this scaffold."
        )
        subtitle.setObjectName("HintLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        placeholder = QFrame()
        placeholder.setObjectName("Panel")
        placeholder.setStyleSheet(
            f"""
            QFrame {{
                background-color: {COLORS["bg"]};
                border: 1px dashed {COLORS["border"]};
                border-radius: 6px;
            }}
            """
        )
        placeholder.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding
        )
        ph_layout = QVBoxLayout(placeholder)
        ph_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        badge = QLabel("RECONSTRUCTION OUTPUT")
        badge.setObjectName("SectionLabel")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(badge)

        msg = QLabel(
            "This panel will host reconstructed activity maps\n"
            "(FBP / OSEM / ML-EM) once the forward model is added.\n\n"
            "Controls below are disabled until then."
        )
        msg.setObjectName("HintLabel")
        msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        msg.setWordWrap(True)
        ph_layout.addWidget(msg)

        layout.addWidget(placeholder, stretch=1)

        self._run_btn = QPushButton("Run reconstruction")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip("Not available — reconstruction not implemented yet.")
        layout.addWidget(self._run_btn)

        self._clear_btn = QPushButton("Clear output")
        self._clear_btn.setEnabled(False)
        self._clear_btn.setToolTip("Not available — reconstruction not implemented yet.")
        layout.addWidget(self._clear_btn)
