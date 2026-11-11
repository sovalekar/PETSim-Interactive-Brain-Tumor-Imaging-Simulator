"""Right-hand panel reserved for Stage 3 image reconstruction."""

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
    """Placeholder for reconstructed PET images (Stage 3 — unavailable).

    Stage 2 records coincidence LORs in memory, but this panel must not show
    fake reconstructions or fabricated activity maps.
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
            "Unavailable until Stage 3. Coincidence events from Stage 2 "
            "acquisition are stored for a future reconstructor — no sinograms, "
            "FBP, OSEM, or ML-EM are implemented yet."
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

        badge = QLabel("STAGE 3 — NOT AVAILABLE")
        badge.setObjectName("SectionLabel")
        badge.setAlignment(Qt.AlignmentFlag.AlignCenter)
        ph_layout.addWidget(badge)

        self._msg = QLabel(
            "No reconstructed images will be shown here in Stage 2.\n\n"
            "Recorded coincidences: 0"
        )
        self._msg.setObjectName("HintLabel")
        self._msg.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self._msg.setWordWrap(True)
        ph_layout.addWidget(self._msg)

        layout.addWidget(placeholder, stretch=1)

        self._run_btn = QPushButton("Run reconstruction")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.setToolTip(
            "Unavailable — Stage 3 reconstruction is not implemented yet."
        )
        layout.addWidget(self._run_btn)

        self._clear_btn = QPushButton("Clear output")
        self._clear_btn.setEnabled(False)
        self._clear_btn.setToolTip(
            "Unavailable — Stage 3 reconstruction is not implemented yet."
        )
        layout.addWidget(self._clear_btn)

    def update_coincidence_count(self, n: int) -> None:
        """Show how many LORs are buffered (still no image)."""
        self._msg.setText(
            "No reconstructed images will be shown here in Stage 2.\n\n"
            f"Recorded coincidences: {n:,}"
        )
