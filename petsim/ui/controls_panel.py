"""Interactive controls for the synthetic brain phantom."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from petsim.models.phantom import PhantomParameters


def _slider_row(label: str, minimum: int, maximum: int, value: int) -> tuple[QLabel, QSlider, QLabel]:
    name = QLabel(label)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    slider.setSingleStep(1)
    value_label = QLabel()
    value_label.setObjectName("ValueLabel")
    value_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return name, slider, value_label


class ControlsPanel(QFrame):
    """Left-hand panel: phantom / tumor knobs that emit updated parameters."""

    parameters_changed = Signal(object)  # PhantomParameters

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("PanelFrame")
        self.setFrameShape(QFrame.Shape.StyledPanel)

        title = QLabel("Simulation controls")
        title.setObjectName("PanelTitle")

        subtitle = QLabel(
            "Adjust the synthetic brain phantom. Controls update the FOV view only — "
            "no PET physics is simulated yet."
        )
        subtitle.setObjectName("MutedLabel")
        subtitle.setWordWrap(True)

        # Internal scale: sliders use integers; we map to float params.
        self._brain_radius = _slider_row("Brain radius", 15, 85, 55)
        self._brain_intensity = _slider_row("Brain intensity", 5, 100, 35)
        self._tumor_radius = _slider_row("Tumor radius", 2, 40, 12)
        self._tumor_x = _slider_row("Tumor X", -80, 80, 18)
        self._tumor_y = _slider_row("Tumor Y", -80, 80, 10)
        self._tumor_intensity = _slider_row("Tumor intensity", 5, 100, 100)

        self._show_tumor = QCheckBox("Show tumor")
        self._show_tumor.setChecked(True)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setFormAlignment(Qt.AlignmentFlag.AlignTop)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(10)

        for name, slider, value_label in (
            self._brain_radius,
            self._brain_intensity,
            self._tumor_radius,
            self._tumor_x,
            self._tumor_y,
            self._tumor_intensity,
        ):
            row = QHBoxLayout()
            row.addWidget(slider, stretch=1)
            row.addWidget(value_label)
            form.addRow(name, row)
            slider.valueChanged.connect(self._on_any_change)

        self._show_tumor.toggled.connect(self._on_any_change)
        form.addRow("", self._show_tumor)

        reset_btn = QPushButton("Reset defaults")
        reset_btn.setObjectName("PrimaryButton")
        reset_btn.clicked.connect(self.reset_defaults)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(14, 14, 14, 14)
        layout.setSpacing(10)
        layout.addWidget(title)
        layout.addWidget(subtitle)
        layout.addLayout(form)
        layout.addStretch(1)
        layout.addWidget(reset_btn)

        self._refresh_value_labels()

    def current_parameters(self) -> PhantomParameters:
        return PhantomParameters(
            brain_radius=self._brain_radius[1].value() / 100.0,
            brain_intensity=self._brain_intensity[1].value() / 100.0,
            tumor_radius=self._tumor_radius[1].value() / 100.0,
            tumor_x=self._tumor_x[1].value() / 100.0,
            tumor_y=self._tumor_y[1].value() / 100.0,
            tumor_intensity=self._tumor_intensity[1].value() / 100.0,
            show_tumor=self._show_tumor.isChecked(),
        ).clamped()

    def reset_defaults(self) -> None:
        defaults = PhantomParameters()
        self._brain_radius[1].setValue(int(round(defaults.brain_radius * 100)))
        self._brain_intensity[1].setValue(int(round(defaults.brain_intensity * 100)))
        self._tumor_radius[1].setValue(int(round(defaults.tumor_radius * 100)))
        self._tumor_x[1].setValue(int(round(defaults.tumor_x * 100)))
        self._tumor_y[1].setValue(int(round(defaults.tumor_y * 100)))
        self._tumor_intensity[1].setValue(int(round(defaults.tumor_intensity * 100)))
        self._show_tumor.setChecked(defaults.show_tumor)
        self._on_any_change()

    def _refresh_value_labels(self) -> None:
        p = self.current_parameters()
        self._brain_radius[2].setText(f"{p.brain_radius:.2f}")
        self._brain_intensity[2].setText(f"{p.brain_intensity:.2f}")
        self._tumor_radius[2].setText(f"{p.tumor_radius:.2f}")
        self._tumor_x[2].setText(f"{p.tumor_x:+.2f}")
        self._tumor_y[2].setText(f"{p.tumor_y:+.2f}")
        self._tumor_intensity[2].setText(f"{p.tumor_intensity:.2f}")

    def _on_any_change(self, *_args) -> None:
        self._refresh_value_labels()
        self.parameters_changed.emit(self.current_parameters())
