"""Left-hand interactive controls for phantom and display parameters."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from petsim.models.phantom import PhantomState


def _slider_row(
    label: str, minimum: int, maximum: int, value: int, suffix: str = ""
) -> tuple[QHBoxLayout, QSlider, QLabel]:
    row = QHBoxLayout()
    name = QLabel(label)
    name.setMinimumWidth(110)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    readout = QLabel()
    readout.setObjectName("ValueLabel")
    readout.setText(f"{value}{suffix}")
    row.addWidget(name)
    row.addWidget(slider, stretch=1)
    row.addWidget(readout)
    return row, slider, readout


class ControlsPanel(QFrame):
    """Editable phantom / display controls. Emits state on every change."""

    state_changed = Signal(object)  # PhantomState
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")
        self._building = True

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("Simulation Controls")
        title.setObjectName("PanelTitle")
        root.addWidget(title)

        hint = QLabel(
            "Adjust the synthetic brain phantom and tumor for visualization. "
            "PET physics and reconstruction are not simulated yet."
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        form = QVBoxLayout(body)
        form.setSpacing(6)

        # --- Brain ---
        brain_hdr = QLabel("Brain phantom")
        brain_hdr.setObjectName("SectionLabel")
        form.addWidget(brain_hdr)

        row, self._brain_radius, self._brain_radius_val = _slider_row(
            "Radius", 20, 75, 55, "%"
        )
        form.addLayout(row)
        row, self._brain_intensity, self._brain_intensity_val = _slider_row(
            "Intensity", 5, 100, 35, "%"
        )
        form.addLayout(row)

        self._show_outline = QCheckBox("Show brain outline")
        self._show_outline.setChecked(True)
        form.addWidget(self._show_outline)

        # --- Tumor ---
        tumor_hdr = QLabel("Tumor")
        tumor_hdr.setObjectName("SectionLabel")
        form.addWidget(tumor_hdr)

        self._tumor_enabled = QCheckBox("Enable tumor")
        self._tumor_enabled.setChecked(True)
        form.addWidget(self._tumor_enabled)

        row, self._tumor_x, self._tumor_x_val = _slider_row("Center X", -70, 70, 18)
        form.addLayout(row)
        row, self._tumor_y, self._tumor_y_val = _slider_row("Center Y", -70, 70, 12)
        form.addLayout(row)
        row, self._tumor_radius, self._tumor_radius_val = _slider_row(
            "Radius", 3, 30, 10, "%"
        )
        form.addLayout(row)
        row, self._tumor_intensity, self._tumor_intensity_val = _slider_row(
            "Intensity", 10, 200, 100, "%"
        )
        form.addLayout(row)

        # --- Display ---
        display_hdr = QLabel("Display")
        display_hdr.setObjectName("SectionLabel")
        form.addWidget(display_hdr)

        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self._colormap = QComboBox()
        self._colormap.addItems(["hot", "inferno", "magma", "viridis", "gray"])
        cmap_row.addWidget(self._colormap, stretch=1)
        form.addLayout(cmap_row)

        self._show_detectors = QCheckBox("Show detector crystals")
        self._show_detectors.setChecked(True)
        form.addWidget(self._show_detectors)

        self._show_labels = QCheckBox("Label detectors (every 8th)")
        self._show_labels.setChecked(False)
        form.addWidget(self._show_labels)

        form.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        btn_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset defaults")
        self._reset_btn.clicked.connect(self._on_reset)
        btn_row.addWidget(self._reset_btn)
        btn_row.addStretch(1)
        root.addLayout(btn_row)

        # Wire signals
        for slider in (
            self._brain_radius,
            self._brain_intensity,
            self._tumor_x,
            self._tumor_y,
            self._tumor_radius,
            self._tumor_intensity,
        ):
            slider.valueChanged.connect(self._emit_state)

        for box in (
            self._tumor_enabled,
            self._show_outline,
            self._show_detectors,
            self._show_labels,
        ):
            box.toggled.connect(self._emit_state)

        self._colormap.currentTextChanged.connect(self._emit_state)

        self._building = False
        self._emit_state()

    def current_state(self) -> PhantomState:
        return PhantomState(
            brain_radius=self._brain_radius.value() / 100.0,
            brain_intensity=self._brain_intensity.value() / 100.0,
            tumor_enabled=self._tumor_enabled.isChecked(),
            tumor_x=self._tumor_x.value() / 100.0,
            tumor_y=self._tumor_y.value() / 100.0,
            tumor_radius=self._tumor_radius.value() / 100.0,
            tumor_intensity=self._tumor_intensity.value() / 100.0,
            show_outline=self._show_outline.isChecked(),
        )

    def display_options(self) -> dict:
        return {
            "colormap": self._colormap.currentText(),
            "show_detectors": self._show_detectors.isChecked(),
            "show_labels": self._show_labels.isChecked(),
        }

    def apply_state(self, state: PhantomState) -> None:
        """Push a PhantomState into the widgets without recursive emits."""
        self._building = True
        s = state.clamp()
        self._brain_radius.setValue(int(round(s.brain_radius * 100)))
        self._brain_intensity.setValue(int(round(s.brain_intensity * 100)))
        self._tumor_enabled.setChecked(s.tumor_enabled)
        self._tumor_x.setValue(int(round(s.tumor_x * 100)))
        self._tumor_y.setValue(int(round(s.tumor_y * 100)))
        self._tumor_radius.setValue(int(round(s.tumor_radius * 100)))
        self._tumor_intensity.setValue(int(round(s.tumor_intensity * 100)))
        self._show_outline.setChecked(s.show_outline)
        self._building = False
        self._update_readouts()
        self._emit_state()

    def _update_readouts(self) -> None:
        self._brain_radius_val.setText(f"{self._brain_radius.value()}%")
        self._brain_intensity_val.setText(f"{self._brain_intensity.value()}%")
        self._tumor_x_val.setText(f"{self._tumor_x.value() / 100:.2f}")
        self._tumor_y_val.setText(f"{self._tumor_y.value() / 100:.2f}")
        self._tumor_radius_val.setText(f"{self._tumor_radius.value()}%")
        self._tumor_intensity_val.setText(f"{self._tumor_intensity.value()}%")

    def _emit_state(self, *_args) -> None:
        if self._building:
            return
        self._update_readouts()
        self.state_changed.emit(self.current_state())

    def _on_reset(self) -> None:
        self.apply_state(PhantomState())
        self._colormap.setCurrentText("hot")
        self._show_detectors.setChecked(True)
        self._show_labels.setChecked(False)
        self.reset_requested.emit()
