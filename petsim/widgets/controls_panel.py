"""Left-hand interactive controls for phantom, acquisition, and display."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QSlider,
    QVBoxLayout,
    QWidget,
)

from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig, AcquisitionStats


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
    """Editable phantom / acquisition / display controls."""

    # Phantom geometry / uptake changed (resets acquisition).
    state_changed = Signal(object)  # PhantomState
    # Colormap / detector labels only (does not reset acquisition).
    display_changed = Signal()
    # Acquisition parameter widgets changed.
    acquisition_config_changed = Signal(object)  # AcquisitionConfig
    start_acquisition = Signal()
    pause_acquisition = Signal()
    reset_acquisition = Signal()
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
            "Adjust the synthetic brain phantom, then run a Stage-2 PET "
            "acquisition (annihilations → 511 keV photon pairs → coincidences). "
            "Image reconstruction is reserved for Stage 3."
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

        # --- Acquisition (Stage 2) ---
        acq_hdr = QLabel("PET acquisition")
        acq_hdr.setObjectName("SectionLabel")
        form.addWidget(acq_hdr)

        btn_row = QHBoxLayout()
        self._start_btn = QPushButton("Start Acquisition")
        self._start_btn.setObjectName("PrimaryButton")
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._reset_acq_btn = QPushButton("Reset Acquisition")
        btn_row.addWidget(self._start_btn)
        btn_row.addWidget(self._pause_btn)
        form.addLayout(btn_row)
        form.addWidget(self._reset_acq_btn)

        row, self._n_events, self._n_events_val = _slider_row(
            "Events", 1000, 20_000, 5000
        )
        # Coarser steps so large ranges stay usable.
        self._n_events.setSingleStep(500)
        self._n_events.setPageStep(1000)
        form.addLayout(row)

        row, self._efficiency, self._efficiency_val = _slider_row(
            "Det. efficiency", 20, 100, 80, "%"
        )
        form.addLayout(row)

        speed_row = QHBoxLayout()
        speed_row.addWidget(QLabel("Anim. speed"))
        self._speed = QComboBox()
        self._speed.addItems(["Slow", "Normal", "Fast"])
        self._speed.setCurrentText("Normal")
        speed_row.addWidget(self._speed, stretch=1)
        form.addLayout(speed_row)

        seed_row = QHBoxLayout()
        self._use_seed = QCheckBox("Random seed")
        self._use_seed.setChecked(False)
        self._seed_edit = QLineEdit("42")
        self._seed_edit.setEnabled(False)
        self._seed_edit.setPlaceholderText("integer seed")
        self._seed_edit.setMaximumWidth(100)
        seed_row.addWidget(self._use_seed)
        seed_row.addWidget(self._seed_edit)
        seed_row.addStretch(1)
        form.addLayout(seed_row)

        stats_hdr = QLabel("Live statistics")
        stats_hdr.setObjectName("SectionLabel")
        form.addWidget(stats_hdr)

        self._stat_total = QLabel("Annihilations: 0")
        self._stat_coin = QLabel("Coincidences: 0")
        self._stat_rate = QLabel("Coincidence rate: — (expected ε²: —)")
        self._stat_progress = QLabel("Progress: 0%")
        for lab in (
            self._stat_total,
            self._stat_coin,
            self._stat_rate,
            self._stat_progress,
        ):
            lab.setObjectName("HintLabel")
            lab.setWordWrap(True)
            form.addWidget(lab)

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

        defaults_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset phantom defaults")
        self._reset_btn.clicked.connect(self._on_reset_phantom)
        defaults_row.addWidget(self._reset_btn)
        defaults_row.addStretch(1)
        root.addLayout(defaults_row)

        # Wire phantom signals
        for slider in (
            self._brain_radius,
            self._brain_intensity,
            self._tumor_x,
            self._tumor_y,
            self._tumor_radius,
            self._tumor_intensity,
        ):
            slider.valueChanged.connect(self._emit_phantom_state)

        for box in (self._tumor_enabled, self._show_outline):
            box.toggled.connect(self._emit_phantom_state)

        # Display-only
        self._colormap.currentTextChanged.connect(self._emit_display)
        self._show_detectors.toggled.connect(self._emit_display)
        self._show_labels.toggled.connect(self._emit_display)

        # Acquisition
        self._start_btn.clicked.connect(self.start_acquisition.emit)
        self._pause_btn.clicked.connect(self.pause_acquisition.emit)
        self._reset_acq_btn.clicked.connect(self.reset_acquisition.emit)
        self._n_events.valueChanged.connect(self._emit_acq_config)
        self._efficiency.valueChanged.connect(self._emit_acq_config)
        self._speed.currentTextChanged.connect(self._emit_acq_config)
        self._use_seed.toggled.connect(self._on_seed_toggled)
        self._seed_edit.editingFinished.connect(self._emit_acq_config)

        self._building = False
        self._emit_phantom_state()
        self._emit_acq_config()
        self.update_stats(AcquisitionStats(target_events=5000, detector_efficiency=0.8))

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

    def acquisition_config(self) -> AcquisitionConfig:
        seed_val: int | None = None
        text = self._seed_edit.text().strip()
        if text:
            try:
                seed_val = int(text)
            except ValueError:
                seed_val = 42
        return AcquisitionConfig(
            n_events=self._n_events.value(),
            detector_efficiency=self._efficiency.value() / 100.0,
            animation_speed=self._speed.currentText(),
            seed=seed_val,
            use_seed=self._use_seed.isChecked(),
        ).clamp()

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
        self._update_phantom_readouts()
        self._emit_phantom_state()

    def update_stats(self, stats: AcquisitionStats) -> None:
        self._stat_total.setText(f"Annihilations: {stats.total_events:,}")
        self._stat_coin.setText(f"Coincidences: {stats.coincidences:,}")
        self._stat_rate.setText(
            f"Coincidence rate: {stats.coincidence_percent:.1f}% "
            f"(expected ε²: {stats.expected_coincidence_percent:.1f}%)"
        )
        self._stat_progress.setText(f"Progress: {stats.progress_percent:.1f}%")

        # Button enablement mirrors acquisition state.
        if stats.finished:
            self._start_btn.setText("Start Acquisition")
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
        elif stats.running:
            self._start_btn.setText("Running…")
            self._start_btn.setEnabled(False)
            self._pause_btn.setEnabled(True)
        elif stats.paused:
            self._start_btn.setText("Resume")
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)
        else:
            self._start_btn.setText("Start Acquisition")
            self._start_btn.setEnabled(True)
            self._pause_btn.setEnabled(False)

    def set_acquisition_controls_enabled(self, enabled: bool) -> None:
        """Optionally lock parameter widgets while running (speed still live)."""
        self._n_events.setEnabled(enabled)
        self._use_seed.setEnabled(enabled)
        self._seed_edit.setEnabled(enabled and self._use_seed.isChecked())

    def _update_phantom_readouts(self) -> None:
        self._brain_radius_val.setText(f"{self._brain_radius.value()}%")
        self._brain_intensity_val.setText(f"{self._brain_intensity.value()}%")
        self._tumor_x_val.setText(f"{self._tumor_x.value() / 100:.2f}")
        self._tumor_y_val.setText(f"{self._tumor_y.value() / 100:.2f}")
        self._tumor_radius_val.setText(f"{self._tumor_radius.value()}%")
        self._tumor_intensity_val.setText(f"{self._tumor_intensity.value()}%")

    def _update_acq_readouts(self) -> None:
        self._n_events_val.setText(f"{self._n_events.value():,}")
        self._efficiency_val.setText(f"{self._efficiency.value()}%")

    def _emit_phantom_state(self, *_args) -> None:
        if self._building:
            return
        self._update_phantom_readouts()
        self.state_changed.emit(self.current_state())

    def _emit_display(self, *_args) -> None:
        if self._building:
            return
        self.display_changed.emit()

    def _emit_acq_config(self, *_args) -> None:
        if self._building:
            return
        self._update_acq_readouts()
        self.acquisition_config_changed.emit(self.acquisition_config())

    def _on_seed_toggled(self, checked: bool) -> None:
        self._seed_edit.setEnabled(checked)
        self._emit_acq_config()

    def _on_reset_phantom(self) -> None:
        self.apply_state(PhantomState())
        self._colormap.setCurrentText("hot")
        self._show_detectors.setChecked(True)
        self._show_labels.setChecked(False)
        self.reset_requested.emit()
        self._emit_display()
