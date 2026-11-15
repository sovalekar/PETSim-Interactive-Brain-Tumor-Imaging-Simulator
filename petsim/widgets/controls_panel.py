"""Left-hand interactive controls for phantom, acquisition, and display."""

from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from petsim.widgets.chrome import CollapsibleSection, MetricCard

from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig, AcquisitionStats
from petsim.physics.attenuation import AttenuationConfig


def _slider_row(
    label: str, minimum: int, maximum: int, value: int, suffix: str = ""
) -> tuple[QVBoxLayout, QSlider, QLabel]:
    row = QVBoxLayout()
    row.setSpacing(2)
    name = QLabel(label)
    name.setObjectName("HintLabel")
    line = QHBoxLayout()
    line.setSpacing(8)
    slider = QSlider(Qt.Orientation.Horizontal)
    slider.setMinimum(minimum)
    slider.setMaximum(maximum)
    slider.setValue(value)
    slider.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
    readout = QLabel()
    readout.setObjectName("ValueLabel")
    readout.setText(f"{value}{suffix}")
    precise = QSpinBox()
    precise.setRange(minimum, maximum)
    precise.setValue(value)
    if suffix:
        precise.setSuffix(suffix)
    precise.setFixedWidth(76)
    precise.setToolTip("Type an exact value. It uses the same range as the slider.")

    def _from_slider(current: int) -> None:
        readout.setText(f"{current}{suffix}")
        precise.blockSignals(True)
        precise.setValue(current)
        precise.blockSignals(False)

    slider.valueChanged.connect(_from_slider)
    precise.valueChanged.connect(slider.setValue)
    readout.hide()
    line.addWidget(slider, stretch=1)
    line.addWidget(precise)
    row.addWidget(name)
    row.addLayout(line)
    return row, slider, readout


class ControlsPanel(QFrame):
    """Editable phantom / acquisition / display controls."""

    # Phantom geometry / uptake changed (resets acquisition).
    state_changed = Signal(object)  # PhantomState
    # Colormap / detector labels only (does not reset acquisition).
    display_changed = Signal()
    # Acquisition parameter widgets changed.
    acquisition_config_changed = Signal(object)  # AcquisitionConfig
    attenuation_changed = Signal(object)  # AttenuationConfig
    attenuation_study_requested = Signal()
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

        hint = QLabel("Phantom, acquisition, then reconstruction.")
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        hint.setToolTip(
            "Blue badges are settings you choose. Green badges are counts from the current run. "
            "The Guide explains the PET terms."
        )
        root.addWidget(hint)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        body.setMinimumWidth(0)
        body.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
        form = QVBoxLayout(body)
        form.setContentsMargins(0, 0, 4, 0)
        form.setSpacing(8)

        phantom = CollapsibleSection("Phantom configuration", "parameter")
        row, self._brain_radius, self._brain_radius_val = _slider_row(
            "Brain radius", 20, 75, 55, "%"
        )
        phantom.content.addLayout(row)
        row, self._brain_intensity, self._brain_intensity_val = _slider_row(
            "Brain intensity", 5, 100, 35, "%"
        )
        phantom.content.addLayout(row)
        self._show_outline = QCheckBox("Show brain outline")
        self._show_outline.setChecked(True)
        phantom.content.addWidget(self._show_outline)
        self._tumor_enabled = QCheckBox("Enable tumor")
        self._tumor_enabled.setChecked(True)
        phantom.content.addWidget(self._tumor_enabled)
        row, self._tumor_x, self._tumor_x_val = _slider_row("Tumor X", -70, 70, 18)
        phantom.content.addLayout(row)
        row, self._tumor_y, self._tumor_y_val = _slider_row("Tumor Y", -70, 70, 12)
        phantom.content.addLayout(row)
        row, self._tumor_radius, self._tumor_radius_val = _slider_row(
            "Tumor radius", 3, 30, 10, "%"
        )
        phantom.content.addLayout(row)
        row, self._tumor_intensity, self._tumor_intensity_val = _slider_row(
            "Tumor intensity", 10, 200, 100, "%"
        )
        phantom.content.addLayout(row)
        form.addWidget(phantom)

        acquisition = CollapsibleSection("PET acquisition", "parameter")
        self._start_btn = QPushButton("Start Acquisition")
        self._start_btn.setObjectName("PrimaryButton")
        self._pause_btn = QPushButton("Pause")
        self._pause_btn.setEnabled(False)
        self._reset_acq_btn = QPushButton("Reset")
        acquisition.content.addWidget(self._start_btn)
        btn_row = QHBoxLayout()
        btn_row.addWidget(self._pause_btn)
        btn_row.addWidget(self._reset_acq_btn)
        acquisition.content.addLayout(btn_row)
        row, self._n_events, self._n_events_val = _slider_row(
            "Annihilations", 1000, 20_000, 5000
        )
        self._n_events.setSingleStep(500)
        self._n_events.setPageStep(1000)
        self._n_events.setToolTip("How many annihilation events to simulate.")
        acquisition.content.addLayout(row)
        row, self._efficiency, self._efficiency_val = _slider_row(
            "Detector efficiency", 20, 100, 80, "%"
        )
        self._efficiency.setToolTip(
            "Probability that one 511 keV photon is detected. An ideal coincidence needs both photons."
        )
        acquisition.content.addLayout(row)
        speed_name = QLabel("Animation speed")
        speed_name.setObjectName("HintLabel")
        self._speed = QComboBox()
        self._speed.addItems(["Slow", "Normal", "Fast"])
        self._speed.setCurrentText("Normal")
        self._speed.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        acquisition.content.addWidget(speed_name)
        acquisition.content.addWidget(self._speed)
        self._use_seed = QCheckBox("Use a fixed random seed")
        self._use_seed.setChecked(False)
        self._seed_edit = QLineEdit("42")
        self._seed_edit.setEnabled(False)
        self._seed_edit.setPlaceholderText("integer seed")
        acquisition.content.addWidget(self._use_seed)
        acquisition.content.addWidget(self._seed_edit)
        self._progress = QProgressBar()
        self._progress.setRange(0, 1000)
        self._progress.setValue(0)
        self._progress.setFormat("Idle")
        self._progress.setToolTip("Fraction of the requested annihilations that have been simulated.")
        acquisition.content.addWidget(self._progress)
        form.addWidget(acquisition)

        atten = CollapsibleSection("Attenuation", "parameter", expanded=False)
        self._atten_on = QCheckBox("511 keV tissue attenuation")
        self._atten_on.setToolTip(
            "Each photon survives with exp(-μ L) along its path through the brain ellipse. "
            "μ is a transmission map. It does not use the tumor uptake."
        )
        atten.content.addWidget(self._atten_on)
        self._mu = QDoubleSpinBox()
        self._mu.setRange(0.0, 0.5)
        self._mu.setDecimals(3)
        self._mu.setSingleStep(0.004)
        self._mu.setValue(0.096)
        self._mu.setSuffix(" /cm")
        self._mu.setMinimumWidth(0)
        self._mu.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._mu.setToolTip("Narrow-beam soft tissue at 511 keV. The default is 0.096 per cm.")
        self._cm = QDoubleSpinBox()
        self._cm.setRange(1.0, 30.0)
        self._cm.setDecimals(1)
        self._cm.setSingleStep(1.0)
        self._cm.setValue(10.0)
        self._cm.setSuffix(" cm")
        self._cm.setMinimumWidth(0)
        self._cm.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self._cm.setToolTip("Centimeters represented by a field-of-view radius of 1.")
        atten.content.addWidget(QLabel("μ (cm⁻¹)"))
        atten.content.addWidget(self._mu)
        atten.content.addWidget(QLabel("Centimeters per FOV radius"))
        atten.content.addWidget(self._cm)
        self._atten_study_btn = QPushButton("Compare attenuation")
        self._atten_study_btn.setToolTip(
            "Uses the same emissions and detection attempts. "
            "Shows the μ map, the ideal image, the attenuated image, and the corrected image. "
            "Correction reweights measured counts. It does not restore absorbed photons."
        )
        atten.content.addWidget(self._atten_study_btn)
        form.addWidget(atten)

        measured = CollapsibleSection("Live statistics", "measured")
        cards = QGridLayout()
        cards.setSpacing(6)
        annihilations = MetricCard("Annihilations")
        coincidences = MetricCard("Coincidences")
        self._stat_total = annihilations.value
        self._stat_coin = coincidences.value
        self._stat_total.setText("0")
        self._stat_coin.setText("0")
        cards.addWidget(annihilations, 0, 0)
        cards.addWidget(coincidences, 0, 1)
        measured.content.addLayout(cards)
        self._stat_rate = QLabel("Rate —")
        self._stat_progress = QLabel("Progress 0%")
        for lab in (self._stat_rate, self._stat_progress):
            lab.setObjectName("HintLabel")
            lab.setWordWrap(True)
            measured.content.addWidget(lab)
        form.addWidget(measured)

        display = CollapsibleSection("Detector and display", "parameter")
        cmap_row = QHBoxLayout()
        cmap_row.addWidget(QLabel("Colormap"))
        self._colormap = QComboBox()
        self._colormap.addItems(["hot", "inferno", "magma", "viridis", "gray"])
        cmap_row.addWidget(self._colormap, stretch=1)
        display.content.addLayout(cmap_row)
        self._show_detectors = QCheckBox("Show detector crystals")
        self._show_detectors.setChecked(True)
        display.content.addWidget(self._show_detectors)
        self._show_labels = QCheckBox("Label detectors (every 8th)")
        self._show_labels.setChecked(False)
        display.content.addWidget(self._show_labels)
        form.addWidget(display)

        form.addStretch(1)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        defaults_row = QHBoxLayout()
        self._reset_btn = QPushButton("Reset phantom defaults")
        self._reset_btn.clicked.connect(self._on_reset_phantom)
        defaults_row.addWidget(self._reset_btn)
        defaults_row.addStretch(1)
        root.addLayout(defaults_row)

        self._brain_radius.setToolTip("Brain disk radius as a percent of the field of view.")
        self._brain_intensity.setToolTip("Relative background uptake of healthy brain tissue.")
        self._tumor_x.setToolTip("Tumor center, horizontal, in percent of the field of view.")
        self._tumor_y.setToolTip("Tumor center, vertical, in percent of the field of view.")
        self._tumor_radius.setToolTip("Tumor disk radius as a percent of the field of view.")
        self._tumor_intensity.setToolTip("Tumor uptake relative to a reference of 100%.")
        self._n_events.setToolTip("Number of positron annihilations to simulate.")
        self._efficiency.setToolTip(
            "The probability that a photon is successfully detected. "
            "An ideal coincidence needs both photons, so the expected rate scales as ε²."
        )
        self._speed.setToolTip("How quickly the acquisition animation advances. Does not change the physics.")
        self._use_seed.setToolTip("Fix the random number generator so the same acquisition can be repeated.")
        self._pause_btn.setToolTip("Pause the acquisition. Parameters stay as they are.")
        self._reset_acq_btn.setToolTip("Discard recorded coincidences and return the counters to zero.")
        self._start_btn.setToolTip("Simulate annihilations and record coincidences on the detector ring.")

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
        self._atten_on.toggled.connect(self._emit_attenuation)
        self._mu.valueChanged.connect(self._emit_attenuation)
        self._cm.valueChanged.connect(self._emit_attenuation)
        self._brain_radius.valueChanged.connect(self._emit_attenuation)
        self._atten_study_btn.clicked.connect(self.attenuation_study_requested.emit)

        self._building = False
        self._emit_phantom_state()
        self._emit_acq_config()
        self._emit_attenuation()
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

    def attenuation_config(self) -> AttenuationConfig:
        return AttenuationConfig(
            enabled=self._atten_on.isChecked(),
            mu_per_cm=float(self._mu.value()),
            cm_per_fov=float(self._cm.value()),
            brain_radius=self._brain_radius.value() / 100.0,
        )

    def study_seed(self) -> int:
        config = self.acquisition_config()
        if config.use_seed and config.seed is not None:
            return int(config.seed)
        return 7

    def set_study_busy(self, busy: bool) -> None:
        self._atten_study_btn.setEnabled(not busy)
        self._atten_study_btn.setText("Comparing…" if busy else "Compare attenuation")

    def _emit_attenuation(self, *_args) -> None:
        if self._building:
            return
        self.attenuation_changed.emit(self.attenuation_config())

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
        self._stat_total.setText(f"{stats.total_events:,}")
        if stats.effects_enabled:
            self._stat_coin.setText(f"{stats.coincidences:,}")
            self._stat_rate.setText(
                f"Accepted {stats.coincidence_percent:.1f}%. "
                f"Ideal ε² {stats.expected_coincidence_percent:.1f}%."
            )
        else:
            self._stat_coin.setText(f"{stats.coincidences:,}")
            self._stat_rate.setText(
                f"Rate {stats.coincidence_percent:.1f}%. "
                f"Ideal ε² {stats.expected_coincidence_percent:.1f}%."
            )
        self._stat_progress.setText(f"Progress: {stats.progress_percent:.1f}%")
        self._progress.setValue(int(round(stats.progress_percent * 10)))
        if stats.running or stats.paused:
            self._progress.setFormat(f"{stats.progress_percent:.0f}% of annihilations")
        elif stats.finished:
            self._progress.setFormat("Acquisition complete")
        else:
            self._progress.setFormat("Idle")

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
