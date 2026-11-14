"""Stage 7 controls and visualizations that do not share other canvases."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFormLayout,
    QFrame,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSlider,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from petsim.physics.acquisition import AcquisitionStats
from petsim.physics.detector_effects import DetectorPhysicsConfig
from petsim.reconstruction.pipeline import ReconstructionResult
from petsim.theme import COLORS


def _style_axes(ax) -> None:
    ax.set_facecolor(COLORS["panel"])
    ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
    ax.xaxis.label.set_color(COLORS["text"])
    ax.yaxis.label.set_color(COLORS["text"])
    ax.title.set_color(COLORS["text"])
    for spine in ax.spines.values():
        spine.set_color(COLORS["border"])


class EnergySpectrumCanvas(FigureCanvasQTAgg):
    """Measured photon energies and the current discrimination window."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(5.2, 2.4), dpi=100, facecolor=COLORS["bg"])
        super().__init__(self.figure)
        self.setParent(parent)
        self._ax = self.figure.add_subplot(111)
        self.show_placeholder("Acquire events to see measured photon energies.")

    def show_placeholder(self, message: str) -> None:
        self._ax.clear()
        _style_axes(self._ax)
        self._ax.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["text_muted"], wrap=True)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.figure.tight_layout()
        self.draw_idle()

    def show_spectrum(self, energies_kev: np.ndarray, low_kev: float, high_kev: float) -> None:
        self._ax.clear()
        _style_axes(self._ax)
        values = np.asarray(energies_kev, dtype=np.float64)
        values = values[np.isfinite(values)]
        if values.size == 0:
            self.show_placeholder("No measured photon energies yet.")
            return
        bins = np.linspace(0.0, 800.0, 81)
        self._ax.hist(values, bins=bins, color=COLORS["detector"], alpha=0.9)
        self._ax.axvline(low_kev, color=COLORS["warn"], linestyle="--", linewidth=1.4, label="Window low")
        self._ax.axvline(high_kev, color=COLORS["tumor_hot"], linestyle="--", linewidth=1.4, label="Window high")
        self._ax.set_xlim(0.0, 800.0)
        self._ax.set_xlabel("Measured photon energy (keV)")
        self._ax.set_ylabel("Photons")
        self._ax.set_title("Measured energy distribution")
        self._ax.legend(facecolor=COLORS["panel"], edgecolor=COLORS["border"], labelcolor=COLORS["text"], fontsize=8)
        self.figure.tight_layout()
        self.draw_idle()


class CoincidenceClassCanvas(FigureCanvasQTAgg):
    """Accepted true, scattered, and random counts on their own axes."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(5.2, 2.4), dpi=100, facecolor=COLORS["bg"])
        super().__init__(self.figure)
        self.setParent(parent)
        self._ax = self.figure.add_subplot(111)
        self.show_counts(AcquisitionStats())

    def show_counts(self, stats: AcquisitionStats) -> None:
        self._ax.clear()
        _style_axes(self._ax)
        labels = ["True\nunscattered", "Scattered", "Random"]
        values = [
            stats.accepted_unscattered,
            stats.scattered_coincidences,
            stats.random_coincidences,
        ]
        colors = [COLORS["accent"], COLORS["warn"], COLORS["detector"]]
        positions = np.arange(len(labels))
        self._ax.bar(positions, values, color=colors)
        self._ax.set_xticks(positions, labels)
        self._ax.set_ylabel("Accepted coincidences")
        self._ax.set_title("Accepted coincidence classes")
        self.figure.tight_layout()
        self.draw_idle()


class IdealDegradedCanvas(FigureCanvasQTAgg):
    """Side-by-side reconstructions. Not the workstation reconstruction canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(5.6, 2.8), dpi=100, facecolor=COLORS["bg"])
        super().__init__(self.figure)
        self.setParent(parent)
        self._axes = self.figure.subplots(1, 2)
        self.show_placeholder("Reconstruct to compare ideal and degraded measurements.")

    def show_placeholder(self, message: str) -> None:
        for ax in self._axes:
            ax.clear()
            _style_axes(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        self._axes[0].text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=COLORS["text_muted"],
            transform=self._axes[0].transAxes,
        )
        self.figure.tight_layout()
        self.draw_idle()

    def show_pair(
        self,
        ideal: np.ndarray,
        degraded: np.ndarray,
        *,
        ideal_title: str,
        degraded_title: str,
    ) -> None:
        titles = (ideal_title, degraded_title)
        images = (np.asarray(ideal, dtype=np.float64), np.asarray(degraded, dtype=np.float64))
        for ax, image, title in zip(self._axes, images, titles, strict=True):
            ax.clear()
            _style_axes(ax)
            finite = image[np.isfinite(image)]
            vmax = float(np.percentile(finite, 99)) if finite.size else 1.0
            if vmax <= 0.0:
                vmax = 1.0
            ax.imshow(
                np.clip(image, 0.0, None),
                cmap="hot",
                origin="upper",
                vmin=0.0,
                vmax=vmax,
                extent=(-1.0, 1.0, -1.0, 1.0),
            )
            ax.set_title(title, fontsize=9)
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.tight_layout()
        self.draw_idle()


def _slider_row(slider: QSlider, readout: QLabel) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.addWidget(slider, stretch=1)
    readout.setMinimumWidth(72)
    layout.addWidget(readout)
    return row


class DetectorPhysicsPanel(QWidget):
    """Detector-effect controls plus three dedicated visualizations."""

    config_changed = Signal(object)
    compare_requested = Signal()
    cancel_compare_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._building = True
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll)

        title = QLabel("Detector Physics")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        hint = QLabel(
            "Effects stay off until you enable them, which preserves the ideal "
            "Stage 2 coincidence model. The acceptance window uses the measured "
            "arrival-time difference |Δt_measured| = |Δt_flight + residual|. "
            "The residual after subtracting the true flight time is diagnostic "
            "only. Scatter is a single 2-D Compton cartoon, not a patient simulation. "
            "Timing does not change the reconstruction algorithm."
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        self._enabled = QCheckBox("Enable detector effects")
        self._enabled.setChecked(False)
        layout.addWidget(self._enabled)

        form = QFormLayout()
        self._energy_fwhm = QSlider(Qt.Orientation.Horizontal)
        self._energy_fwhm.setRange(0, 30)
        self._energy_fwhm.setValue(12)
        self._energy_fwhm_val = QLabel("12%")
        form.addRow("Energy resolution FWHM", _slider_row(self._energy_fwhm, self._energy_fwhm_val))
        self._window_low = QSpinBox()
        self._window_low.setRange(100, 511)
        self._window_low.setValue(435)
        self._window_low.setSuffix(" keV")
        form.addRow("Energy window low", self._window_low)
        self._window_high = QSpinBox()
        self._window_high.setRange(511, 800)
        self._window_high.setValue(650)
        self._window_high.setSuffix(" keV")
        form.addRow("Energy window high", self._window_high)
        self._timing_fwhm = QSlider(Qt.Orientation.Horizontal)
        self._timing_fwhm.setRange(0, 2000)
        self._timing_fwhm.setValue(400)
        self._timing_fwhm_val = QLabel("400 ps")
        form.addRow("Timing resolution FWHM", _slider_row(self._timing_fwhm, self._timing_fwhm_val))
        self._timing_window = QSpinBox()
        self._timing_window.setRange(1, 20000)
        self._timing_window.setSingleStep(100)
        self._timing_window.setValue(4000)
        self._timing_window.setSuffix(" ps")
        form.addRow("Timing window |Δt_measured|", self._timing_window)
        self._random = QSlider(Qt.Orientation.Horizontal)
        self._random.setRange(0, 100)
        self._random.setValue(0)
        self._random_val = QLabel("0%")
        form.addRow("Random coincidence chance", _slider_row(self._random, self._random_val))
        self._scatter = QSlider(Qt.Orientation.Horizontal)
        self._scatter.setRange(0, 100)
        self._scatter.setValue(0)
        self._scatter_val = QLabel("0%")
        form.addRow("Scatter probability / photon", _slider_row(self._scatter, self._scatter_val))
        layout.addLayout(form)

        self._live_mode = QLabel("Live: ideal acquisition (detector effects off).")
        self._live_accounting = QLabel(
            "Annihilations 0 · singles 0 · undetected 0 · accepted 0"
        )
        self._live_rejected = QLabel("Energy rejected 0 · timing rejected 0 (not double-counted)")
        for label in (self._live_mode, self._live_accounting, self._live_rejected):
            label.setObjectName("HintLabel")
            label.setWordWrap(True)
            layout.addWidget(label)

        energy_box = QGroupBox("Energy spectrum")
        energy_layout = QVBoxLayout(energy_box)
        self.energy_canvas = EnergySpectrumCanvas(self)
        self.energy_canvas.setMinimumHeight(220)
        energy_layout.addWidget(self.energy_canvas)
        layout.addWidget(energy_box)

        class_box = QGroupBox("Accepted coincidence classes")
        class_layout = QVBoxLayout(class_box)
        self.class_canvas = CoincidenceClassCanvas(self)
        self.class_canvas.setMinimumHeight(220)
        class_layout.addWidget(self.class_canvas)
        layout.addWidget(class_box)

        compare_box = QGroupBox("Ideal versus degraded reconstruction")
        compare_layout = QVBoxLayout(compare_box)
        compare_note = QLabel(
            "Both images use the selected algorithm on ring hits only. "
            "Ideal uses efficiency-detected pairs before scatter. "
            "Degraded uses accepted true, scattered, and random coincidences. "
            "Intensity scales are independent. This canvas is separate from "
            "the workstation reconstruction, convergence, count-comparison, "
            "and experiment plots."
        )
        compare_note.setWordWrap(True)
        compare_note.setObjectName("HintLabel")
        compare_layout.addWidget(compare_note)
        controls = QHBoxLayout()
        self._method = QComboBox()
        self._method.addItems(["FBP", "MLEM"])
        self._iterations = QComboBox()
        self._iterations.addItems(["1", "5", "10", "20"])
        self._iterations.setCurrentText("5")
        self._compare_btn = QPushButton("Reconstruct ideal vs degraded")
        self._compare_btn.setObjectName("PrimaryButton")
        self._cancel_compare_btn = QPushButton("Cancel comparison")
        self._cancel_compare_btn.setEnabled(False)
        controls.addWidget(QLabel("Algorithm"))
        controls.addWidget(self._method)
        controls.addWidget(QLabel("MLEM iterations"))
        controls.addWidget(self._iterations)
        controls.addWidget(self._compare_btn)
        controls.addWidget(self._cancel_compare_btn)
        controls.addStretch(1)
        compare_layout.addLayout(controls)
        self._compare_status = QLabel("No comparison yet.")
        self._compare_status.setObjectName("HintLabel")
        self._compare_status.setWordWrap(True)
        compare_layout.addWidget(self._compare_status)
        self.comparison_canvas = IdealDegradedCanvas(self)
        self.comparison_canvas.setMinimumHeight(260)
        compare_layout.addWidget(self.comparison_canvas)
        layout.addWidget(compare_box)
        layout.addStretch(1)

        self._enabled.toggled.connect(self._emit)
        self._energy_fwhm.valueChanged.connect(self._emit)
        self._window_low.valueChanged.connect(self._emit)
        self._window_high.valueChanged.connect(self._emit)
        self._timing_fwhm.valueChanged.connect(self._emit)
        self._timing_window.valueChanged.connect(self._emit)
        self._random.valueChanged.connect(self._emit)
        self._scatter.valueChanged.connect(self._emit)
        self._compare_btn.clicked.connect(self.compare_requested.emit)
        self._cancel_compare_btn.clicked.connect(self.cancel_compare_requested.emit)
        self._building = False
        self._emit()

    def current_config(self) -> DetectorPhysicsConfig:
        return DetectorPhysicsConfig(
            enabled=self._enabled.isChecked(),
            energy_fwhm_percent=float(self._energy_fwhm.value()),
            energy_window_low_kev=float(self._window_low.value()),
            energy_window_high_kev=float(self._window_high.value()),
            timing_fwhm_ps=float(self._timing_fwhm.value()),
            timing_window_ps=float(self._timing_window.value()),
            random_fraction=self._random.value() / 100.0,
            scatter_probability=self._scatter.value() / 100.0,
        ).clamp()

    def comparison_method(self) -> str:
        return "mlem" if self._method.currentText().upper() == "MLEM" else "fbp"

    def comparison_iterations(self) -> int:
        return int(self._iterations.currentText())

    def update_stats(
        self,
        stats: AcquisitionStats,
        energies_kev: np.ndarray | None = None,
        *,
        redraw_plots: bool = True,
    ) -> None:
        if self._enabled.isChecked():
            self._live_mode.setText(
                "Live: detector effects apply to new annihilations. "
                "Generated annihilations, singles, accepted coincidences, and rejections are separate."
            )
        else:
            self._live_mode.setText("Live: ideal acquisition (detector effects off).")
        self._live_accounting.setText(
            f"Annihilations {stats.total_events:,} · singles {stats.detected_singles:,} · "
            f"undetected {stats.undetected:,} · accepted {stats.coincidences:,} "
            f"(true {stats.accepted_unscattered:,}, scattered {stats.scattered_coincidences:,}, "
            f"random {stats.random_coincidences:,})"
        )
        self._live_rejected.setText(
            f"Energy rejected {stats.energy_rejected:,} · "
            f"timing rejected {stats.timing_rejected:,} "
            "(a pair that fails energy is not also counted as a timing rejection)"
        )
        if not redraw_plots:
            return
        self.class_canvas.show_counts(stats)
        config = self.current_config()
        if energies_kev is None:
            return
        self.energy_canvas.show_spectrum(
            energies_kev,
            config.energy_window_low_kev,
            config.energy_window_high_kev,
        )

    def show_comparison(self, ideal: ReconstructionResult, degraded: ReconstructionResult) -> None:
        self._cancel_compare_btn.setEnabled(False)
        self._compare_btn.setEnabled(True)
        if ideal.image.size == 0 and degraded.image.size == 0:
            self.comparison_canvas.show_placeholder("No endpoints were available to reconstruct.")
            self._compare_status.setText(ideal.message or degraded.message or "Nothing to reconstruct.")
            return
        method = ideal.method.upper()
        self.comparison_canvas.show_pair(
            ideal.image,
            degraded.image,
            ideal_title=f"Ideal {method} ({ideal.n_coincidences:,} pairs)",
            degraded_title=f"Degraded {method} ({degraded.n_coincidences:,} pairs)",
        )
        note = "Comparison finished."
        if ideal.cancelled or degraded.cancelled:
            note = "Comparison cancelled. Partial images are shown only if that reconstruction finished."
        self._compare_status.setText(
            f"{note} {ideal.message} {degraded.message}".strip()
        )

    def show_comparison_busy(self, message: str) -> None:
        self._compare_btn.setEnabled(False)
        self._cancel_compare_btn.setEnabled(True)
        self._compare_status.setText(message)

    def show_comparison_message(self, message: str) -> None:
        self._compare_btn.setEnabled(True)
        self._cancel_compare_btn.setEnabled(False)
        self._compare_status.setText(message)

    def clear_visuals(self) -> None:
        self.energy_canvas.show_placeholder("Acquire events to see measured photon energies.")
        self.class_canvas.show_counts(AcquisitionStats())
        self.comparison_canvas.show_placeholder(
            "Reconstruct to compare ideal and degraded measurements."
        )
        self.show_comparison_message("Comparison cleared.")

    def _emit(self, *_args) -> None:
        self._energy_fwhm_val.setText(f"{self._energy_fwhm.value()}%")
        self._timing_fwhm_val.setText(f"{self._timing_fwhm.value()} ps")
        self._random_val.setText(f"{self._random.value()}%")
        self._scatter_val.setText(f"{self._scatter.value()}%")
        if self._building:
            return
        self.config_changed.emit(self.current_config())
