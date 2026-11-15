"""Stage 8 comparison of ideal, uncorrected, random-corrected, and scatter-scaled images."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QFrame,
    QProgressBar,
    QHeaderView,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from petsim.corrections.metrics import ImageQualityMetrics
from petsim.corrections.reconstruct import (
    MODE_BOTH,
    MODE_IDEAL,
    MODE_RANDOM,
    MODE_SCATTER,
    MODE_UNCORRECTED,
)
from petsim.corrections.studies import CorrectionStudyConfig, CorrectionStudyResult, CorrectionTrial
from petsim.corrections_worker import ComparisonBundle
from petsim.models.phantom import PhantomState
from petsim.reconstruction.pipeline import ReconstructionResult
from petsim.theme import COLORS
from petsim.widgets.chrome import numeric_item, section_row, style_axes

MODE_TITLES = {
    MODE_IDEAL: "Ideal",
    MODE_UNCORRECTED: "Uncorrected",
    MODE_RANDOM: "Random correction",
    MODE_SCATTER: "Scatter scaling",
    MODE_BOTH: "Random + scatter scale",
}
DISPLAY_MODES = (
    MODE_IDEAL,
    MODE_UNCORRECTED,
    MODE_RANDOM,
    MODE_SCATTER,
    MODE_BOTH,
)


def _style_ax(ax) -> None:
    style_axes(ax, grid=True)


class _Plot(FigureCanvasQTAgg):
    def __init__(self, title: str, ylabel: str, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(4.4, 2.6), dpi=100, facecolor=COLORS["bg"])
        super().__init__(self.figure)
        self.setParent(parent)
        self._ax = self.figure.add_subplot(111)
        self._title = title
        self._ylabel = ylabel
        self.setToolTip(title)
        self.show_placeholder("Run a correction study to fill this plot.")

    def show_placeholder(self, message: str) -> None:
        self._ax.clear()
        _style_ax(self._ax)
        self._ax.set_title(self._title, fontsize=9)
        self._ax.text(0.5, 0.5, message, ha="center", va="center", color=COLORS["text_muted"], transform=self._ax.transAxes, wrap=True)
        self._ax.set_xticks([])
        self._ax.set_yticks([])
        self.figure.tight_layout()
        self.draw_idle()

    def show_series(self, series: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], xlabel: str) -> None:
        self._ax.clear()
        _style_ax(self._ax)
        self._ax.set_title(self._title, fontsize=9)
        self._ax.set_xlabel(xlabel)
        self._ax.set_ylabel(self._ylabel)
        if not series:
            self.show_placeholder("Not enough finished trials for this slice.")
            return
        for name, (xs, means, stds) in series.items():
            self._ax.errorbar(xs, means, yerr=stds, marker="o", capsize=3, label=name)
        self._ax.legend(fontsize=7, facecolor=COLORS["panel"], edgecolor=COLORS["border"], labelcolor=COLORS["text"])
        self.figure.tight_layout()
        self.draw_idle()


class ComparisonCanvas(FigureCanvasQTAgg):
    """Up to five reconstructions with one shared intensity scale."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self.figure = Figure(figsize=(8.4, 2.5), dpi=100, facecolor=COLORS["bg"])
        super().__init__(self.figure)
        self.setParent(parent)
        self._axes = self.figure.subplots(1, 5)
        self.show_placeholder("Compare corrections after an acquisition.")

    def show_placeholder(self, message: str) -> None:
        for ax in self._axes:
            ax.clear()
            _style_ax(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.texts.clear()
        self.figure.text(
            0.5,
            0.45,
            message,
            ha="center",
            va="center",
            color=COLORS["text_muted"],
            fontsize=10,
            wrap=True,
        )
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05, wspace=0.2)
        self.draw_idle()

    def show_images(self, images: list[tuple[str, np.ndarray]]) -> None:
        finite = []
        for _title, image in images:
            values = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
            finite.append(values[np.isfinite(values)])
        stacked = np.concatenate(finite) if any(part.size for part in finite) else np.zeros(1)
        positive = stacked[stacked > 0]
        vmax = float(np.percentile(positive, 99)) if positive.size else 1.0
        vmax = max(vmax, 1e-6)
        self.figure.texts.clear()
        for ax, (title, image) in zip(self._axes, images, strict=False):
            ax.clear()
            _style_ax(ax)
            ax.imshow(
                np.maximum(np.asarray(image, dtype=np.float64), 0.0),
                cmap="hot",
                origin="lower",
                vmin=0.0,
                vmax=vmax,
                extent=(-1.0, 1.0, -1.0, 1.0),
            )
            ax.set_title(title, fontsize=8)
            ax.set_xticks([])
            ax.set_yticks([])
        for ax in self._axes[len(images) :]:
            ax.clear()
            _style_ax(ax)
            ax.set_xticks([])
            ax.set_yticks([])
        self.figure.subplots_adjust(left=0.02, right=0.98, top=0.82, bottom=0.05, wspace=0.2)
        self.draw_idle()


def _fmt(value: float | None, digits: int = 3) -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}"


def _mean_std(values: list[float | None]) -> tuple[float | None, float]:
    usable = [float(value) for value in values if value is not None and np.isfinite(value)]
    if not usable:
        return None, 0.0
    if len(usable) == 1:
        return usable[0], 0.0
    arr = np.asarray(usable, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


class CorrectionsPanel(QWidget):
    """Compare correction modes and run a matched-acquisition study."""

    compare_requested = Signal()
    cancel_compare_requested = Signal()
    study_requested = Signal()
    cancel_study_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        body = QWidget()
        layout = QVBoxLayout(body)
        scroll.setWidget(body)
        root.addWidget(scroll)

        title = QLabel("PET Corrections & Validation")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        hint = QLabel(
            "Random coincidence correction subtracts a uniform crystal-pair background "
            "at the configured rate. That changes the sinogram’s shape. "
            "Global scatter scaling is a simplified educational approximation: a "
            "300–420 keV side window estimates how many photopeak prompts are scattered, "
            "and every remaining bin is multiplied by that one fraction. It estimates "
            "the total, not the spatial distribution. NRMSE, CRC, and localization "
            "error divide out a positive scale, so those scores can stay the same "
            "while the shared color scale shows a dimmer image. Neither step reads "
            "event labels or the tumor position."
        )
        hint.setWordWrap(True)
        hint.setObjectName("HintLabel")
        hint.setToolTip(
            "Total contamination is a count. A spatial correction would be a "
            "different sinogram shape. The side-window lines are broader than the "
            "small-angle scatter that stays in the photopeak, so they are not used "
            "as that shape. Prompts are clipped at zero before FBP and MLEM. "
            "RMSE uses a nonnegative least-squares scale against the phantom raster. "
            "NRMSE divides that RMSE by the RMS of the activity map."
        )
        layout.addWidget(hint)

        controls = QHBoxLayout()
        self._method = QComboBox()
        self._method.addItems(["FBP", "MLEM"])
        self._method.setToolTip("Algorithm used for every image in the comparison and the study.")
        self._iterations = QComboBox()
        self._iterations.addItems(["1", "5", "10"])
        self._iterations.setCurrentText("5")
        self._iterations.setToolTip("MLEM iterations. Ignored by FBP.")
        self._compare = QPushButton("Compare five reconstructions")
        self._compare.setObjectName("PrimaryButton")
        self._compare.setToolTip(
            "Ideal, uncorrected, random coincidence correction, global scatter "
            "scaling, and both. Same acquisition and the same algorithm. "
            "Shared color scale. Scatter scaling changes brightness, not shape."
        )
        self._cancel_compare = QPushButton("Cancel comparison")
        self._cancel_compare.setObjectName("DangerButton")
        self._cancel_compare.setToolTip("Stop the five-image comparison. Finished images stay until you reset.")
        self._cancel_compare.setEnabled(False)
        controls.addWidget(QLabel("Algorithm"))
        controls.addWidget(self._method)
        controls.addWidget(QLabel("MLEM iterations"))
        controls.addWidget(self._iterations)
        controls.addWidget(self._compare)
        controls.addWidget(self._cancel_compare)
        controls.addStretch(1)
        layout.addLayout(controls)
        self._compare_status = QLabel("Acquire coincidences, then compare.")
        self._compare_status.setWordWrap(True)
        self._compare_status.setObjectName("HintLabel")
        layout.addWidget(self._compare_status)
        self.comparison_canvas = ComparisonCanvas(self)
        self.comparison_canvas.setMinimumHeight(240)
        layout.addWidget(self.comparison_canvas)

        self._table = QTableWidget(0, 11)
        self._table.setHorizontalHeaderLabels(
            [
                "Mode",
                "Loc. error",
                "TBR",
                "Contrast",
                "Bg CV",
                "RMSE",
                "NRMSE",
                "CRC",
                "Runtime (s)",
                "Random counts",
                "Scatter counts",
            ]
        )
        self._table.setMinimumHeight(160)
        self._table.setAlternatingRowColors(True)
        self._table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        self._table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(section_row("Image scores and estimate columns"))
        self._table.setToolTip(
            "Scores are computed from the image in that row. "
            "Localization, TBR, contrast, and CV use Stage 4 ROIs. "
            "RMSE and NRMSE are scale-normalized, so a global scatter scale "
            "does not move them. CRC is measured contrast divided by phantom-ROI contrast."
        )
        random_header = self._table.horizontalHeaderItem(9)
        scatter_header = self._table.horizontalHeaderItem(10)
        if random_header is not None:
            random_header.setToolTip(
                "Expected random coincidences removed from the sinogram. "
                "This subtraction changes the spatial distribution."
            )
        if scatter_header is not None:
            scatter_header.setToolTip(
                "Estimated photopeak scatter total from the side window. "
                "Applied as one scale on every bin, not as a spatial template."
            )
        layout.addWidget(self._table)

        layout.addWidget(section_row("Matched study settings", "parameter"))
        study = QGroupBox("Matched correction study")
        study.setToolTip(
            "Each trial simulates one prompt list, then reconstructs every selected mode from that list."
        )
        form = QGridLayout(study)
        self._count = QSpinBox()
        self._count.setRange(200, 5000)
        self._count.setValue(800)
        self._count.setToolTip("Accepted coincidence target for each simulated list.")
        form.addWidget(QLabel("Coincidences"), 0, 0)
        form.addWidget(self._count, 0, 1)
        self._trials = QSpinBox()
        self._trials.setRange(1, 5)
        self._trials.setValue(1)
        form.addWidget(QLabel("Trials"), 0, 2)
        form.addWidget(self._trials, 0, 3)
        self._random_boxes = []
        random_row = QHBoxLayout()
        for value in (0.0, 0.1, 0.25):
            box = QCheckBox(f"{value:.2f}")
            box.setChecked(value in (0.0, 0.25))
            box.setToolTip("Random insertion probability per annihilation.")
            self._random_boxes.append((value, box))
            random_row.addWidget(box)
        form.addWidget(QLabel("Random probability"), 1, 0)
        form.addLayout(random_row, 1, 1, 1, 3)
        self._scatter_boxes = []
        scatter_row = QHBoxLayout()
        for value in (0.0, 0.1, 0.25):
            box = QCheckBox(f"{value:.2f}")
            box.setChecked(value in (0.0, 0.25))
            box.setToolTip("Per-photon scatter probability in the Stage 7 cartoon.")
            self._scatter_boxes.append((value, box))
            scatter_row.addWidget(box)
        form.addWidget(QLabel("Scatter probability"), 2, 0)
        form.addLayout(scatter_row, 2, 1, 1, 3)
        self._narrow_window = QCheckBox("Also use a narrow 500–522 keV window")
        self._narrow_window.setToolTip("Adds a second energy window. The wide window 435–650 keV is always included.")
        form.addWidget(self._narrow_window, 3, 0, 1, 4)
        buttons = QHBoxLayout()
        self._start_study = QPushButton("Start correction study")
        self._cancel_study = QPushButton("Cancel study")
        self._cancel_study.setObjectName("DangerButton")
        self._cancel_study.setToolTip("Stop the matched study. Completed trials remain plotted.")
        self._cancel_study.setEnabled(False)
        self._reset = QPushButton("Reset")
        buttons.addWidget(self._start_study)
        buttons.addWidget(self._cancel_study)
        buttons.addWidget(self._reset)
        buttons.addStretch(1)
        form.addLayout(buttons, 4, 0, 1, 4)
        layout.addWidget(study)
        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        layout.addWidget(self._progress)
        self._study_status = QLabel("The study uses detector effects so the selected random and scatter rates are actually simulated.")
        self._study_status.setWordWrap(True)
        self._study_status.setObjectName("HintLabel")
        layout.addWidget(self._study_status)

        grid = QGridLayout()
        self.plot_modes = _Plot("Modes at the strongest contamination", "NRMSE")
        self.plot_random = _Plot("Contrast vs random contamination", "Contrast")
        self.plot_scatter = _Plot("Localization error vs scatter", "Error (FOV)")
        self.plot_counts = _Plot("Image quality vs coincidence count", "NRMSE")
        self.plot_modes.setToolTip(
            "NRMSE by mode at the highest random and scatter settings. "
            "Scatter scaling can match the uncorrected NRMSE because NRMSE removes a global scale."
        )
        self.plot_random.setToolTip("Relative contrast versus random probability. Error bars are the sample standard deviation across trials.")
        self.plot_scatter.setToolTip("Tumor localization error versus scatter probability.")
        self.plot_counts.setToolTip("Scale-normalized RMSE versus accepted coincidence count. This study uses one count unless you change it.")
        for plot in (self.plot_modes, self.plot_random, self.plot_scatter, self.plot_counts):
            plot.setMinimumHeight(220)
        grid.addWidget(self.plot_modes, 0, 0)
        grid.addWidget(self.plot_random, 0, 1)
        grid.addWidget(self.plot_scatter, 1, 0)
        grid.addWidget(self.plot_counts, 1, 1)
        layout.addLayout(grid)

        self._compare.clicked.connect(self.compare_requested.emit)
        self._cancel_compare.clicked.connect(self.cancel_compare_requested.emit)
        self._start_study.clicked.connect(self.study_requested.emit)
        self._cancel_study.clicked.connect(self.cancel_study_requested.emit)
        self._reset.clicked.connect(self.reset_view)

    def comparison_method(self) -> str:
        return "mlem" if self._method.currentText().upper() == "MLEM" else "fbp"

    def comparison_iterations(self) -> int:
        return int(self._iterations.currentText())

    def build_study_config(self, phantom: PhantomState) -> CorrectionStudyConfig | None:
        randoms = tuple(value for value, box in self._random_boxes if box.isChecked())
        scatters = tuple(value for value, box in self._scatter_boxes if box.isChecked())
        if not randoms or not scatters:
            return None
        windows = [(435.0, 650.0)]
        if self._narrow_window.isChecked():
            windows.append((500.0, 522.0))
        return CorrectionStudyConfig(
            phantom=phantom,
            methods=(self.comparison_method(),),
            count_targets=(int(self._count.value()),),
            random_fractions=randoms,
            scatter_probabilities=scatters,
            energy_windows=tuple(windows),
            n_trials=int(self._trials.value()),
            n_iterations=self.comparison_iterations(),
        ).clamp()

    def show_comparison_busy(self, message: str) -> None:
        self._compare.setEnabled(False)
        self._cancel_compare.setEnabled(True)
        self._compare_status.setText(message)

    def show_comparison_message(self, message: str) -> None:
        self._compare.setEnabled(True)
        self._cancel_compare.setEnabled(False)
        self._compare_status.setText(message)

    def show_comparison(self, bundle: ComparisonBundle) -> None:
        self.show_comparison_message(bundle.message)
        images = []
        rows: list[tuple[str, ImageQualityMetrics | None, ReconstructionResult | None]] = []
        for mode in DISPLAY_MODES:
            result = bundle.images.get(mode)
            if result is None:
                continue
            images.append((MODE_TITLES[mode], result.image))
            rows.append((MODE_TITLES[mode], bundle.metrics.get(mode), result))
        if images:
            self.comparison_canvas.show_images(images)
        else:
            self.comparison_canvas.show_placeholder("No reconstruction images were returned.")
        self._fill_table(rows)

    def show_study_progress(self, done: int, total: int, message: str) -> None:
        self._start_study.setEnabled(False)
        self._cancel_study.setEnabled(True)
        self._progress.setValue(0 if total <= 0 else int(100 * done / total))
        self._study_status.setText(message)

    def show_study_result(self, result: CorrectionStudyResult) -> None:
        self._start_study.setEnabled(True)
        self._cancel_study.setEnabled(False)
        self._progress.setValue(100 if not result.cancelled else self._progress.value())
        self._study_status.setText(result.message)
        self._plot_study(result.trials)

    def show_study_error(self, message: str) -> None:
        self._start_study.setEnabled(True)
        self._cancel_study.setEnabled(False)
        self._study_status.setText(message)

    def reset_view(self) -> None:
        self.comparison_canvas.show_placeholder("Compare corrections after an acquisition.")
        self._table.setRowCount(0)
        self._compare_status.setText("Comparison cleared.")
        self._compare.setEnabled(True)
        self._cancel_compare.setEnabled(False)
        self._study_status.setText("Study results cleared.")
        self._progress.setValue(0)
        self._start_study.setEnabled(True)
        self._cancel_study.setEnabled(False)
        for plot in (self.plot_modes, self.plot_random, self.plot_scatter, self.plot_counts):
            plot.show_placeholder("Run a correction study to fill this plot.")

    def clear_comparison(self) -> None:
        self.comparison_canvas.show_placeholder("Acquisition changed. Compare again when data are ready.")
        self._table.setRowCount(0)
        self._compare.setEnabled(True)
        self._cancel_compare.setEnabled(False)

    def _fill_table(self, rows: list[tuple[str, ImageQualityMetrics | None, ReconstructionResult | None]]) -> None:
        self._table.setRowCount(len(rows))
        for row_index, (name, metrics, result) in enumerate(rows):
            values = [
                name,
                _fmt(None if metrics is None else metrics.localization_error, 4),
                _fmt(None if metrics is None else metrics.tbr),
                _fmt(None if metrics is None else metrics.contrast),
                _fmt(None if metrics is None else metrics.background_cv),
                _fmt(None if metrics is None else metrics.rmse),
                _fmt(None if metrics is None else metrics.nrmse),
                _fmt(None if metrics is None else metrics.crc),
                _fmt(None if metrics is None else metrics.runtime_s, 2),
                _fmt(None if result is None else result.random_estimate_counts, 0),
                _fmt(None if result is None else result.scatter_estimate_counts, 0),
            ]
            for column, value in enumerate(values):
                item = QTableWidgetItem(value) if column == 0 else numeric_item(value)
                self._table.setItem(row_index, column, item)

    def _plot_study(self, trials: list[CorrectionTrial]) -> None:
        finished = [trial for trial in trials if trial.status == "ok" and trial.metrics is not None]
        if not finished:
            for plot in (self.plot_modes, self.plot_random, self.plot_scatter, self.plot_counts):
                plot.show_placeholder("No successful trials to plot.")
            return
        max_random = max(trial.random_fraction for trial in finished)
        max_scatter = max(trial.scatter_probability for trial in finished)
        min_count = min(trial.count_target for trial in finished)
        method = finished[0].method

        def subset(**filters) -> list[CorrectionTrial]:
            chosen = []
            for trial in finished:
                if trial.method != method:
                    continue
                if any(getattr(trial, key) != value for key, value in filters.items()):
                    continue
                chosen.append(trial)
            return chosen

        mode_slice = subset(
            random_fraction=max_random,
            scatter_probability=max_scatter,
            count_target=min_count,
        )
        mode_x = []
        mode_mean = []
        mode_std = []
        for index, mode in enumerate(DISPLAY_MODES):
            values = [
                trial.metrics.nrmse
                for trial in mode_slice
                if trial.mode == mode and trial.metrics is not None
            ]
            mean, std = _mean_std(values)
            if mean is None:
                continue
            mode_x.append(index)
            mode_mean.append(mean)
            mode_std.append(std)
        if mode_x:
            self.plot_modes.show_series(
                {"NRMSE": (np.asarray(mode_x), np.asarray(mode_mean), np.asarray(mode_std))},
                "Mode index (ideal, uncorrected, random, scatter scale, both)",
            )
        else:
            self.plot_modes.show_placeholder("This study did not include the reference mode slice.")

        self.plot_random.show_series(
            _lines_versus(
                subset(scatter_probability=max_scatter, count_target=min_count),
                x_name="random_fraction",
                metric="contrast",
                modes=(MODE_UNCORRECTED, MODE_BOTH),
            ),
            "Random probability",
        )
        self.plot_scatter.show_series(
            _lines_versus(
                subset(random_fraction=max_random, count_target=min_count),
                x_name="scatter_probability",
                metric="localization_error",
                modes=(MODE_UNCORRECTED, MODE_BOTH),
            ),
            "Scatter probability",
        )
        self.plot_counts.show_series(
            _lines_versus(
                subset(random_fraction=max_random, scatter_probability=max_scatter),
                x_name="count_target",
                metric="nrmse",
                modes=(MODE_UNCORRECTED, MODE_BOTH),
            ),
            "Accepted coincidences",
        )


def _lines_versus(
    trials: list[CorrectionTrial],
    *,
    x_name: str,
    metric: str,
    modes: tuple[str, ...],
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]]:
    series = {}
    for mode in modes:
        grouped: dict[float, list[float | None]] = {}
        for trial in trials:
            if trial.mode != mode or trial.metrics is None:
                continue
            grouped.setdefault(float(getattr(trial, x_name)), []).append(getattr(trial.metrics, metric))
        xs = []
        means = []
        stds = []
        for x_value in sorted(grouped):
            mean, std = _mean_std(grouped[x_value])
            if mean is None:
                continue
            xs.append(x_value)
            means.append(mean)
            stds.append(std)
        if xs:
            series[MODE_TITLES.get(mode, mode)] = (
                np.asarray(xs, dtype=np.float64),
                np.asarray(means, dtype=np.float64),
                np.asarray(stds, dtype=np.float64),
            )
    return series
