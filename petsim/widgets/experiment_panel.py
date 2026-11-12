"""Stage 6 image-quality experiment workspace."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QProgressBar,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from petsim.experiments.config import DEFAULT_COUNT_TARGETS, ExperimentConfig
from petsim.experiments.export import export_experiment
from petsim.experiments.runner import ExperimentResult
from petsim.models.phantom import PhantomState
from petsim.theme import COLORS

_METHOD_COLORS = {"fbp": COLORS["detector"], "mlem": COLORS["accent"]}


def _fmt_pm(mean: float | None, std: float | None) -> str:
    if mean is None:
        return "—"
    if std is None:
        return f"{mean:.3f}"
    return f"{mean:.3f} ± {std:.3f}"


class MetricPlot(FigureCanvasQTAgg):
    """One experiment metric versus detected coincidence count."""

    def __init__(self, title: str, ylabel: str, parent: QWidget | None = None) -> None:
        self._figure = Figure(figsize=(4.0, 2.6), dpi=100, facecolor=COLORS["panel"])
        super().__init__(self._figure)
        self.setParent(parent)
        self._title = title
        self._ylabel = ylabel
        self.show_placeholder("Run an experiment to plot this metric.")

    def _style(self, ax) -> None:
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.title.set_color(COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text_muted"])
        ax.yaxis.label.set_color(COLORS["text_muted"])

    def show_placeholder(self, message: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title(self._title, fontsize=10)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=COLORS["text_muted"],
            fontsize=8,
            wrap=True,
            transform=ax.transAxes,
        )
        self._figure.tight_layout(pad=0.4)
        self.draw_idle()

    def show_series(
        self,
        rows,
        *,
        mean_attr: str,
        std_attr: str,
    ) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style(ax)
        ax.set_title(self._title, fontsize=10)
        ax.set_xlabel("Detected coincidences")
        ax.set_ylabel(self._ylabel)
        methods = []
        for row in rows:
            if row.method not in methods:
                methods.append(row.method)
        drew = False
        for method in methods:
            subset = [row for row in rows if row.method == method and getattr(row, mean_attr) is not None]
            if not subset:
                continue
            xs = [row.count_target for row in subset]
            ys = [getattr(row, mean_attr) for row in subset]
            yerr = [getattr(row, std_attr) or 0.0 for row in subset]
            ax.errorbar(
                xs,
                ys,
                yerr=yerr,
                marker="o",
                capsize=3,
                label=method.upper(),
                color=_METHOD_COLORS.get(method, COLORS["text"]),
            )
            drew = True
        if not drew:
            ax.text(
                0.5,
                0.5,
                "No successful trials for this metric.",
                ha="center",
                va="center",
                color=COLORS["text_muted"],
                transform=ax.transAxes,
            )
        else:
            ax.legend(fontsize=7, framealpha=0.35)
            ax.grid(True, color=COLORS["grid"], alpha=0.5)
        self._figure.tight_layout(pad=0.45)
        self.draw_idle()


class ExperimentPanel(QWidget):
    """Scrollable experiment setup, progress, table, and four metric plots."""

    start_requested = Signal()
    cancel_requested = Signal()
    reset_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self._result: ExperimentResult | None = None
        self._busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(8, 8, 8, 8)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setSpacing(8)
        scroll.setWidget(body)
        root.addWidget(scroll)

        title = QLabel("Image Quality Experiments")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)
        hint = QLabel(
            "Repeated educational simulations at fixed detected-coincidence counts. "
            "Error bars are the standard deviation across independent acquisitions, "
            "not MLEM convergence and not clinical performance. FBP may pre-smooth "
            "the sinogram; MLEM uses unsmoothed counts from the same trial."
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        layout.addWidget(hint)

        setup = QGroupBox("Experiment setup")
        form = QGridLayout(setup)
        self._fbp = QCheckBox("FBP")
        self._fbp.setChecked(True)
        self._mlem = QCheckBox("MLEM")
        self._mlem.setChecked(True)
        methods = QHBoxLayout()
        methods.addWidget(self._fbp)
        methods.addWidget(self._mlem)
        methods.addStretch(1)
        form.addWidget(QLabel("Methods"), 0, 0)
        form.addLayout(methods, 0, 1)

        counts = QHBoxLayout()
        self._count_boxes: dict[int, QCheckBox] = {}
        for count in DEFAULT_COUNT_TARGETS:
            box = QCheckBox(f"{count:,}")
            box.setChecked(True)
            self._count_boxes[count] = box
            counts.addWidget(box)
        form.addWidget(QLabel("Coincidences"), 1, 0)
        form.addLayout(counts, 1, 1)

        self._trials = QSpinBox()
        self._trials.setRange(1, 30)
        self._trials.setValue(3)
        form.addWidget(QLabel("Trials per count"), 2, 0)
        form.addWidget(self._trials, 2, 1)

        self._seed = QSpinBox()
        self._seed.setRange(0, 2_000_000_000)
        self._seed.setValue(1)
        form.addWidget(QLabel("Random seed"), 3, 0)
        form.addWidget(self._seed, 3, 1)

        self._iterations = QComboBox()
        self._iterations.addItems(["1", "5", "10", "20", "50"])
        self._iterations.setCurrentText("10")
        form.addWidget(QLabel("MLEM iterations"), 4, 0)
        form.addWidget(self._iterations, 4, 1)

        self._filter = QComboBox()
        self._filter.addItems(["Hann", "Ramp"])
        form.addWidget(QLabel("FBP filter"), 5, 0)
        form.addWidget(self._filter, 5, 1)

        self._efficiency = QDoubleSpinBox()
        self._efficiency.setRange(0.20, 1.0)
        self._efficiency.setSingleStep(0.05)
        self._efficiency.setValue(0.80)
        form.addWidget(QLabel("Detector efficiency"), 6, 0)
        form.addWidget(self._efficiency, 6, 1)
        layout.addWidget(setup)

        buttons = QHBoxLayout()
        self._start = QPushButton("Start Experiment")
        self._start.setObjectName("PrimaryButton")
        self._start.clicked.connect(self._on_start)
        self._cancel = QPushButton("Cancel")
        self._cancel.setEnabled(False)
        self._cancel.clicked.connect(self.cancel_requested.emit)
        self._reset = QPushButton("Reset Results")
        self._reset.clicked.connect(self._on_reset)
        self._export = QPushButton("Export CSV and PNG")
        self._export.setEnabled(False)
        self._export.clicked.connect(self._on_export)
        buttons.addWidget(self._start)
        buttons.addWidget(self._cancel)
        buttons.addWidget(self._reset)
        buttons.addWidget(self._export)
        layout.addLayout(buttons)

        self._progress = QProgressBar()
        self._progress.setRange(0, 100)
        self._progress.setValue(0)
        self._progress.setFormat("%p%")
        layout.addWidget(self._progress)
        self._status = QLabel("Choose settings and start an experiment. The phantom is taken from Simulation Controls.")
        self._status.setWordWrap(True)
        self._status.setObjectName("HintLabel")
        layout.addWidget(self._status)

        self._table = QTableWidget(0, 8)
        self._table.setHorizontalHeaderLabels(
            [
                "Count",
                "Method",
                "Trials",
                "Loc. error (FOV)",
                "TBR",
                "Contrast",
                "Bg CV",
                "Runtime (s)",
            ]
        )
        self._table.setMinimumHeight(140)
        layout.addWidget(self._table)

        grid = QGridLayout()
        self.plot_localization = MetricPlot("Localization error", "Error (FOV)")
        self.plot_tbr = MetricPlot("Tumor-to-background ratio", "TBR (dimensionless)")
        self.plot_cv = MetricPlot("Background coefficient of variation", "CV (dimensionless)")
        self.plot_runtime = MetricPlot("Reconstruction runtime", "Seconds")
        for plot in self._plots():
            plot.setMinimumHeight(220)
        grid.addWidget(self.plot_localization, 0, 0)
        grid.addWidget(self.plot_tbr, 0, 1)
        grid.addWidget(self.plot_cv, 1, 0)
        grid.addWidget(self.plot_runtime, 1, 1)
        layout.addLayout(grid)

    def _plots(self) -> list[MetricPlot]:
        return [self.plot_localization, self.plot_tbr, self.plot_cv, self.plot_runtime]

    def figures(self) -> dict:
        return {
            "localization_error": self.plot_localization.figure,
            "tumor_to_background_ratio": self.plot_tbr.figure,
            "background_cv": self.plot_cv.figure,
            "runtime": self.plot_runtime.figure,
        }

    def build_config(self, phantom: PhantomState) -> ExperimentConfig | None:
        methods = []
        if self._fbp.isChecked():
            methods.append("fbp")
        if self._mlem.isChecked():
            methods.append("mlem")
        counts = tuple(count for count, box in self._count_boxes.items() if box.isChecked())
        if not methods or not counts:
            return None
        filt = "hann" if self._filter.currentText().lower().startswith("hann") else "ramp"
        return ExperimentConfig(
            phantom=phantom,
            methods=tuple(methods),
            count_targets=counts,
            n_trials=int(self._trials.value()),
            seed=int(self._seed.value()),
            n_iterations=int(self._iterations.currentText()),
            filter_name=filt,
            detector_efficiency=float(self._efficiency.value()),
        ).clamp()

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._start.setEnabled(not busy)
        self._cancel.setEnabled(busy)
        self._export.setEnabled((not busy) and self._result is not None)

    def show_progress(self, done: int, total: int, message: str) -> None:
        pct = 0 if total <= 0 else int(100 * done / total)
        self._progress.setValue(min(pct, 100))
        self._status.setText(message)

    def show_result(self, result: ExperimentResult) -> None:
        self._result = result
        self.set_busy(False)
        self._progress.setValue(100 if not result.cancelled else self._progress.value())
        self._status.setText(result.message)
        self._fill_table(result)
        rows = result.aggregates
        self.plot_localization.show_series(
            rows, mean_attr="localization_error_mean", std_attr="localization_error_std"
        )
        self.plot_tbr.show_series(rows, mean_attr="tbr_mean", std_attr="tbr_std")
        self.plot_cv.show_series(
            rows, mean_attr="background_cv_mean", std_attr="background_cv_std"
        )
        self.plot_runtime.show_series(rows, mean_attr="runtime_s_mean", std_attr="runtime_s_std")

    def show_error(self, message: str) -> None:
        self.set_busy(False)
        self._status.setText(f"Experiment failed: {message}")

    def clear_results(self) -> None:
        self._result = None
        self._progress.setValue(0)
        self._table.setRowCount(0)
        for plot in self._plots():
            plot.show_placeholder("Results cleared.")
        self._status.setText("Experiment results cleared.")
        self.set_busy(False)

    def _fill_table(self, result: ExperimentResult) -> None:
        self._table.setRowCount(len(result.aggregates))
        for row_index, row in enumerate(result.aggregates):
            values = [
                f"{row.count_target:,}",
                row.method.upper(),
                f"{row.n_ok}/{row.n_trials}",
                _fmt_pm(row.localization_error_mean, row.localization_error_std),
                _fmt_pm(row.tbr_mean, row.tbr_std),
                _fmt_pm(row.contrast_mean, row.contrast_std),
                _fmt_pm(row.background_cv_mean, row.background_cv_std),
                _fmt_pm(row.runtime_s_mean, row.runtime_s_std),
            ]
            for column, text in enumerate(values):
                item = QTableWidgetItem(text)
                item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)
                self._table.setItem(row_index, column, item)
        self._table.resizeColumnsToContents()

    def _on_start(self) -> None:
        if self._busy:
            return
        self.start_requested.emit()

    def _on_reset(self) -> None:
        self.clear_results()
        self.reset_requested.emit()

    def _on_export(self) -> None:
        if self._result is None:
            return
        directory = QFileDialog.getExistingDirectory(self, "Export experiment results")
        if not directory:
            return
        written = export_experiment(self._result, directory, figures=self.figures())
        self._status.setText(
            "Exported " + ", ".join(path.name for path in written.values()) + f" to {directory}"
        )
