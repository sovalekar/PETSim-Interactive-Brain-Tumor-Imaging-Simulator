"""Right-hand reconstruction panel: FBP visualization + Stage 4 analysis."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle
from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFrame,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from petsim.analysis.count_comparison import CountComparisonResult
from petsim.analysis.pipeline import QuantitativeAnalysisResult
from petsim.analysis.roi import ROIMasks
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult
from petsim.theme import COLORS


def _fmt_xy(xy: tuple[float, float] | None) -> str:
    if xy is None:
        return "—"
    return f"({xy[0]:+.3f}, {xy[1]:+.3f})"


def _fmt_num(value: float | None, *, digits: int = 3, suffix: str = "") -> str:
    if value is None or not np.isfinite(value):
        return "—"
    return f"{value:.{digits}f}{suffix}"


def _analysis_value(analysis: QuantitativeAnalysisResult | None, name: str) -> str:
    if analysis is None:
        return "—"
    if name == "loc":
        if analysis.localization.reliable and analysis.localization.error is not None:
            return _fmt_num(analysis.localization.error, digits=4)
        return "—"
    if not analysis.contrast.valid:
        return "—"
    value = {
        "tbr": analysis.contrast.tbr,
        "contrast": analysis.contrast.contrast,
        "cv": analysis.contrast.background_cv,
    }[name]
    return _fmt_num(value, digits=3)


def _format_algorithm_table(
    fbp: ReconstructionResult | None,
    fbp_analysis: QuantitativeAnalysisResult | None,
    mlem: ReconstructionResult | None,
    mlem_analysis: QuantitativeAnalysisResult | None,
) -> str:
    """Text table. Metrics are educational, not a claim that MLEM is better."""
    fbp_iters = "—"
    mlem_iters = "—"
    fbp_time = "—"
    mlem_time = "—"
    if fbp is not None:
        fbp_time = f"{fbp.runtime_s:.2f} s"
    if mlem is not None:
        mlem_time = f"{mlem.runtime_s:.2f} s"
        if mlem.history is not None:
            mlem_iters = str(len(mlem.history.iterations))
    lines = [
        "Same ROI geometry and nonnegative reconstructed intensities.",
        "Ground truth is used only for localization error and ROI placement.",
        f"{'Metric':<32} {'FBP':>12} {'MLEM':>12}",
        f"{'Localization error (FOV)':<32} {_analysis_value(fbp_analysis, 'loc'):>12} {_analysis_value(mlem_analysis, 'loc'):>12}",
        f"{'Tumor-to-background ratio':<32} {_analysis_value(fbp_analysis, 'tbr'):>12} {_analysis_value(mlem_analysis, 'tbr'):>12}",
        f"{'Relative contrast':<32} {_analysis_value(fbp_analysis, 'contrast'):>12} {_analysis_value(mlem_analysis, 'contrast'):>12}",
        f"{'Background CV':<32} {_analysis_value(fbp_analysis, 'cv'):>12} {_analysis_value(mlem_analysis, 'cv'):>12}",
        f"{'Runtime':<32} {fbp_time:>12} {mlem_time:>12}",
        f"{'MLEM iterations':<32} {fbp_iters:>12} {mlem_iters:>12}",
        "A higher TBR or lower error on one run is not a general ranking.",
    ]
    return "\n".join(lines)


class ReconstructionCanvas(FigureCanvasQTAgg):
    """Embedded Matplotlib canvas for recon / sinogram / comparison views."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._figure = Figure(figsize=(4.2, 4.2), dpi=100, facecolor=COLORS["panel"])
        super().__init__(self._figure)
        self.setParent(parent)
        self._show_placeholder("Run acquisition, then click Run Reconstruction.")

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.title.set_color(COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text_muted"])
        ax.yaxis.label.set_color(COLORS["text_muted"])

    def _show_placeholder(self, message: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=COLORS["text_muted"],
            fontsize=10,
            wrap=True,
            transform=ax.transAxes,
        )
        self._figure.tight_layout(pad=0.6)
        self.draw_idle()

    def show_result(
        self,
        result: ReconstructionResult,
        *,
        display_mode: str,
        colormap: str,
        phantom: np.ndarray | None,
        rois: ROIMasks | None = None,
        show_rois: bool = False,
        true_xy: tuple[float, float] | None = None,
        estimated_xy: tuple[float, float] | None = None,
        tumor_radius: float | None = None,
    ) -> None:
        mode = display_mode.lower()
        if not result.has_data:
            self._show_placeholder(result.message or "No reconstruction data.")
            return

        self._figure.clear()
        cmap = colormap if colormap else "hot"

        if mode.startswith("sinogram"):
            self._draw_sinogram(result, cmap)
            self._figure.tight_layout(pad=0.5)
        elif mode.startswith("comparison"):
            self._draw_comparison(result, cmap, phantom)
            self._figure.subplots_adjust(left=0.08, right=0.88, wspace=0.35)
        else:
            self._draw_image(
                result,
                cmap,
                rois=rois,
                show_rois=show_rois,
                true_xy=true_xy,
                estimated_xy=estimated_xy,
                tumor_radius=tumor_radius,
            )
            self._figure.tight_layout(pad=0.5)

        self.draw_idle()

    def _draw_image(
        self,
        result: ReconstructionResult,
        cmap: str,
        *,
        rois: ROIMasks | None,
        show_rois: bool,
        true_xy: tuple[float, float] | None,
        estimated_xy: tuple[float, float] | None,
        tumor_radius: float | None,
    ) -> None:
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        fov = result.config.fov
        img = result.image_display
        vmax = result.display_vmax(99.5)
        im = ax.imshow(
            img,
            origin="lower",
            extent=(-fov, fov, -fov, fov),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        if show_rois and rois is not None:
            # Contour ROI masks in FOV coordinates (same axes as imshow).
            axis = np.linspace(-fov, fov, img.shape[0])
            if np.any(rois.tumor):
                ax.contour(
                    axis,
                    axis,
                    rois.tumor.astype(float),
                    levels=[0.5],
                    colors=[COLORS["tumor_hot"]],
                    linewidths=1.2,
                )
            if np.any(rois.background):
                ax.contour(
                    axis,
                    axis,
                    rois.background.astype(float),
                    levels=[0.5],
                    colors=[COLORS["detector"]],
                    linewidths=1.0,
                    linestyles="--",
                )
        if show_rois and true_xy is not None and tumor_radius is not None:
            ax.add_patch(
                Circle(
                    true_xy,
                    tumor_radius,
                    fill=False,
                    edgecolor=COLORS["tumor"],
                    linewidth=1.0,
                    linestyle=":",
                )
            )
        if true_xy is not None:
            ax.plot(
                true_xy[0],
                true_xy[1],
                "+",
                color=COLORS["tumor_hot"],
                markersize=10,
                markeredgewidth=1.5,
                label="True tumor",
            )
        if estimated_xy is not None:
            ax.plot(
                estimated_xy[0],
                estimated_xy[1],
                "x",
                color=COLORS["accent"],
                markersize=9,
                markeredgewidth=1.5,
                label="Estimated",
            )
        if true_xy is not None or estimated_xy is not None:
            leg = ax.legend(loc="upper right", fontsize=7, framealpha=0.35)
            for text in leg.get_texts():
                text.set_color(COLORS["text"])
        cbar = self._figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        if result.config.method == "mlem":
            done = 0 if result.history is None else len(result.history.iterations)
            ax.set_title(f"MLEM reconstruction ({done} iterations)", fontsize=10)
        else:
            ax.set_title(f"FBP reconstruction ({result.config.filter_name})", fontsize=10)
        ax.set_xlabel("x (FOV)")
        ax.set_ylabel("y (FOV)")

    def _draw_sinogram(self, result: ReconstructionResult, cmap: str) -> None:
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        sino = result.sinogram
        extent = (
            float(sino.s_edges[0]),
            float(sino.s_edges[-1]),
            float(np.degrees(sino.theta_edges[0])),
            float(np.degrees(sino.theta_edges[-1])),
        )
        im = ax.imshow(
            sino.data,
            origin="lower",
            extent=extent,
            cmap=cmap,
            aspect="auto",
            interpolation="nearest",
        )
        cbar = self._figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        cbar.set_label("counts", color=COLORS["text_muted"], fontsize=8)
        ax.set_title("Measured sinogram p(θ, s)", fontsize=10)
        ax.set_xlabel("s (radial)")
        ax.set_ylabel("θ (degrees)")

    def _draw_comparison(
        self,
        result: ReconstructionResult,
        cmap: str,
        phantom: np.ndarray | None,
    ) -> None:
        ax1 = self._figure.add_subplot(1, 2, 1)
        ax2 = self._figure.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            self._style_axis(ax)

        fov = result.config.fov
        recon = result.image_display
        vmax = result.display_vmax(99.5)

        if phantom is not None and phantom.size:
            ph = np.asarray(phantom, dtype=np.float64)
            ph_vmax = float(max(ph.max(), 1e-12))
            im1 = ax1.imshow(
                ph,
                origin="lower",
                extent=(-1.0, 1.0, -1.0, 1.0),
                cmap=cmap,
                vmin=0.0,
                vmax=ph_vmax,
                interpolation="bilinear",
                aspect="equal",
            )
            ax1.set_xlim(-fov, fov)
            ax1.set_ylim(-fov, fov)
        else:
            im1 = ax1.imshow(
                np.zeros_like(recon),
                origin="lower",
                extent=(-fov, fov, -fov, fov),
                cmap=cmap,
                aspect="equal",
            )
            ax1.text(
                0.5,
                0.5,
                "No acquired\nphantom snapshot",
                ha="center",
                va="center",
                color=COLORS["text_muted"],
                transform=ax1.transAxes,
                fontsize=8,
            )

        im2 = ax2.imshow(
            recon,
            origin="lower",
            extent=(-fov, fov, -fov, fov),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        ax1.set_title("Acquired phantom", fontsize=9)
        algo = "MLEM" if result.config.method == "mlem" else "FBP"
        ax2.set_title(f"{algo} reconstruction", fontsize=9)
        for ax in (ax1, ax2):
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)

        cbar = self._figure.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors=COLORS["text_muted"], labelsize=6)
        _ = im1

    def show_algorithm_pair(
        self,
        fbp: ReconstructionResult,
        mlem: ReconstructionResult,
        *,
        colormap: str,
        shared_scale: bool,
        fbp_rois: ROIMasks | None = None,
        mlem_rois: ROIMasks | None = None,
        show_rois: bool = False,
    ) -> None:
        """Draw FBP and MLEM side by side on this reconstruction figure only."""
        self._figure.clear()
        ax1 = self._figure.add_subplot(1, 2, 1)
        ax2 = self._figure.add_subplot(1, 2, 2)
        for ax in (ax1, ax2):
            self._style_axis(ax)
        cmap = colormap if colormap else "hot"
        fbp_vmax = fbp.display_vmax(99.5)
        mlem_vmax = mlem.display_vmax(99.5)
        if shared_scale:
            fbp_vmax = mlem_vmax = max(fbp_vmax, mlem_vmax)
        self._imshow_recon(ax1, fbp, cmap, fbp_vmax, "FBP")
        self._imshow_recon(ax2, mlem, cmap, mlem_vmax, "MLEM")
        if show_rois and fbp_rois is not None:
            self._contour_rois(ax1, fbp, fbp_rois)
        if show_rois and mlem_rois is not None:
            self._contour_rois(ax2, mlem, mlem_rois)
        self._figure.subplots_adjust(left=0.08, right=0.95, wspace=0.35)
        self.draw_idle()

    def _imshow_recon(self, ax, result: ReconstructionResult, cmap: str, vmax: float, title: str) -> None:
        fov = result.config.fov
        ax.imshow(
            result.image_display,
            origin="lower",
            extent=(-fov, fov, -fov, fov),
            cmap=cmap,
            vmin=0.0,
            vmax=vmax,
            interpolation="bilinear",
            aspect="equal",
        )
        iters = ""
        if result.config.method == "mlem":
            done = 0 if result.history is None else len(result.history.iterations)
            iters = f", {done} iter"
        filt = ""
        if result.config.method != "mlem":
            filt = f", {result.config.filter_name}"
        ax.set_title(f"{title}{filt}{iters}", fontsize=9)
        ax.set_xlabel("x (FOV)", fontsize=8)
        ax.set_ylabel("y (FOV)", fontsize=8)

    def _contour_rois(self, ax, result: ReconstructionResult, rois: ROIMasks) -> None:
        fov = result.config.fov
        n = result.image.shape[0]
        axis = np.linspace(-fov, fov, n)
        if np.any(rois.tumor):
            ax.contour(
                axis,
                axis,
                rois.tumor.astype(float),
                levels=[0.5],
                colors=[COLORS["tumor_hot"]],
                linewidths=1.0,
            )
        if np.any(rois.background):
            ax.contour(
                axis,
                axis,
                rois.background.astype(float),
                levels=[0.5],
                colors=[COLORS["detector"]],
                linewidths=0.9,
                linestyles="--",
            )


class CountComparisonCanvas(FigureCanvasQTAgg):
    """Dedicated canvas for background CV versus coincidence count.

    Independent of ``ReconstructionCanvas`` so count comparison never replaces
    the reconstructed PET image.
    """

    def __init__(self, parent: QWidget | None = None) -> None:
        self._figure = Figure(figsize=(4.2, 2.4), dpi=100, facecolor=COLORS["panel"])
        super().__init__(self._figure)
        self.setParent(parent)
        self.show_placeholder(
            "Run Count Comparison to plot background CV vs coincidence count."
        )

    def _style_axis(self, ax) -> None:
        ax.set_facecolor(COLORS["bg"])
        ax.tick_params(colors=COLORS["text_muted"], labelsize=8)
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.title.set_color(COLORS["text"])
        ax.xaxis.label.set_color(COLORS["text_muted"])
        ax.yaxis.label.set_color(COLORS["text_muted"])

    def show_placeholder(self, message: str) -> None:
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("Coincidence Count Comparison", fontsize=10)
        ax.text(
            0.5,
            0.5,
            message,
            ha="center",
            va="center",
            color=COLORS["text_muted"],
            fontsize=9,
            wrap=True,
            transform=ax.transAxes,
        )
        self._figure.tight_layout(pad=0.5)
        self.draw_idle()

    def show_comparison(self, comparison: CountComparisonResult) -> None:
        """Plot background CV versus detected coincidence count."""
        xs = [
            level.actual_count
            for level in comparison.levels
            if level.available and level.background_cv is not None
        ]
        ys = [
            level.background_cv
            for level in comparison.levels
            if level.available and level.background_cv is not None
        ]
        self._figure.clear()
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        ax.set_title(
            "Background coefficient of variation vs coincidence count",
            fontsize=10,
        )
        if not xs:
            ax.set_xticks([])
            ax.set_yticks([])
            ax.text(
                0.5,
                0.5,
                comparison.message or "No count-comparison CV points available.",
                ha="center",
                va="center",
                color=COLORS["text_muted"],
                fontsize=9,
                wrap=True,
                transform=ax.transAxes,
            )
        else:
            ax.plot(xs, ys, "o-", color=COLORS["detector"], markersize=6)
            ax.set_xlabel("Detected coincidences")
            ax.set_ylabel("Background CV (dimensionless)")
            ax.set_xscale("log")
            ax.grid(True, color=COLORS["grid"], alpha=0.6)
            ax.set_xticks(xs)
            ax.set_xticklabels([f"{int(x):,}" for x in xs])
        self._figure.tight_layout(pad=0.5)
        self.draw_idle()


class ConvergenceCanvas(FigureCanvasQTAgg):
    """MLEM Poisson log-likelihood and relative change. Own figure."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._figure = Figure(figsize=(4.2, 2.3), dpi=100, facecolor=COLORS["panel"])
        super().__init__(self._figure)
        self.setParent(parent)
        self.show_placeholder("Run MLEM to plot likelihood and relative change.")

    def _style_axis(self, ax) -> None:
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
        self._style_axis(ax)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_title("MLEM convergence", fontsize=10)
        ax.text(
            0.5, 0.5, message,
            ha="center", va="center", color=COLORS["text_muted"],
            fontsize=9, wrap=True, transform=ax.transAxes,
        )
        self._figure.tight_layout(pad=0.45)
        self.draw_idle()

    def show_history(self, history) -> None:
        self._figure.clear()
        if history is None or len(getattr(history, "iterations", [])) == 0:
            self.show_placeholder("No MLEM iterations recorded.")
            return
        iters = np.asarray(history.iterations)
        ll = np.asarray(history.log_likelihood, dtype=np.float64)
        rel = np.asarray(history.relative_change, dtype=np.float64)
        ax1 = self._figure.add_subplot(111)
        self._style_axis(ax1)
        finite = np.isfinite(ll)
        if np.any(finite):
            ax1.plot(iters[finite], ll[finite], "o-", color=COLORS["detector"], markersize=4)
        ax1.set_xlabel("Iteration")
        ax1.set_ylabel("Poisson log L (unnormalized)")
        ax1.set_title("MLEM convergence", fontsize=10)
        ax2 = ax1.twinx()
        ax2.plot(iters, rel, "s--", color=COLORS["warn"], markersize=3.5)
        ax2.set_ylabel("Relative image change", color=COLORS["warn"])
        ax2.tick_params(colors=COLORS["warn"], labelsize=7)
        ax2.spines["right"].set_color(COLORS["warn"])
        self._figure.tight_layout(pad=0.5)
        self.draw_idle()


class ReconstructionPanel(QFrame):
    """Stage 3 controls + Stage 4 quantitative analysis + Matplotlib view."""

    run_requested = Signal(object)  # ReconstructionConfig
    compare_requested = Signal(object, object)  # FBP config, MLEM config
    cancel_requested = Signal()
    clear_requested = Signal()
    analyze_requested = Signal()
    count_comparison_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._result: ReconstructionResult | None = None
        self._fbp_result: ReconstructionResult | None = None
        self._mlem_result: ReconstructionResult | None = None
        self._fbp_analysis: QuantitativeAnalysisResult | None = None
        self._mlem_analysis: QuantitativeAnalysisResult | None = None
        self._phantom: np.ndarray | None = None
        self._analysis: QuantitativeAnalysisResult | None = None
        self._comparison: CountComparisonResult | None = None
        self._live_iterations: list[int] = []
        self._live_likelihood: list[float] = []
        self._live_change: list[float] = []
        self._n_coincidences = 0
        self._busy = False
        self._analysis_busy = False

        root = QVBoxLayout(self)
        root.setContentsMargins(12, 12, 12, 12)
        root.setSpacing(8)

        title = QLabel("Reconstruction")
        title.setObjectName("PanelTitle")
        root.addWidget(title)

        subtitle = QLabel(
            "Stage 3–5: FBP or MLEM from measured coincidences, then quantitative "
            "comparison (educational metrics only)."
        )
        subtitle.setObjectName("HintLabel")
        subtitle.setWordWrap(True)
        root.addWidget(subtitle)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
        body = QWidget()
        layout = QVBoxLayout(body)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)
        scroll.setWidget(body)
        root.addWidget(scroll, stretch=1)

        # --- Controls ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self._method = QComboBox()
        self._method.addItems(
            ["Filtered Backprojection (FBP)", "Maximum Likelihood EM (MLEM)"]
        )
        self._method.currentTextChanged.connect(self._on_method_changed)
        row.addWidget(self._method, stretch=1)
        layout.addLayout(row)

        self._iter_row = QWidget()
        iter_layout = QHBoxLayout(self._iter_row)
        iter_layout.setContentsMargins(0, 0, 0, 0)
        iter_layout.addWidget(QLabel("Iterations"))
        self._iterations = QComboBox()
        self._iterations.addItems(["1", "5", "10", "20", "50"])
        self._iterations.setCurrentText("10")
        iter_layout.addWidget(self._iterations, stretch=1)
        self._iter_row.setVisible(False)
        layout.addWidget(self._iter_row)

        self._filter_row = QWidget()
        filter_layout = QHBoxLayout(self._filter_row)
        filter_layout.setContentsMargins(0, 0, 0, 0)
        filter_layout.addWidget(QLabel("Filter"))
        self._filter = QComboBox()
        self._filter.addItems(["Hann", "Ramp"])
        self._filter.setCurrentText("Hann")
        self._filter.currentTextChanged.connect(self._on_recon_param_changed)
        filter_layout.addWidget(self._filter, stretch=1)
        layout.addWidget(self._filter_row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Display"))
        self._display = QComboBox()
        self._display.addItems(
            ["Reconstructed Image", "Sinogram", "Comparison", "FBP vs MLEM"]
        )
        self._display.currentTextChanged.connect(self._refresh_canvas)
        row.addWidget(self._display, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Colormap"))
        self._cmap = QComboBox()
        self._cmap.addItems(["hot", "inferno", "magma", "viridis", "gray"])
        self._cmap.currentTextChanged.connect(self._refresh_canvas)
        row.addWidget(self._cmap, stretch=1)
        layout.addLayout(row)

        self._shared_scale = QCheckBox("Shared intensity scale for FBP vs MLEM")
        self._shared_scale.setChecked(False)
        self._shared_scale.setToolTip(
            "When checked, both algorithm images use the same display maximum. "
            "Uncheck to scale each image independently. Quantitative metrics "
            "do not use this display choice."
        )
        self._shared_scale.toggled.connect(self._refresh_canvas)
        layout.addWidget(self._shared_scale)

        self.canvas = ReconstructionCanvas(self)
        self.canvas.setMinimumHeight(280)
        self.canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        layout.addWidget(self.canvas)

        self._status = QLabel("Recorded coincidences: 0")
        self._status.setObjectName("HintLabel")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._run_btn = QPushButton("Run Reconstruction")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._compare_algs_btn = QPushButton("Compare FBP and MLEM")
        self._compare_algs_btn.setEnabled(False)
        self._compare_algs_btn.clicked.connect(self._on_compare_algorithms)
        layout.addWidget(self._compare_algs_btn)

        self._cancel_btn = QPushButton("Cancel Reconstruction")
        self._cancel_btn.setEnabled(False)
        self._cancel_btn.clicked.connect(self.cancel_requested.emit)
        layout.addWidget(self._cancel_btn)

        conv_box = QGroupBox("MLEM Convergence")
        conv_layout = QVBoxLayout(conv_box)
        conv_note = QLabel(
            "Poisson log-likelihood Σ (y log ŷ − ŷ), constant log(y!) omitted. "
            "Relative change is ‖xⁿ − xⁿ⁻¹‖ / ‖xⁿ⁻¹‖."
        )
        conv_note.setObjectName("HintLabel")
        conv_note.setWordWrap(True)
        conv_layout.addWidget(conv_note)
        self.convergence_canvas = ConvergenceCanvas(self)
        self.convergence_canvas.setMinimumHeight(180)
        conv_layout.addWidget(self.convergence_canvas)
        layout.addWidget(conv_box)

        self._clear_btn = QPushButton("Clear Output")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(self._clear_btn)

        # --- Stage 4 Quantitative Analysis ---
        analysis_box = QGroupBox("Quantitative Analysis")
        analysis_layout = QVBoxLayout(analysis_box)
        analysis_layout.setSpacing(6)

        hint = QLabel(
            "Educational reconstruction metrics (FOV units / dimensionless ratios). "
            "Not calibrated clinical activity concentrations."
        )
        hint.setObjectName("HintLabel")
        hint.setWordWrap(True)
        analysis_layout.addWidget(hint)

        self._analyze_btn = QPushButton("Run Analysis")
        self._analyze_btn.setEnabled(False)
        self._analyze_btn.clicked.connect(self._on_analyze)
        analysis_layout.addWidget(self._analyze_btn)

        self._overlay_rois = QCheckBox("Overlay tumor / background ROIs")
        self._overlay_rois.setChecked(True)
        self._overlay_rois.toggled.connect(self._refresh_canvas)
        analysis_layout.addWidget(self._overlay_rois)

        self._analysis_status = QLabel(
            "Run reconstruction, then Run Analysis for localization / contrast."
        )
        self._analysis_status.setObjectName("HintLabel")
        self._analysis_status.setWordWrap(True)
        analysis_layout.addWidget(self._analysis_status)

        loc_box = QGroupBox("Tumor Localization")
        loc_grid = QGridLayout(loc_box)
        self._loc_true = QLabel("—")
        self._loc_est = QLabel("—")
        self._loc_err = QLabel("—")
        self._loc_method = QLabel("—")
        self._loc_method.setWordWrap(True)
        for r, (label, widget) in enumerate(
            (
                ("True center (FOV)", self._loc_true),
                ("Estimated center (FOV)", self._loc_est),
                ("Localization error (FOV)", self._loc_err),
                ("Method", self._loc_method),
            )
        ):
            loc_grid.addWidget(QLabel(label), r, 0)
            loc_grid.addWidget(widget, r, 1)
        analysis_layout.addWidget(loc_box)

        contrast_box = QGroupBox("Image Contrast")
        contrast_grid = QGridLayout(contrast_box)
        self._c_tumor = QLabel("—")
        self._c_bg = QLabel("—")
        self._c_tbr = QLabel("—")
        self._c_contrast = QLabel("—")
        for r, (label, widget) in enumerate(
            (
                ("Tumor ROI mean", self._c_tumor),
                ("Background ROI mean", self._c_bg),
                ("Tumor-to-background ratio", self._c_tbr),
                ("Relative contrast", self._c_contrast),
            )
        ):
            contrast_grid.addWidget(QLabel(label), r, 0)
            contrast_grid.addWidget(widget, r, 1)
        analysis_layout.addWidget(contrast_box)

        noise_box = QGroupBox("Image Noise")
        noise_grid = QGridLayout(noise_box)
        self._n_cv = QLabel("—")
        noise_grid.addWidget(QLabel("Background coefficient of variation"), 0, 0)
        noise_grid.addWidget(self._n_cv, 0, 1)
        analysis_layout.addWidget(noise_box)

        algo_box = QGroupBox("FBP versus MLEM")
        algo_layout = QVBoxLayout(algo_box)
        self._algo_table = QLabel(
            "Run Compare FBP and MLEM to score both algorithms with the same ROIs. "
            "Localization uses the reconstructed image; tumor coordinates are ground "
            "truth for error and ROI placement only."
        )
        self._algo_table.setObjectName("HintLabel")
        self._algo_table.setWordWrap(True)
        self._algo_table.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        algo_layout.addWidget(self._algo_table)
        analysis_layout.addWidget(algo_box)

        layout.addWidget(analysis_box)

        count_box = QGroupBox("Coincidence Count Comparison")
        count_layout = QVBoxLayout(count_box)
        self._compare_btn = QPushButton("Count Comparison")
        self._compare_btn.setEnabled(False)
        self._compare_btn.clicked.connect(self._on_count_comparison)
        count_layout.addWidget(self._compare_btn)
        self._count_table = QLabel(
            "Target levels: 1,000 / 3,000 / 5,000 / 10,000 detected coincidences "
            "(nested prefixes of the current acquisition)."
        )
        self._count_table.setObjectName("HintLabel")
        self._count_table.setWordWrap(True)
        self._count_table.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        count_layout.addWidget(self._count_table)
        self.count_canvas = CountComparisonCanvas(self)
        self.count_canvas.setMinimumHeight(200)
        self.count_canvas.setSizePolicy(
            QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred
        )
        count_layout.addWidget(self.count_canvas)
        layout.addWidget(count_box)

    def _is_mlem(self) -> bool:
        return "mlem" in self._method.currentText().lower()

    def current_config(self) -> ReconstructionConfig:
        filt = "hann" if self._filter.currentText().lower().startswith("hann") else "ramp"
        return ReconstructionConfig(
            method="mlem" if self._is_mlem() else "fbp",
            filter_name=filt,
            n_theta=96,
            n_s=128,
            image_size=128,
            fov=1.0,
            s_max=0.92,
            smooth_sigma_s=1.5,
            smooth_sigma_theta=0.75,
            n_iterations=int(self._iterations.currentText()),
        ).clamp()

    def _on_method_changed(self, *_args) -> None:
        mlem = self._is_mlem()
        self._iter_row.setVisible(mlem)
        self._filter_row.setVisible(not mlem)

    def update_coincidence_count(self, n: int) -> None:
        self._n_coincidences = int(n)
        self._run_btn.setEnabled(n > 0 and not self._busy)
        self._compare_algs_btn.setEnabled(n > 0 and not self._busy)
        self._compare_btn.setEnabled(
            n > 0 and not self._busy and not self._analysis_busy
        )
        if self._result is None:
            self._status.setText(
                f"Recorded coincidences: {n:,}\n"
                "Acquire data, then run FBP reconstruction."
            )

    def set_acquired_phantom(self, activity: np.ndarray | None) -> None:
        self._phantom = None if activity is None else np.asarray(activity, dtype=np.float64)

    def set_busy(self, busy: bool) -> None:
        self._busy = busy
        self._run_btn.setEnabled((self._n_coincidences > 0) and not busy)
        self._compare_algs_btn.setEnabled((self._n_coincidences > 0) and not busy)
        self._cancel_btn.setEnabled(busy)
        self._update_analysis_buttons()
        if busy:
            self._status.setText("Reconstructing… (UI remains responsive)")
            self._run_btn.setText("Running…")
        else:
            self._run_btn.setText("Run Reconstruction")

    def set_analysis_busy(self, busy: bool) -> None:
        self._analysis_busy = busy
        self._update_analysis_buttons()
        if busy:
            self._analysis_status.setText("Analyzing… (UI remains responsive)")

    def _update_analysis_buttons(self) -> None:
        ready = self._result is not None and self._result.has_data
        idle = not self._busy and not self._analysis_busy
        self._analyze_btn.setEnabled(ready and idle)
        self._compare_btn.setEnabled(self._n_coincidences > 0 and idle)

    def show_result(self, result: ReconstructionResult) -> None:
        self._result = result
        if result.config.method == "mlem":
            self._mlem_result = result
            self.convergence_canvas.show_history(result.history)
        else:
            self._fbp_result = result
        # Image-derived metrics belong to the previous reconstruction.
        # Count-comparison and the other algorithm's canvas stay in place.
        self.clear_image_analysis(
            "New reconstruction — run analysis to refresh localization and contrast."
        )
        self.clear_algorithm_metrics()
        self._clear_btn.setEnabled(True)
        self.set_busy(False)
        neg = int(np.sum(result.image < 0))
        self._status.setText(
            f"{result.message}\n"
            f"Negatives in raw FBP (display-clipped): {neg:,} pixels"
        )
        self._update_analysis_buttons()
        self._refresh_canvas()

    def show_error(self, message: str) -> None:
        self.set_busy(False)
        self._status.setText(f"Reconstruction failed: {message}")
        self.canvas._show_placeholder(f"Error:\n{message}")

    def show_analysis(self, analysis: QuantitativeAnalysisResult) -> None:
        self._analysis = analysis
        self.set_analysis_busy(False)
        loc = analysis.localization
        contrast = analysis.contrast

        self._loc_true.setText(_fmt_xy(loc.true_xy))
        self._loc_est.setText(_fmt_xy(loc.estimated_xy) if loc.reliable else "—")
        if loc.reliable and loc.error is not None:
            self._loc_err.setText(_fmt_num(loc.error, digits=4))
        else:
            self._loc_err.setText("—")
        self._loc_method.setText(loc.method if loc.reliable else loc.message)

        if contrast.valid:
            self._c_tumor.setText(_fmt_num(contrast.tumor_mean, digits=4))
            self._c_bg.setText(_fmt_num(contrast.background_mean, digits=4))
            self._c_tbr.setText(_fmt_num(contrast.tbr, digits=3) + " (dimensionless)")
            self._c_contrast.setText(
                _fmt_num(contrast.contrast, digits=3) + " (dimensionless)"
            )
            self._n_cv.setText(
                _fmt_num(contrast.background_cv, digits=3) + " (dimensionless)"
            )
        else:
            self._c_tumor.setText("—")
            self._c_bg.setText("—")
            self._c_tbr.setText("—")
            self._c_contrast.setText("—")
            self._n_cv.setText("—")

        parts = [analysis.message]
        if not loc.reliable:
            parts.append(loc.message)
        if not contrast.valid:
            parts.append(contrast.message)
        parts.append(f"Intensity basis: {contrast.intensity_basis}")
        self._analysis_status.setText("\n".join(parts))
        self._refresh_canvas()

    def show_count_comparison(self, comparison: CountComparisonResult) -> None:
        self._comparison = comparison
        self.set_analysis_busy(False)
        lines = [comparison.message, ""]
        lines.append(
            f"{'Target':>8}  {'Actual':>8}  {'LocErr':>8}  {'TBR':>8}  {'BgCV':>8}  Note"
        )
        for level in comparison.levels:
            if level.available:
                lines.append(
                    f"{level.target_count:8d}  {level.actual_count:8d}  "
                    f"{_fmt_num(level.localization_error, digits=3):>8}  "
                    f"{_fmt_num(level.tbr, digits=3):>8}  "
                    f"{_fmt_num(level.background_cv, digits=3):>8}  OK"
                )
            else:
                lines.append(
                    f"{level.target_count:8d}  {level.actual_count:8d}  "
                    f"{'—':>8}  {'—':>8}  {'—':>8}  {level.message}"
                )
        self._count_table.setText("\n".join(lines))
        self.count_canvas.show_comparison(comparison)

    def show_analysis_error(self, message: str) -> None:
        self.set_analysis_busy(False)
        self._analysis_status.setText(f"Analysis failed: {message}")

    def clear_image_analysis(self, reason: str = "Analysis cleared.") -> None:
        """Clear localization and contrast for the current reconstruction.

        Does not touch the coincidence-count comparison canvas.
        """
        self._analysis = None
        self._loc_true.setText("—")
        self._loc_est.setText("—")
        self._loc_err.setText("—")
        self._loc_method.setText("—")
        self._c_tumor.setText("—")
        self._c_bg.setText("—")
        self._c_tbr.setText("—")
        self._c_contrast.setText("—")
        self._n_cv.setText("—")
        self._analysis_status.setText(reason)
        self._update_analysis_buttons()

    def clear_count_comparison(self, reason: str | None = None) -> None:
        """Clear the count-comparison table and its dedicated plot."""
        self._comparison = None
        message = (
            reason
            or "Target levels: 1,000 / 3,000 / 5,000 / 10,000 detected coincidences "
            "(nested prefixes of the current acquisition)."
        )
        self._count_table.setText(message)
        if hasattr(self, "count_canvas"):
            self.count_canvas.show_placeholder(
                "Run Count Comparison to plot background CV vs coincidence count."
            )

    def clear_algorithm_metrics(self) -> None:
        self._fbp_analysis = None
        self._mlem_analysis = None
        if hasattr(self, "_algo_table"):
            self._algo_table.setText(
                "Run Compare FBP and MLEM to score both algorithms with the same ROIs."
            )

    def clear_analysis(self, reason: str = "Analysis cleared.") -> None:
        self.clear_image_analysis(reason)
        self.clear_count_comparison(reason)
        self.clear_algorithm_metrics()

    def invalidate(self, reason: str = "Acquisition data changed.") -> None:
        """Clear results when the underlying coincidences are reset/replaced."""
        self._result = None
        self._fbp_result = None
        self._mlem_result = None
        self._clear_btn.setEnabled(False)
        self.set_busy(False)
        self.clear_analysis(reason)
        self.canvas._show_placeholder(f"{reason}\nRun Reconstruction when ready.")
        self.convergence_canvas.show_placeholder(
            "Convergence cleared. Run MLEM to plot a new likelihood curve."
        )
        self._status.setText(
            f"Recorded coincidences: {self._n_coincidences:,}\n{reason}"
        )

    def clear_output(self) -> None:
        self.invalidate("Output cleared.")

    def _on_recon_param_changed(self, *_args) -> None:
        if self._result is not None or self._analysis is not None:
            self.clear_image_analysis(
                "Filter changed — metrics may not match a new reconstruction. "
                "Re-run Reconstruction, then Run Analysis."
            )

    def begin_mlem_progress(self) -> None:
        self._live_iterations = []
        self._live_likelihood = []
        self._live_change = []

    def show_mlem_progress(
        self, iteration: int, total: int, loglik: float, rel: float
    ) -> None:
        """Update only the convergence canvas and the status line."""
        self._live_iterations.append(int(iteration))
        self._live_likelihood.append(float(loglik))
        self._live_change.append(float(rel))
        ll_text = f"{loglik:.4g}" if np.isfinite(loglik) else "undefined"
        self._status.setText(
            f"MLEM iteration {iteration}/{total} · log L = {ll_text} · "
            f"relative change {rel:.3g}"
        )
        from types import SimpleNamespace

        self.convergence_canvas.show_history(
            SimpleNamespace(
                iterations=np.asarray(self._live_iterations),
                log_likelihood=np.asarray(self._live_likelihood),
                relative_change=np.asarray(self._live_change),
            )
        )

    def show_algorithm_pair(
        self, fbp: ReconstructionResult, mlem: ReconstructionResult
    ) -> None:
        self._fbp_result = fbp
        self._mlem_result = mlem
        self._result = mlem if mlem.has_data else fbp
        self.convergence_canvas.show_history(mlem.history)
        self.clear_image_analysis(
            "FBP and MLEM finished. Review the comparison table, or run analysis "
            "on the active (MLEM) image."
        )
        self.clear_algorithm_metrics()
        self._clear_btn.setEnabled(True)
        self.set_busy(False)
        self._status.setText(f"{fbp.message}\n{mlem.message}")
        idx = self._display.findText("FBP vs MLEM")
        if idx >= 0:
            self._display.setCurrentIndex(idx)
        else:
            self._refresh_canvas()

    def show_algorithm_metrics(
        self,
        fbp_analysis: QuantitativeAnalysisResult | None,
        mlem_analysis: QuantitativeAnalysisResult | None,
    ) -> None:
        self._fbp_analysis = fbp_analysis
        self._mlem_analysis = mlem_analysis
        self._algo_table.setText(
            _format_algorithm_table(
                self._fbp_result, fbp_analysis, self._mlem_result, mlem_analysis
            )
        )
        if self._display.currentText().lower().startswith("fbp vs"):
            self._refresh_canvas()

    def _refresh_canvas(self, *_args) -> None:
        """Redraw only the reconstruction canvas (colormap, display mode, ROIs)."""
        mode = self._display.currentText()
        if mode.lower().startswith("fbp vs"):
            fbp = self._fbp_result
            mlem = self._mlem_result
            if (
                fbp is not None
                and mlem is not None
                and fbp.has_data
                and mlem.has_data
            ):
                self.canvas.show_algorithm_pair(
                    fbp,
                    mlem,
                    colormap=self._cmap.currentText(),
                    shared_scale=self._shared_scale.isChecked(),
                    fbp_rois=None if self._fbp_analysis is None else self._fbp_analysis.rois,
                    mlem_rois=None if self._mlem_analysis is None else self._mlem_analysis.rois,
                    show_rois=self._overlay_rois.isChecked(),
                )
            else:
                self.canvas._show_placeholder(
                    "Run Compare FBP and MLEM to show both reconstructions."
                )
            return
        if self._result is None:
            return

        rois = None
        true_xy = None
        est_xy = None
        tumor_radius = None
        if self._analysis is not None:
            rois = self._analysis.rois
            true_xy = self._analysis.localization.true_xy
            if self._analysis.localization.reliable:
                est_xy = self._analysis.localization.estimated_xy
            # Approximate tumor radius from ROI extent when available.
            if rois is not None and true_xy is not None and np.any(rois.tumor):
                # Infer radius from mask area: A = π r² → r = sqrt(A/π)
                area = float(np.count_nonzero(rois.tumor)) * (
                    (2.0 * self._result.config.fov / max(self._result.image.shape[0] - 1, 1))
                    ** 2
                )
                tumor_radius = float(np.sqrt(area / np.pi))

        self.canvas.show_result(
            self._result,
            display_mode=mode,
            colormap=self._cmap.currentText(),
            phantom=self._phantom,
            rois=rois,
            show_rois=self._overlay_rois.isChecked(),
            true_xy=true_xy,
            estimated_xy=est_xy,
            tumor_radius=tumor_radius,
        )

    def _on_run(self) -> None:
        if self._n_coincidences <= 0 or self._busy:
            return
        self.set_busy(True)
        if self._is_mlem():
            self.begin_mlem_progress()
        self.run_requested.emit(self.current_config())

    def _on_compare_algorithms(self) -> None:
        if self._n_coincidences <= 0 or self._busy:
            return
        self.set_busy(True)
        self.begin_mlem_progress()
        base = self.current_config()
        self.compare_requested.emit(base.as_method("fbp"), base.as_method("mlem"))

    def _on_clear(self) -> None:
        self.clear_output()
        self.clear_requested.emit()

    def _on_analyze(self) -> None:
        if self._result is None or self._analysis_busy:
            return
        self.set_analysis_busy(True)
        self.analyze_requested.emit()

    def _on_count_comparison(self) -> None:
        if self._n_coincidences <= 0 or self._analysis_busy:
            return
        self.set_analysis_busy(True)
        self.count_comparison_requested.emit()
