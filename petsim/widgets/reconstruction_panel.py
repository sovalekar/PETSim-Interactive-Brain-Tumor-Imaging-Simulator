"""Right-hand Stage 3 reconstruction panel: sinogram, FBP image, comparison."""

from __future__ import annotations

import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from PySide6.QtCore import Signal
from PySide6.QtWidgets import (
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QVBoxLayout,
    QWidget,
)

from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult
from petsim.theme import COLORS


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
            # colorbar + dual axes: tight_layout warns; use subplots_adjust instead.
            self._figure.subplots_adjust(left=0.08, right=0.88, wspace=0.35)
        else:
            self._draw_image(result, cmap)
            self._figure.tight_layout(pad=0.5)

        self.draw_idle()

    def _draw_image(self, result: ReconstructionResult, cmap: str) -> None:
        ax = self._figure.add_subplot(111)
        self._style_axis(ax)
        fov = result.config.fov
        # Display non-negative view; underlying result.image is unchanged.
        img = result.image_display
        im = ax.imshow(
            img,
            origin="lower",
            extent=(-fov, fov, -fov, fov),
            cmap=cmap,
            interpolation="nearest",
            aspect="equal",
        )
        cbar = self._figure.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
        cbar.ax.tick_params(colors=COLORS["text_muted"], labelsize=7)
        ax.set_title("FBP reconstruction", fontsize=10)
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
        vmax = float(max(recon.max(), 1e-12))

        if phantom is not None and phantom.size:
            # Resample / display phantom on the same FOV framing.
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
            interpolation="nearest",
            aspect="equal",
        )
        ax1.set_title("Acquired phantom", fontsize=9)
        ax2.set_title("FBP reconstruction", fontsize=9)
        for ax in (ax1, ax2):
            ax.set_xlabel("x", fontsize=8)
            ax.set_ylabel("y", fontsize=8)

        cbar = self._figure.colorbar(im2, ax=[ax1, ax2], fraction=0.03, pad=0.04)
        cbar.ax.tick_params(colors=COLORS["text_muted"], labelsize=6)
        # Keep im1 referenced so linters know the phantom color scale is intentional.
        _ = im1


class ReconstructionPanel(QFrame):
    """Stage 3 controls + Matplotlib visualization."""

    run_requested = Signal(object)  # ReconstructionConfig
    clear_requested = Signal()

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        self._result: ReconstructionResult | None = None
        self._phantom: np.ndarray | None = None
        self._n_coincidences = 0
        self._busy = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(8)

        title = QLabel("Reconstruction")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "Stage 3: build a measured sinogram from coincidences, then "
            "reconstruct with filtered backprojection (FBP). Educational only."
        )
        subtitle.setObjectName("HintLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        # --- Controls ---
        row = QHBoxLayout()
        row.addWidget(QLabel("Method"))
        self._method = QComboBox()
        self._method.addItems(["Filtered Backprojection"])
        row.addWidget(self._method, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Filter"))
        self._filter = QComboBox()
        self._filter.addItems(["Ramp", "Hann"])
        row.addWidget(self._filter, stretch=1)
        layout.addLayout(row)

        row = QHBoxLayout()
        row.addWidget(QLabel("Display"))
        self._display = QComboBox()
        self._display.addItems(
            ["Reconstructed Image", "Sinogram", "Comparison"]
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

        self.canvas = ReconstructionCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

        self._status = QLabel("Recorded coincidences: 0")
        self._status.setObjectName("HintLabel")
        self._status.setWordWrap(True)
        layout.addWidget(self._status)

        self._run_btn = QPushButton("Run Reconstruction")
        self._run_btn.setObjectName("PrimaryButton")
        self._run_btn.setEnabled(False)
        self._run_btn.clicked.connect(self._on_run)
        layout.addWidget(self._run_btn)

        self._clear_btn = QPushButton("Clear Output")
        self._clear_btn.setEnabled(False)
        self._clear_btn.clicked.connect(self._on_clear)
        layout.addWidget(self._clear_btn)

    def current_config(self) -> ReconstructionConfig:
        filt = "hann" if self._filter.currentText().lower().startswith("hann") else "ramp"
        return ReconstructionConfig(
            method="fbp",
            filter_name=filt,
            n_theta=180,
            n_s=128,
            image_size=128,
            fov=1.0,
            s_max=0.92,
        ).clamp()

    def update_coincidence_count(self, n: int) -> None:
        self._n_coincidences = int(n)
        self._run_btn.setEnabled(n > 0 and not self._busy)
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
        if busy:
            self._status.setText("Reconstructing… (UI remains responsive)")
            self._run_btn.setText("Running…")
        else:
            self._run_btn.setText("Run Reconstruction")

    def show_result(self, result: ReconstructionResult) -> None:
        self._result = result
        self._clear_btn.setEnabled(True)
        self.set_busy(False)
        neg = int(np.sum(result.image < 0))
        self._status.setText(
            f"{result.message}\n"
            f"Negatives in raw FBP (display-clipped): {neg:,} pixels"
        )
        self._refresh_canvas()

    def show_error(self, message: str) -> None:
        self.set_busy(False)
        self._status.setText(f"Reconstruction failed: {message}")
        self.canvas._show_placeholder(f"Error:\n{message}")

    def invalidate(self, reason: str = "Acquisition data changed.") -> None:
        """Clear results when the underlying coincidences are reset/replaced."""
        self._result = None
        self._clear_btn.setEnabled(False)
        self.set_busy(False)
        self.canvas._show_placeholder(f"{reason}\nRun Reconstruction when ready.")
        self._status.setText(
            f"Recorded coincidences: {self._n_coincidences:,}\n{reason}"
        )

    def clear_output(self) -> None:
        self.invalidate("Output cleared.")

    def _refresh_canvas(self, *_args) -> None:
        if self._result is None:
            return
        self.canvas.show_result(
            self._result,
            display_mode=self._display.currentText(),
            colormap=self._cmap.currentText(),
            phantom=self._phantom,
        )

    def _on_run(self) -> None:
        if self._n_coincidences <= 0 or self._busy:
            return
        self.set_busy(True)
        self.run_requested.emit(self.current_config())

    def _on_clear(self) -> None:
        self.clear_output()
        self.clear_requested.emit()
