"""Matplotlib canvas: 48-element detector ring + synthetic brain phantom."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Circle, FancyBboxPatch
from matplotlib.transforms import Affine2D
import numpy as np

from petsim.models.phantom import PhantomParameters, SyntheticBrainPhantom
from petsim.models.scanner import DetectorRingGeometry
from petsim.theme import COLORS


class DetectorRingCanvas(FigureCanvasQTAgg):
    """Dark-themed FOV view with detector crystals surrounding the phantom."""

    def __init__(
        self,
        parent=None,
        geometry: DetectorRingGeometry | None = None,
        phantom: SyntheticBrainPhantom | None = None,
    ) -> None:
        self._figure = Figure(figsize=(5.5, 5.5), dpi=100, facecolor=COLORS["bg_panel"])
        super().__init__(self._figure)
        self.setParent(parent)

        self.geometry = geometry or DetectorRingGeometry(n_detectors=48)
        self.phantom = phantom or SyntheticBrainPhantom(resolution=256)
        self._params = PhantomParameters()

        self._ax = self._figure.add_subplot(111)
        self._image = None
        self._setup_axes()
        self.redraw(self._params)

    def _setup_axes(self) -> None:
        ax = self._ax
        ax.set_facecolor(COLORS["bg"])
        ax.set_aspect("equal")
        ax.set_xlim(-1.15, 1.15)
        ax.set_ylim(-1.15, 1.15)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_color(COLORS["border"])
        ax.set_title(
            f"{self.geometry.n_detectors}-detector ring · synthetic phantom",
            color=COLORS["text_muted"],
            fontsize=10,
            pad=8,
        )
        self._figure.tight_layout(pad=0.6)

    def redraw(self, params: PhantomParameters) -> None:
        """Re-rasterize the phantom and redraw detectors for the given parameters."""
        self._params = params.clamped()
        activity = self.phantom.rasterize(self._params)

        ax = self._ax
        ax.clear()
        self._setup_axes()

        # Phantom activity map (visualization only — not a reconstructed PET image)
        extent = (-1.0, 1.0, -1.0, 1.0)
        self._image = ax.imshow(
            activity,
            extent=extent,
            origin="upper",
            cmap="inferno",
            vmin=0.0,
            vmax=1.0,
            interpolation="bilinear",
            zorder=1,
        )

        # Soft FOV guide
        ax.add_patch(
            Circle(
                (0.0, 0.0),
                1.0,
                fill=False,
                linestyle="--",
                linewidth=0.8,
                edgecolor=COLORS["border"],
                zorder=2,
            )
        )

        # Detector ring
        ring = Circle(
            (0.0, 0.0),
            self.geometry.radius,
            fill=False,
            linewidth=1.2,
            edgecolor=COLORS["accent_dim"],
            alpha=0.7,
            zorder=3,
        )
        ax.add_patch(ring)

        xs, ys = self.geometry.detector_positions()
        angles = self.geometry.detector_angles()
        crystal_w = 0.055
        crystal_h = 0.090

        for x, y, theta in zip(xs, ys, angles, strict=True):
            # Orient each crystal tangent to the ring
            patch = FancyBboxPatch(
                (-crystal_w / 2, -crystal_h / 2),
                crystal_w,
                crystal_h,
                boxstyle="round,pad=0.002,rounding_size=0.01",
                facecolor=COLORS["accent"],
                edgecolor=COLORS["text"],
                linewidth=0.4,
                alpha=0.92,
                zorder=4,
            )
            transform = (
                Affine2D().rotate(theta + np.pi / 2).translate(x, y) + ax.transData
            )
            patch.set_transform(transform)
            ax.add_patch(patch)

        # Tumor marker crosshair (helps when intensity blends into brain)
        if self._params.show_tumor:
            ax.plot(
                self._params.tumor_x,
                self._params.tumor_y,
                marker="+",
                markersize=10,
                markeredgewidth=1.2,
                color=COLORS["text"],
                alpha=0.55,
                zorder=5,
            )

        self.draw_idle()
