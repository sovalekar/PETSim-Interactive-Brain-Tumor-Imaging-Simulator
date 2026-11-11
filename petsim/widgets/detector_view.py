"""Center panel: circular detector ring around the synthetic brain phantom."""

from __future__ import annotations

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Ellipse
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.theme import COLORS


class DetectorCanvas(FigureCanvasQTAgg):
    """Embedded Matplotlib canvas for the FOV + 48-crystal ring."""

    def __init__(self, parent: QWidget | None = None) -> None:
        self._figure = Figure(figsize=(5.5, 5.5), dpi=100, facecolor=COLORS["panel"])
        super().__init__(self._figure)
        self.setParent(parent)

        self._ax = self._figure.add_subplot(111)
        self._phantom = BrainPhantom()
        self._ring = DetectorRing(n_detectors=48)
        self._colormap = "hot"
        self._show_detectors = True
        self._show_labels = False
        self._image_artist = None
        self._draw_scene()

    def update_view(
        self,
        state: PhantomState,
        *,
        colormap: str = "hot",
        show_detectors: bool = True,
        show_labels: bool = False,
    ) -> None:
        self._phantom.state = state
        self._colormap = colormap
        self._show_detectors = show_detectors
        self._show_labels = show_labels
        self._draw_scene()

    def _draw_scene(self) -> None:
        ax = self._ax
        ax.clear()
        ax.set_facecolor(COLORS["bg"])

        # Phantom activity map
        image = self._phantom.rasterize()
        extent = (-1.0, 1.0, -1.0, 1.0)
        self._image_artist = ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=self._colormap,
            vmin=0.0,
            vmax=max(1.0, float(image.max()) if image.size else 1.0),
            interpolation="bilinear",
            zorder=1,
        )

        # FOV circle
        fov = Circle(
            (0.0, 0.0),
            1.0,
            fill=False,
            edgecolor=COLORS["fov"],
            linewidth=1.0,
            linestyle="--",
            zorder=2,
        )
        ax.add_patch(fov)

        state = self._phantom.state.clamp()
        if state.show_outline:
            # Approximate elliptical brain outline for the display silhouette.
            outline = Ellipse(
                (0.0, 0.0),
                width=2.0 * state.brain_radius * 0.85,
                height=2.0 * state.brain_radius,
                fill=False,
                edgecolor=COLORS["brain_edge"],
                linewidth=1.2,
                alpha=0.85,
                zorder=3,
            )
            ax.add_patch(outline)

        if state.tumor_enabled:
            tumor = Circle(
                (state.tumor_x, state.tumor_y),
                state.tumor_radius,
                fill=False,
                edgecolor=COLORS["tumor_hot"],
                linewidth=1.4,
                linestyle="-",
                zorder=4,
            )
            ax.add_patch(tumor)

        # Detector ring
        ring = Circle(
            (0.0, 0.0),
            self._ring.radius,
            fill=False,
            edgecolor=COLORS["border"],
            linewidth=1.0,
            zorder=5,
        )
        ax.add_patch(ring)

        if self._show_detectors:
            positions = self._ring.crystal_positions()
            wedges = self._ring.crystal_wedges()
            for i, ((x, y), (t0, t1)) in enumerate(zip(positions, wedges)):
                arc = Arc(
                    (0.0, 0.0),
                    width=2.0 * self._ring.radius,
                    height=2.0 * self._ring.radius,
                    angle=0.0,
                    theta1=np_degrees(t0),
                    theta2=np_degrees(t1),
                    color=COLORS["detector"],
                    linewidth=3.5,
                    zorder=6,
                )
                ax.add_patch(arc)
                # Small radial tick at each crystal center
                ax.plot(
                    [x * 0.97, x * 1.03],
                    [y * 0.97, y * 1.03],
                    color=COLORS["detector_active"],
                    linewidth=1.0,
                    zorder=7,
                )
                if self._show_labels and i % 8 == 0:
                    lx, ly = x * 1.12, y * 1.12
                    ax.text(
                        lx,
                        ly,
                        str(i),
                        color=COLORS["text_muted"],
                        fontsize=8,
                        ha="center",
                        va="center",
                        zorder=8,
                    )

        ax.set_xlim(-1.25, 1.25)
        ax.set_ylim(-1.25, 1.25)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(False)

        ax.set_title(
            f"Detector ring · {self._ring.n_detectors} crystals",
            color=COLORS["text_muted"],
            fontsize=10,
            pad=8,
        )

        self._figure.tight_layout(pad=0.6)
        self.draw_idle()


def np_degrees(radians: float) -> float:
    """Local degrees helper to avoid a top-level numpy import name clash."""
    import math

    return math.degrees(radians)


class DetectorView(QFrame):
    """Panel wrapper around the detector Matplotlib canvas."""

    def __init__(self, parent: QWidget | None = None) -> None:
        super().__init__(parent)
        self.setObjectName("Panel")

        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(6)

        title = QLabel("Acquisition Geometry")
        title.setObjectName("PanelTitle")
        layout.addWidget(title)

        subtitle = QLabel(
            "48-element circular detector ring surrounding the synthetic brain phantom."
        )
        subtitle.setObjectName("HintLabel")
        subtitle.setWordWrap(True)
        layout.addWidget(subtitle)

        self.canvas = DetectorCanvas(self)
        layout.addWidget(self.canvas, stretch=1)

    def update_view(
        self,
        state: PhantomState,
        *,
        colormap: str = "hot",
        show_detectors: bool = True,
        show_labels: bool = False,
    ) -> None:
        self.canvas.update_view(
            state,
            colormap=colormap,
            show_detectors=show_detectors,
            show_labels=show_labels,
        )
