"""Center panel: detector ring, phantom, and animated PET acquisition overlay."""

from __future__ import annotations

import math
from collections import deque

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure
from matplotlib.patches import Arc, Circle, Ellipse
from PySide6.QtWidgets import QFrame, QLabel, QVBoxLayout, QWidget

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.pet_engine import AnnihilationEvent
from petsim.theme import COLORS


def _degrees(radians: float) -> float:
    return math.degrees(radians)


class DetectorCanvas(FigureCanvasQTAgg):
    """Embedded Matplotlib canvas for the FOV + 48-crystal ring + events."""

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

        # Recent events kept for a short trail so motion is visible.
        self._recent_events: deque[AnnihilationEvent] = deque(maxlen=12)
        self._overlay_artists: list = []
        self._crystal_arcs: list[Arc] = []
        self._base_drawn = False

        self._draw_base_scene()

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
        self._recent_events.clear()
        self._draw_base_scene()

    def clear_events(self) -> None:
        self._recent_events.clear()
        self._clear_overlay()
        self._reset_crystal_highlights()
        self.draw_idle()

    def show_events(self, events: list[AnnihilationEvent]) -> None:
        """Append a batch of simulated events and refresh the overlay only."""
        for ev in events:
            self._recent_events.append(ev)
        self._redraw_overlay()

    def _draw_base_scene(self) -> None:
        ax = self._ax
        ax.clear()
        self._overlay_artists = []
        self._crystal_arcs = []
        ax.set_facecolor(COLORS["bg"])

        image = self._phantom.rasterize()
        extent = (-1.0, 1.0, -1.0, 1.0)
        ax.imshow(
            image,
            origin="lower",
            extent=extent,
            cmap=self._colormap,
            vmin=0.0,
            vmax=max(1.0, float(image.max()) if image.size else 1.0),
            interpolation="bilinear",
            zorder=1,
        )

        ax.add_patch(
            Circle(
                (0.0, 0.0),
                1.0,
                fill=False,
                edgecolor=COLORS["fov"],
                linewidth=1.0,
                linestyle="--",
                zorder=2,
            )
        )

        state = self._phantom.state.clamp()
        if state.show_outline:
            ax.add_patch(
                Ellipse(
                    (0.0, 0.0),
                    width=2.0 * state.brain_radius * 0.85,
                    height=2.0 * state.brain_radius,
                    fill=False,
                    edgecolor=COLORS["brain_edge"],
                    linewidth=1.2,
                    alpha=0.85,
                    zorder=3,
                )
            )

        if state.tumor_enabled:
            ax.add_patch(
                Circle(
                    (state.tumor_x, state.tumor_y),
                    state.tumor_radius,
                    fill=False,
                    edgecolor=COLORS["tumor_hot"],
                    linewidth=1.4,
                    zorder=4,
                )
            )

        ax.add_patch(
            Circle(
                (0.0, 0.0),
                self._ring.radius,
                fill=False,
                edgecolor=COLORS["border"],
                linewidth=1.0,
                zorder=5,
            )
        )

        if self._show_detectors:
            positions = self._ring.crystal_positions()
            wedges = self._ring.crystal_wedges()
            for i, ((x, y), (t0, t1)) in enumerate(zip(positions, wedges)):
                arc = Arc(
                    (0.0, 0.0),
                    width=2.0 * self._ring.radius,
                    height=2.0 * self._ring.radius,
                    angle=0.0,
                    theta1=_degrees(t0),
                    theta2=_degrees(t1),
                    color=COLORS["detector"],
                    linewidth=3.5,
                    zorder=6,
                )
                ax.add_patch(arc)
                self._crystal_arcs.append(arc)
                ax.plot(
                    [x * 0.97, x * 1.03],
                    [y * 0.97, y * 1.03],
                    color=COLORS["detector_active"],
                    linewidth=1.0,
                    zorder=7,
                )
                if self._show_labels and i % 8 == 0:
                    ax.text(
                        x * 1.12,
                        y * 1.12,
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
            f"Detector ring · {self._ring.n_detectors} crystals · Stage 2 acquisition",
            color=COLORS["text_muted"],
            fontsize=10,
            pad=8,
        )

        self._figure.tight_layout(pad=0.6)
        self._base_drawn = True
        self.draw_idle()

    def _clear_overlay(self) -> None:
        for artist in self._overlay_artists:
            try:
                artist.remove()
            except ValueError:
                pass
        self._overlay_artists = []

    def _reset_crystal_highlights(self) -> None:
        for arc in self._crystal_arcs:
            arc.set_color(COLORS["detector"])
            arc.set_linewidth(3.5)
            arc.set_alpha(1.0)

    def _redraw_overlay(self) -> None:
        """Draw photon paths / hits from recent events without clearing the base."""
        self._clear_overlay()
        self._reset_crystal_highlights()
        ax = self._ax

        # Highlight crystals hit by the newest batch (last few events).
        illuminate: dict[int, str] = {}
        for ev in self._recent_events:
            if ev.is_coincidence:
                illuminate[ev.detector_a] = COLORS["accent"]
                illuminate[ev.detector_b] = COLORS["accent"]
            else:
                # Partial / missed detections use a warmer warn color.
                if ev.detected_a:
                    illuminate.setdefault(ev.detector_a, COLORS["warn"])
                if ev.detected_b:
                    illuminate.setdefault(ev.detector_b, COLORS["warn"])

        for idx, color in illuminate.items():
            if 0 <= idx < len(self._crystal_arcs):
                arc = self._crystal_arcs[idx]
                arc.set_color(color)
                arc.set_linewidth(5.5)
                arc.set_alpha(1.0)

        # Draw paths: coincidence = accent, unsuccessful = muted danger.
        for i, ev in enumerate(self._recent_events):
            is_newest = i >= len(self._recent_events) - 3
            alpha = 0.95 if is_newest else 0.35
            if ev.is_coincidence:
                color = COLORS["accent"]
                lw = 1.6 if is_newest else 1.0
            else:
                color = COLORS["danger"]
                lw = 1.2 if is_newest else 0.8

            x0, y0 = ev.annihilation_xy
            xa, ya = ev.hit_a_xy
            xb, yb = ev.hit_b_xy

            (line_a,) = ax.plot(
                [x0, xa],
                [y0, ya],
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=20,
            )
            (line_b,) = ax.plot(
                [x0, xb],
                [y0, yb],
                color=color,
                linewidth=lw,
                alpha=alpha,
                solid_capstyle="round",
                zorder=20,
            )
            self._overlay_artists.extend([line_a, line_b])

            # Annihilation marker
            (ann,) = ax.plot(
                [x0],
                [y0],
                marker="o",
                markersize=4 if is_newest else 2.5,
                color=COLORS["text"],
                alpha=alpha,
                zorder=21,
            )
            self._overlay_artists.append(ann)

            # Hit markers only when that photon was detected
            if ev.detected_a:
                (ha,) = ax.plot(
                    [xa],
                    [ya],
                    marker="s",
                    markersize=4,
                    color=color,
                    alpha=alpha,
                    zorder=22,
                )
                self._overlay_artists.append(ha)
            if ev.detected_b:
                (hb,) = ax.plot(
                    [xb],
                    [yb],
                    marker="s",
                    markersize=4,
                    color=color,
                    alpha=alpha,
                    zorder=22,
                )
                self._overlay_artists.append(hb)

        self.draw_idle()


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
            "48-element circular detector ring. Green LORs = coincidences; "
            "red = failed single-photon detection. Paths follow simulated annihilations."
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

    def show_events(self, events: list[AnnihilationEvent]) -> None:
        self.canvas.show_events(events)

    def clear_events(self) -> None:
        self.canvas.clear_events()
