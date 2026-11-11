"""Synthetic 2-D brain phantom with an adjustable tumor region.

This module only describes geometry and a display raster for visualization.
It does not model PET physics, tracer kinetics, or attenuation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class PhantomState:
    """User-adjustable phantom parameters (display units in FOV radius = 1)."""

    brain_radius: float = 0.55
    brain_intensity: float = 0.35
    tumor_enabled: bool = True
    tumor_x: float = 0.18
    tumor_y: float = 0.12
    tumor_radius: float = 0.10
    tumor_intensity: float = 1.0
    show_outline: bool = True

    def clamp(self) -> PhantomState:
        """Return a copy with values constrained to sane ranges."""
        return PhantomState(
            brain_radius=float(np.clip(self.brain_radius, 0.20, 0.75)),
            brain_intensity=float(np.clip(self.brain_intensity, 0.05, 1.0)),
            tumor_enabled=bool(self.tumor_enabled),
            tumor_x=float(np.clip(self.tumor_x, -0.70, 0.70)),
            tumor_y=float(np.clip(self.tumor_y, -0.70, 0.70)),
            tumor_radius=float(np.clip(self.tumor_radius, 0.03, 0.30)),
            tumor_intensity=float(np.clip(self.tumor_intensity, 0.1, 2.0)),
            show_outline=bool(self.show_outline),
        )


@dataclass
class BrainPhantom:
    """Rasterizes a simple elliptical brain + circular tumor for display."""

    state: PhantomState = field(default_factory=PhantomState)
    grid_size: int = 256

    def rasterize(self) -> np.ndarray:
        """Return a (N, N) float32 activity map in [0, max_intensity]."""
        state = self.state.clamp()
        n = self.grid_size
        axis = np.linspace(-1.0, 1.0, n, dtype=np.float32)
        xx, yy = np.meshgrid(axis, axis)

        # Mildly elliptical "brain" silhouette (wider in x).
        brain_mask = (xx / 0.85) ** 2 + (yy / 1.0) ** 2 <= state.brain_radius**2
        image = np.zeros((n, n), dtype=np.float32)
        image[brain_mask] = state.brain_intensity

        # Soft radial falloff inside the brain for a less flat look.
        r_norm = np.sqrt((xx / 0.85) ** 2 + yy**2) / max(state.brain_radius, 1e-6)
        soft = np.clip(1.0 - 0.35 * r_norm, 0.55, 1.0).astype(np.float32)
        image[brain_mask] *= soft[brain_mask]

        if state.tumor_enabled:
            tumor_mask = (xx - state.tumor_x) ** 2 + (yy - state.tumor_y) ** 2 <= (
                state.tumor_radius**2
            )
            # Keep tumor inside the brain silhouette when possible.
            tumor_mask &= brain_mask
            image[tumor_mask] = state.tumor_intensity

        return image

    def tumor_center(self) -> tuple[float, float]:
        state = self.state.clamp()
        return state.tumor_x, state.tumor_y

    def brain_radius(self) -> float:
        return self.state.clamp().brain_radius
