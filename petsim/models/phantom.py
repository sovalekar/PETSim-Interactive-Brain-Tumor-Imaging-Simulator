"""Synthetic brain + tumor phantom parameters and rasterization.

This module produces a simple activity-map style image for UI visualization
only. It does not model PET physics (attenuation, scatter, coincidence, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np


@dataclass(frozen=True, slots=True)
class PhantomParameters:
    """User-adjustable phantom knobs (normalized FOV coordinates in [-1, 1])."""

    brain_radius: float = 0.55
    brain_intensity: float = 0.35
    tumor_radius: float = 0.12
    tumor_x: float = 0.18
    tumor_y: float = 0.10
    tumor_intensity: float = 1.0
    show_tumor: bool = True

    def clamped(self) -> PhantomParameters:
        """Return a copy with values forced into safe UI ranges."""
        return replace(
            self,
            brain_radius=float(np.clip(self.brain_radius, 0.15, 0.85)),
            brain_intensity=float(np.clip(self.brain_intensity, 0.05, 1.0)),
            tumor_radius=float(np.clip(self.tumor_radius, 0.02, 0.40)),
            tumor_x=float(np.clip(self.tumor_x, -0.80, 0.80)),
            tumor_y=float(np.clip(self.tumor_y, -0.80, 0.80)),
            tumor_intensity=float(np.clip(self.tumor_intensity, 0.05, 1.0)),
            show_tumor=bool(self.show_tumor),
        )


class SyntheticBrainPhantom:
    """Rasterizes a circular brain + optional circular tumor into a 2D map."""

    def __init__(self, resolution: int = 256) -> None:
        if resolution < 32:
            raise ValueError("resolution must be >= 32")
        self.resolution = resolution
        yy, xx = np.mgrid[0:resolution, 0:resolution]
        # Map pixel indices to normalized FOV coordinates in [-1, 1]
        self._x = (xx / (resolution - 1)) * 2.0 - 1.0
        self._y = 1.0 - (yy / (resolution - 1)) * 2.0

    def rasterize(self, params: PhantomParameters) -> np.ndarray:
        """Return a float32 activity map in [0, 1] for display."""
        p = params.clamped()
        field = np.zeros((self.resolution, self.resolution), dtype=np.float32)

        brain_mask = (self._x**2 + self._y**2) <= (p.brain_radius**2)
        # Soft radial falloff inside the brain for a less flat look
        r = np.sqrt(self._x**2 + self._y**2, dtype=np.float32)
        brain_profile = np.clip(1.0 - 0.35 * (r / max(p.brain_radius, 1e-6)), 0.0, 1.0)
        field[brain_mask] = p.brain_intensity * brain_profile[brain_mask]

        if p.show_tumor:
            dx = self._x - p.tumor_x
            dy = self._y - p.tumor_y
            tumor_mask = (dx**2 + dy**2) <= (p.tumor_radius**2)
            # Tumor overwrites brain voxels; intensity is absolute display level
            field[tumor_mask] = p.tumor_intensity

        return np.clip(field, 0.0, 1.0)
