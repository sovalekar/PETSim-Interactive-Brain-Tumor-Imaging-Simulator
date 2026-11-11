"""Circular PET detector-ring geometry shared by display and Stage 2 physics.

Crystal angles / positions define where annihilations map to detector indices.
Photon transport and coincidence logic live in ``petsim.physics``.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class DetectorRing:
    """Fixed 48-crystal ring surrounding the FOV."""

    n_detectors: int = 48
    radius: float = 0.92

    def crystal_angles(self) -> np.ndarray:
        """Polar angles (radians) of crystal centers, starting at +x."""
        return np.linspace(0.0, 2.0 * np.pi, self.n_detectors, endpoint=False)

    def crystal_positions(self) -> np.ndarray:
        """Return (N, 2) crystal center coordinates in FOV units."""
        theta = self.crystal_angles()
        xy = np.column_stack(
            (self.radius * np.cos(theta), self.radius * np.sin(theta))
        )
        return xy.astype(np.float64)

    def crystal_wedges(
        self, angular_width: float | None = None
    ) -> list[tuple[float, float]]:
        """Return (theta0, theta1) pairs for drawing each crystal arc."""
        if angular_width is None:
            angular_width = (2.0 * np.pi / self.n_detectors) * 0.72
        centers = self.crystal_angles()
        half = angular_width / 2.0
        return [(float(c - half), float(c + half)) for c in centers]
