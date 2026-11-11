"""Scanner geometry placeholders for the UI (no coincidence / LoR physics)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True, slots=True)
class DetectorRingGeometry:
    """Circular ring of discrete detector crystals around the FOV origin."""

    n_detectors: int = 48
    radius: float = 0.92

    def __post_init__(self) -> None:
        if self.n_detectors < 3:
            raise ValueError("n_detectors must be >= 3")
        if self.radius <= 0:
            raise ValueError("radius must be positive")

    def detector_angles(self) -> np.ndarray:
        """Angles (radians) for each detector, starting at +x, CCW."""
        return np.linspace(0.0, 2.0 * np.pi, self.n_detectors, endpoint=False)

    def detector_positions(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (x, y) arrays of detector centers in FOV coordinates."""
        angles = self.detector_angles()
        x = self.radius * np.cos(angles)
        y = self.radius * np.sin(angles)
        return x, y
