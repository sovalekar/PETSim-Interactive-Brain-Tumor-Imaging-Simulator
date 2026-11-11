"""Simplified 2-D PET physics engine (Stage 2).

Physical model (educational, intentionally idealized)
----------------------------------------------------
Positron annihilation produces two 511 keV gamma photons:

    e⁺ + e⁻ → 2γ (511 keV each)

Simplifications for this stage (documented for learners):

* Ideal back-to-back emission (exactly opposite directions).
* No tissue attenuation (both photons always reach the ring geometrically).
* No Compton scattering / randoms / pile-up.
* No positron range (annihilation location = emission location).
* Independent Bernoulli detection with efficiency ε per photon.
  Coincidence probability is therefore ``P(coincidence) = ε²``.

These coincidences are stored for a future reconstruction stage. This module
does **not** form sinograms or reconstruct images.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom
from petsim.physics.geometry import (
    hit_to_detector_index,
    ray_circle_intersection,
    unit_directions_from_angles,
)

# Rest-mass energy of the electron/positron (keV). Stored for documentation /
# future energy-window features; Stage 2 does not use energy blurring.
PHOTON_ENERGY_KEV = 511.0


@dataclass
class AnnihilationEvent:
    """One simulated annihilation and its two photon outcomes."""

    annihilation_xy: tuple[float, float]
    angle: float  # emission angle of photon A (rad); photon B = angle + π
    hit_a_xy: tuple[float, float]
    hit_b_xy: tuple[float, float]
    detector_a: int
    detector_b: int
    detected_a: bool
    detected_b: bool
    energy_kev: float = PHOTON_ENERGY_KEV

    @property
    def is_coincidence(self) -> bool:
        """True only when *both* photons are detected (valid LOR)."""
        return self.detected_a and self.detected_b

    @property
    def detector_pair(self) -> tuple[int, int]:
        """Ordered (min, max) crystal indices for the line of response."""
        a, b = int(self.detector_a), int(self.detector_b)
        return (a, b) if a <= b else (b, a)


class PETEngine:
    """Sample annihilations from a phantom activity map and form coincidences."""

    def __init__(
        self,
        phantom: BrainPhantom | None = None,
        ring: DetectorRing | None = None,
        *,
        seed: int | None = None,
    ) -> None:
        self.phantom = phantom or BrainPhantom()
        self.ring = ring or DetectorRing(n_detectors=48)
        self.rng = np.random.default_rng(seed)
        self._seed = seed
        # Cached sampling tables rebuilt when the activity map changes.
        self._activity: np.ndarray | None = None
        self._pixel_centers: np.ndarray | None = None  # (Npix, 2)
        self._probabilities: np.ndarray | None = None
        self._pixel_half_width: float = 0.0

    def set_seed(self, seed: int | None) -> None:
        """Replace the RNG (used when the user sets an optional seed)."""
        self._seed = seed
        self.rng = np.random.default_rng(seed)

    def invalidate_activity_cache(self) -> None:
        """Call after the phantom geometry / uptake map changes."""
        self._activity = None
        self._pixel_centers = None
        self._probabilities = None

    def _ensure_sampling_tables(self) -> None:
        """Build a discrete PDF over phantom pixels proportional to uptake."""
        if self._probabilities is not None:
            return

        activity = self.phantom.rasterize().astype(np.float64)
        self._activity = activity
        n = activity.shape[0]
        # Pixel centers on the same [-1, 1] grid used by the display raster.
        axis = np.linspace(-1.0, 1.0, n, dtype=np.float64)
        self._pixel_half_width = 0.5 * (axis[1] - axis[0]) if n > 1 else 0.0
        xx, yy = np.meshgrid(axis, axis)
        self._pixel_centers = np.column_stack((xx.ravel(), yy.ravel()))

        flat = activity.ravel()
        total = flat.sum()
        if total <= 0.0:
            # Degenerate phantom: fall back to uniform over non-zero mask, or
            # a single center pixel so sampling never crashes.
            flat = np.ones_like(flat)
            total = flat.sum()
        self._probabilities = flat / total

    def sample_annihilation_positions(self, n: int) -> np.ndarray:
        """Draw ``n`` annihilation sites weighted by radiotracer uptake.

        Higher-uptake pixels (e.g. tumor) are chosen more often. A small
        uniform jitter inside each pixel avoids an artificial grid pattern.
        """
        if n <= 0:
            return np.zeros((0, 2), dtype=np.float64)

        self._ensure_sampling_tables()
        assert self._pixel_centers is not None and self._probabilities is not None

        indices = self.rng.choice(
            self._pixel_centers.shape[0],
            size=n,
            replace=True,
            p=self._probabilities,
        )
        positions = self._pixel_centers[indices].copy()
        if self._pixel_half_width > 0.0:
            jitter = self.rng.uniform(
                -self._pixel_half_width,
                self._pixel_half_width,
                size=positions.shape,
            )
            positions += jitter
        return positions

    def simulate_batch(
        self,
        n_events: int,
        *,
        detector_efficiency: float = 0.8,
    ) -> list[AnnihilationEvent]:
        """Simulate ``n_events`` annihilations and return per-event records.

        Parameters
        ----------
        n_events :
            Number of positron annihilations to generate.
        detector_efficiency :
            Independent single-photon detection probability ε in ``[0, 1]``.
            Expected coincidence fraction ≈ ε².
        """
        eps = float(np.clip(detector_efficiency, 0.0, 1.0))
        if n_events <= 0:
            return []

        origins = self.sample_annihilation_positions(n_events)

        # Emission angle of photon A; photon B travels exactly opposite (ideal).
        # Sampling [0, π) is enough to cover unique LORs; we use [0, 2π) so
        # animated arrows look uniformly random around the ring.
        angles = self.rng.uniform(0.0, 2.0 * np.pi, size=n_events)
        dir_a = unit_directions_from_angles(angles)
        dir_b = -dir_a

        hit_a = ray_circle_intersection(origins, dir_a, self.ring.radius)
        hit_b = ray_circle_intersection(origins, dir_b, self.ring.radius)

        det_a = hit_to_detector_index(hit_a, self.ring.n_detectors)
        det_b = hit_to_detector_index(hit_b, self.ring.n_detectors)

        # Independent Bernoulli trials for each photon.
        detected_a = self.rng.random(n_events) < eps
        detected_b = self.rng.random(n_events) < eps

        events: list[AnnihilationEvent] = []
        for i in range(n_events):
            events.append(
                AnnihilationEvent(
                    annihilation_xy=(float(origins[i, 0]), float(origins[i, 1])),
                    angle=float(angles[i]),
                    hit_a_xy=(float(hit_a[i, 0]), float(hit_a[i, 1])),
                    hit_b_xy=(float(hit_b[i, 0]), float(hit_b[i, 1])),
                    detector_a=int(det_a[i]),
                    detector_b=int(det_b[i]),
                    detected_a=bool(detected_a[i]),
                    detected_b=bool(detected_b[i]),
                    energy_kev=PHOTON_ENERGY_KEV,
                )
            )
        return events

    def expected_coincidence_fraction(self, detector_efficiency: float) -> float:
        """Return the analytic P(coincidence) = ε² for comparison with data."""
        eps = float(np.clip(detector_efficiency, 0.0, 1.0))
        return eps * eps
