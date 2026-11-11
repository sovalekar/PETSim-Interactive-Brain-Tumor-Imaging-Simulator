"""2-D geometry helpers for the circular PET detector ring.

All coordinates use the same normalized FOV units as the Stage 1 phantom
(detector ring radius ≈ 0.92, FOV circle radius = 1.0).
"""

from __future__ import annotations

import numpy as np


def ray_circle_intersection(
    origins: np.ndarray,
    directions: np.ndarray,
    radius: float,
) -> np.ndarray:
    """Intersect rays with a circle centered at the origin.

    Each ray is ``P(t) = origin + t * direction`` with ``t > 0``.
    Directions must be unit vectors. Origins are assumed **inside** the
    circle (annihilations occur in the phantom, which sits inside the ring).

    Solving ``|origin + t d|^2 = R^2`` gives a quadratic in ``t``. Because the
    origin is inside the ring there is exactly one forward hit (``t > 0``).

    Parameters
    ----------
    origins :
        Shape ``(N, 2)`` annihilation positions.
    directions :
        Shape ``(N, 2)`` unit direction vectors.
    radius :
        Detector-ring radius.

    Returns
    -------
    hits : ndarray, shape ``(N, 2)``
        Intersection points on the circle.
    """
    origins = np.asarray(origins, dtype=np.float64)
    directions = np.asarray(directions, dtype=np.float64)

    # Quadratic coefficients for |o + t d|^2 = R^2 with |d| = 1:
    #   t^2 + 2 (o·d) t + (|o|^2 - R^2) = 0
    b = 2.0 * np.sum(origins * directions, axis=1)
    c = np.sum(origins * origins, axis=1) - radius**2
    discriminant = b * b - 4.0 * c
    # Numerical safety: clamp tiny negatives from floating-point error.
    discriminant = np.maximum(discriminant, 0.0)
    sqrt_disc = np.sqrt(discriminant)

    # Forward root: (-b + sqrt(disc)) / 2  (the + branch is the outward hit).
    t = (-b + sqrt_disc) / 2.0
    # If somehow t is non-positive (pathological float case), use the other root.
    t_alt = (-b - sqrt_disc) / 2.0
    t = np.where(t > 1e-12, t, t_alt)

    hits = origins + t[:, None] * directions
    return hits


def hit_to_detector_index(
    hits: np.ndarray,
    n_detectors: int,
) -> np.ndarray:
    """Map ring hit positions to crystal indices in ``[0, n_detectors)``.

    Crystal ``i`` is centered at polar angle ``2π i / N`` (same convention as
    :class:`~petsim.models.detector.DetectorRing`). We assign each hit to the
    nearest crystal center by angle.
    """
    hits = np.asarray(hits, dtype=np.float64)
    theta = np.arctan2(hits[:, 1], hits[:, 0])  # [-π, π]
    theta = np.mod(theta, 2.0 * np.pi)  # [0, 2π)
    # Nearest crystal: round(θ / Δθ) mod N, with Δθ = 2π / N.
    index = np.rint(theta * n_detectors / (2.0 * np.pi)).astype(np.int64)
    return np.mod(index, n_detectors)


def unit_directions_from_angles(angles: np.ndarray) -> np.ndarray:
    """Convert emission angles (radians) to unit direction vectors."""
    angles = np.asarray(angles, dtype=np.float64)
    return np.column_stack((np.cos(angles), np.sin(angles)))
