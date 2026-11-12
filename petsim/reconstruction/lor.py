"""Convert measured LOR endpoints into parallel-beam coordinates (θ, s).

Normal-form line equation
-------------------------
A line is written as

    x cos θ + y sin θ = s

where ``θ`` is the angle of the line's **normal** and ``s`` is the signed
perpendicular distance from the scanner origin to the line.

We fold representations into the conventional PET domain

    θ ∈ [0, π),   s ∈ ℝ

so that detector pair (i, j) and (j, i) map to the same bin.

True annihilation positions are **not** used here — only the two measured
ring-hit / detector-endpoint coordinates from Stage 2 coincidences.
"""

from __future__ import annotations

import numpy as np


def endpoints_to_sinogram_coords(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Map LOR endpoints to parallel-beam ``(theta, s)``.

    Parameters
    ----------
    hits_a, hits_b :
        Arrays of shape ``(M, 2)`` with measured ring intersection points
        for each coincidence.

    Returns
    -------
    theta : ndarray, shape ``(M,)``
        Projection angles in radians, each in ``[0, π)``.
    s : ndarray, shape ``(M,)``
        Signed radial offsets (same units as the endpoints).
    """
    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] == 0:
        return np.zeros(0, dtype=np.float64), np.zeros(0, dtype=np.float64)

    # Direction along the LOR (from A toward B).
    delta = b - a
    length = np.linalg.norm(delta, axis=1)
    # Degenerate pairs (same crystal) should not occur for true coincidences;
    # guard anyway so the pipeline never divides by zero.
    valid = length > 1e-12
    tangent = np.zeros_like(delta)
    tangent[valid] = delta[valid] / length[valid, None]

    # Unit normal obtained by rotating the tangent by +90°: (-ty, tx).
    normal = np.column_stack((-tangent[:, 1], tangent[:, 0]))

    # Signed distance from origin to the line (use midpoint for stability).
    mid = 0.5 * (a + b)
    s = np.sum(mid * normal, axis=1)

    # Angle of the normal in (-π, π].
    theta = np.arctan2(normal[:, 1], normal[:, 0])

    # Fold into θ ∈ [0, π), flipping the sign of s when needed.
    # If θ is negative, add π and negate s? Standard approach:
    #   bring θ into [0, 2π), then if θ ≥ π: θ -= π, s = -s
    theta = np.mod(theta, 2.0 * np.pi)
    flip = theta >= np.pi
    theta = np.where(flip, theta - np.pi, theta)
    s = np.where(flip, -s, s)

    # Invalid LORs → place at a harmless location (will rarely be binned).
    theta = np.where(valid, theta, 0.0)
    s = np.where(valid, s, 0.0)
    return theta.astype(np.float64), s.astype(np.float64)


def analytic_point_source_coords(
    x0: float,
    y0: float,
    thetas: np.ndarray,
) -> np.ndarray:
    """Analytic parallel-beam samples ``s(θ) = x0 cos θ + y0 sin θ``.

    Independent of detector-ring geometry — used for validation tests so
    reconstruction correctness is not checked only against the LOR converter.
    """
    thetas = np.asarray(thetas, dtype=np.float64)
    return x0 * np.cos(thetas) + y0 * np.sin(thetas)
