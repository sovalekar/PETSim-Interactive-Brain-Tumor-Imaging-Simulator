"""Estimated random and scatter sinograms.

Neither function accepts event labels, annihilation coordinates, or tumor
geometry. They see counts, scanner geometry, and the random rate the
acquisition was configured to use.

Random estimate
---------------
Stage 7 inserts one extra coincidence with probability ρ for each
annihilation, uniformly among distinct crystals, and accepts it without a
delayed timing window. The estimate is the expectation of that process:

    μ_R = ρ × N_annihilations

    r_hat = μ_R × (histogram of every unordered crystal pair) / (number of pairs)

Assumptions: the configured ρ is the insertion probability, pairs are
uniform, every random is accepted, and crystals are equally efficient. This
is not a delayed-coincidence measurement. If detector effects were off, the
caller must pass ρ = 0 because those randoms were never generated.

Scatter estimate
----------------
After any random subtraction, scatter is a wide Gaussian smooth of the
nonnegative residual, scaled so its tails match the residual tails.

Tails are radial bins with |s| ≥ 0.78 FOV units. That cut is scanner
geometry: the phantom brain radius is at most 0.75, and the elliptical
support stays inside this cut. The phantom radius is not an input.

Assumptions: scatter is smooth and additive, and after random removal the
tails are mostly scatter. A scatter-only correction on data that still
contain randoms will treat part of the flat random background as scatter.
This is not a single-scatter simulation and not a clinical scatter correction.
"""

from __future__ import annotations

import numpy as np

from petsim.models.detector import DetectorRing
from petsim.reconstruction.sinogram import build_sinogram, smooth_sinogram_data

# Outside the largest allowed brain support. Not read from the phantom.
TAIL_S_MIN = 0.78


def estimate_random_sinogram(
    n_annihilations: int,
    random_fraction: float,
    *,
    n_detectors: int = 48,
    ring_radius: float = 0.92,
    n_theta: int = 96,
    n_s: int = 128,
    s_max: float = 0.92,
) -> np.ndarray:
    """Expected random-count sinogram. Shape ``(n_theta, n_s)``."""
    mu = float(max(int(n_annihilations), 0)) * float(np.clip(random_fraction, 0.0, 1.0))
    n_theta = int(n_theta)
    n_s = int(n_s)
    empty = np.zeros((n_theta, n_s), dtype=np.float64)
    if mu <= 0.0 or int(n_detectors) < 2:
        return empty
    crystals = DetectorRing(n_detectors=int(n_detectors), radius=float(ring_radius)).crystal_positions()
    left, right = np.triu_indices(int(n_detectors), k=1)
    sinogram = build_sinogram(
        crystals[left],
        crystals[right],
        n_theta=n_theta,
        n_s=n_s,
        s_max=float(s_max),
    )
    total = float(sinogram.data.sum())
    if total <= 0.0:
        return empty
    return sinogram.data * (mu / total)


def estimate_scatter_sinogram(
    measured: np.ndarray,
    s_edges: np.ndarray,
    *,
    tail_s_min: float = TAIL_S_MIN,
    sigma_s_bins: float | None = None,
    sigma_theta_bins: float = 2.0,
) -> np.ndarray:
    """Smooth tail-scaled scatter estimate. Same shape as ``measured``."""
    data = np.asarray(measured, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("measured sinogram must be two-dimensional")
    nonnegative = np.maximum(data, 0.0)
    if float(nonnegative.sum()) <= 0.0:
        return np.zeros_like(data)
    n_s = data.shape[1]
    sigma_s = float(max(n_s / 12.0, 2.0) if sigma_s_bins is None else sigma_s_bins)
    smooth = smooth_sinogram_data(
        nonnegative,
        sigma_s=sigma_s,
        sigma_theta=float(max(sigma_theta_bins, 0.0)),
    )
    edges = np.asarray(s_edges, dtype=np.float64)
    centers = 0.5 * (edges[:-1] + edges[1:])
    if centers.size != n_s:
        raise ValueError("s_edges length must be n_s + 1")
    tail = np.abs(centers) >= float(tail_s_min)
    if not np.any(tail):
        return np.zeros_like(data)
    tail_measured = float(nonnegative[:, tail].sum())
    tail_smooth = float(smooth[:, tail].sum())
    if tail_measured <= 0.0 or tail_smooth <= 0.0:
        return np.zeros_like(data)
    estimate = smooth * (tail_measured / tail_smooth)
    return np.maximum(estimate, 0.0)


def apply_sinogram_corrections(
    measured: np.ndarray,
    *,
    correct_random: bool,
    correct_scatter: bool,
    n_annihilations: int,
    random_fraction: float,
    s_edges: np.ndarray,
    n_detectors: int = 48,
    ring_radius: float = 0.92,
    s_max: float = 0.92,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corrected prompts, random estimate, and scatter estimate.

    The corrected prompts are clipped at zero. Random subtraction, when
    enabled, happens before the scatter tail fit so the tails are not asked
    to explain the randoms.
    """
    data = np.asarray(measured, dtype=np.float64)
    if correct_random:
        random_est = estimate_random_sinogram(
            n_annihilations,
            random_fraction,
            n_detectors=n_detectors,
            ring_radius=ring_radius,
            n_theta=data.shape[0],
            n_s=data.shape[1],
            s_max=s_max,
        )
    else:
        random_est = np.zeros_like(data)
    residual = data - random_est if correct_random else data
    if correct_scatter:
        scatter_est = estimate_scatter_sinogram(residual, s_edges)
    else:
        scatter_est = np.zeros_like(data)
    corrected = np.maximum(data - random_est - scatter_est, 0.0)
    corrected = np.where(np.isfinite(corrected), corrected, 0.0)
    return corrected, random_est, scatter_est
