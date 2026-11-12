"""Build a 2-D PET sinogram by binning measured LOR coordinates.

The sinogram ``p[i_theta, i_s]`` accumulates coincidence counts in projection
space. It is formed from Stage 2 detector measurements, **not** by taking a
Radon transform of the ground-truth phantom.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.reconstruction.lor import endpoints_to_sinogram_coords


@dataclass
class Sinogram:
    """Discrete parallel-beam sinogram and its axis metadata."""

    data: np.ndarray  # shape (n_theta, n_s), float64 counts
    theta_edges: np.ndarray  # length n_theta + 1, radians in [0, π]
    s_edges: np.ndarray  # length n_s + 1
    n_events: int = 0

    @property
    def n_theta(self) -> int:
        return int(self.data.shape[0])

    @property
    def n_s(self) -> int:
        return int(self.data.shape[1])

    @property
    def theta_centers(self) -> np.ndarray:
        return 0.5 * (self.theta_edges[:-1] + self.theta_edges[1:])

    @property
    def s_centers(self) -> np.ndarray:
        return 0.5 * (self.s_edges[:-1] + self.s_edges[1:])

    @property
    def s_max(self) -> float:
        return float(max(abs(self.s_edges[0]), abs(self.s_edges[-1])))


def _gauss_kernel(sigma: float) -> np.ndarray:
    """Normalized 1-D Gaussian kernel (σ in bins). σ ≤ 0 → identity."""
    if sigma <= 0.0:
        return np.array([1.0], dtype=np.float64)
    radius = int(np.ceil(3.0 * sigma))
    x = np.arange(-radius, radius + 1, dtype=np.float64)
    kernel = np.exp(-0.5 * (x / sigma) ** 2)
    kernel /= kernel.sum()
    return kernel


def smooth_sinogram_data(
    data: np.ndarray,
    *,
    sigma_s: float = 0.0,
    sigma_theta: float = 0.0,
) -> np.ndarray:
    """Mild separable Gaussian smoothing of a sinogram (counts → soft counts).

    Sparse PET sinograms (few thousand coincidences into many bins) are mostly
    empty. Ramp/Hann FBP then reconstructs each occupied bin as a bright LOR
    streak. A light pre-smooth is a standard low-count remedy: it does **not**
    invent activity from the phantom; it only spreads measured counts locally
    in ``(θ, s)`` before filtering.
    """
    out = np.asarray(data, dtype=np.float64).copy()
    if out.size == 0:
        return out
    total = float(out.sum())
    ks = _gauss_kernel(float(sigma_s))
    kt = _gauss_kernel(float(sigma_theta))
    if ks.size > 1:
        out = np.apply_along_axis(lambda row: np.convolve(row, ks, mode="same"), 1, out)
    if kt.size > 1:
        out = np.apply_along_axis(lambda col: np.convolve(col, kt, mode="same"), 0, out)
    # Edge effects of ``mode='same'`` can drop a little mass; restore total counts.
    new_total = float(out.sum())
    if total > 0.0 and new_total > 0.0:
        out *= total / new_total
    return out


def build_sinogram(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    *,
    n_theta: int = 96,
    n_s: int = 128,
    s_max: float = 0.92,
) -> Sinogram:
    """Bin coincidence endpoints into a ``(n_theta, n_s)`` sinogram.

    Parameters
    ----------
    hits_a, hits_b :
        Measured ring endpoints for each accepted coincidence, shape ``(M, 2)``.
    n_theta :
        Number of angular bins over ``[0, π)``. Default 96 is matched to the
        48-crystal ring (≈2 samples per crystal pitch) rather than an overly
        fine 180-bin grid that leaves most bins empty at educational counts.
    n_s :
        Number of radial bins over ``[-s_max, s_max]``.
    s_max :
        Radial half-width (typically the detector-ring radius).
    """
    n_theta = int(max(n_theta, 1))
    n_s = int(max(n_s, 1))
    s_max = float(abs(s_max))

    theta_edges = np.linspace(0.0, np.pi, n_theta + 1, dtype=np.float64)
    s_edges = np.linspace(-s_max, s_max, n_s + 1, dtype=np.float64)
    data = np.zeros((n_theta, n_s), dtype=np.float64)

    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] == 0:
        return Sinogram(data=data, theta_edges=theta_edges, s_edges=s_edges, n_events=0)

    theta, s = endpoints_to_sinogram_coords(a, b)

    # Digitize into bins. values exactly on the right edge go into the last bin.
    i_theta = np.digitize(theta, theta_edges) - 1
    i_s = np.digitize(s, s_edges) - 1
    i_theta = np.clip(i_theta, 0, n_theta - 1)
    # Drop samples outside the radial FOV rather than clipping (avoids edge bias).
    inside = (i_s >= 0) & (i_s < n_s)
    i_theta = i_theta[inside]
    i_s = i_s[inside]

    # Accumulate counts. np.add.at supports repeated indices.
    np.add.at(data, (i_theta, i_s), 1.0)

    return Sinogram(
        data=data,
        theta_edges=theta_edges,
        s_edges=s_edges,
        n_events=int(a.shape[0]),
    )


def analytic_disk_sinogram(
    *,
    radius: float = 0.25,
    n_theta: int = 180,
    n_s: int = 128,
    s_max: float = 1.0,
    amplitude: float = 1.0,
) -> Sinogram:
    """Analytic parallel-beam sinogram of a centered uniform disk.

    For a disk of radius ``R`` centered at the origin,

        p(s, θ) = 2 A √(R² − s²)   for |s| ≤ R,   else 0

    independent of θ. Used as a geometry-independent FBP validation fixture.
    """
    theta_edges = np.linspace(0.0, np.pi, n_theta + 1, dtype=np.float64)
    s_edges = np.linspace(-s_max, s_max, n_s + 1, dtype=np.float64)
    s_centers = 0.5 * (s_edges[:-1] + s_edges[1:])
    profile = np.zeros(n_s, dtype=np.float64)
    mask = np.abs(s_centers) <= radius
    profile[mask] = 2.0 * amplitude * np.sqrt(np.maximum(radius**2 - s_centers[mask] ** 2, 0.0))
    data = np.tile(profile, (n_theta, 1))
    return Sinogram(data=data, theta_edges=theta_edges, s_edges=s_edges, n_events=0)
