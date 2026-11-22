"""Time-of-flight geometry on the existing parallel-beam lines.

Stage 7 already stores an arrival-time difference when detector physics is on,
using a 15 cm ring. This module does not change that path. TOF reconstruction
uses the attenuation scale, 10 cm per field-of-view radius by default, so a
timing shift is a distance in the same coordinates as the image and the μ map.

Sign
----
``Δt = t_A - t_B``. A positive value means photon A traveled farther, so the
annihilation is closer to detector B. The most likely point is the chord
midpoint shifted toward B by ``c Δt / 2``.

The along-line coordinate ``t`` matches the attenuation chord,

    x = s cos θ - t sin θ
    y = s sin θ + t cos θ

after ``endpoints_to_sinogram_coords`` folds ``θ`` into ``[0, π)``. Folding
reverses ``t``, and the shift is measured in that folded frame.

Random coincidences have no annihilation path. This module does not assign
them a Gaussian TOF kernel.

The spatial kernel width is ``σ_x = c σ_t / 2``, with ``σ_t`` the existing
FWHM conversion. This is a 2-D Gaussian along the measured line. It is not a
clinical TOF system matrix.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.physics.detector_effects import SPEED_OF_LIGHT_CM_PER_PS, fwhm_to_sigma
from petsim.reconstruction.lor import endpoints_to_sinogram_coords

# Re-export so callers do not need a second definition of c.
SPEED_OF_LIGHT_CM_PER_PS = SPEED_OF_LIGHT_CM_PER_PS


@dataclass
class TOFConfig:
    """Coincidence timing used by acquisition and by TOF-MLEM.

    Disabled leaves the Stage 11 random stream and reconstructions unchanged.
    ``fwhm_ps`` is the FWHM of the arrival-time *difference*, matching Stage 7.
    """

    enabled: bool = False
    fwhm_ps: float = 400.0
    cm_per_fov: float = 10.0
    n_t: int = 21
    t_max: float = 1.45

    def clamp(self) -> TOFConfig:
        return TOFConfig(
            enabled=bool(self.enabled),
            fwhm_ps=float(np.clip(self.fwhm_ps, 0.0, 5000.0)),
            cm_per_fov=float(np.clip(self.cm_per_fov, 1.0, 40.0)),
            n_t=int(np.clip(self.n_t, 3, 81)),
            t_max=float(np.clip(self.t_max, 0.2, 2.0)),
        )

    @property
    def active(self) -> bool:
        """True when reconstruction should use a TOF kernel."""
        cfg = self.clamp()
        return bool(cfg.enabled and cfg.cm_per_fov > 0.0)


def spatial_sigma_fov(fwhm_ps: float, cm_per_fov: float) -> float:
    """Gaussian σ along the line of response, in field-of-view units.

    ``σ_x = c σ_t / 2``. ``σ_t`` is ``fwhm_to_sigma(FWHM)`` of the time difference.
    """
    sigma_ps = fwhm_to_sigma(float(fwhm_ps))
    if sigma_ps <= 0.0 or cm_per_fov <= 0.0:
        return 0.0
    return float(SPEED_OF_LIGHT_CM_PER_PS * sigma_ps / 2.0 / float(cm_per_fov))


def flight_time_difference_ps(
    path_a_fov: np.ndarray,
    path_b_fov: np.ndarray,
    cm_per_fov: float,
) -> np.ndarray:
    """``t_A - t_B`` in picoseconds. Longer path for A gives a positive value."""
    length_cm = (np.asarray(path_a_fov, dtype=np.float64) - np.asarray(path_b_fov, dtype=np.float64)) * float(
        cm_per_fov
    )
    return length_cm / SPEED_OF_LIGHT_CM_PER_PS


def signed_offset_fov(delta_t_ps: np.ndarray, cm_per_fov: float) -> np.ndarray:
    """``c Δt / 2`` in field-of-view units, positive toward detector B."""
    if cm_per_fov <= 0.0:
        return np.zeros_like(np.asarray(delta_t_ps, dtype=np.float64))
    return SPEED_OF_LIGHT_CM_PER_PS * np.asarray(delta_t_ps, dtype=np.float64) / 2.0 / float(cm_per_fov)


def time_bin_centers(n_t: int, t_max: float) -> np.ndarray:
    """Along-line sample locations covering ``[-t_max, t_max]``."""
    n = max(int(n_t), 1)
    return np.linspace(-float(t_max), float(t_max), n, dtype=np.float64)


def most_likely_points(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    delta_t_ps: np.ndarray,
    cm_per_fov: float,
) -> np.ndarray:
    """Point on each measured chord at ``c Δt / 2`` from the midpoint.

    The true annihilation coordinate is not an input. Random pairs should not
    be passed here: a flat time is not a TOF localization.
    """
    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    shift = signed_offset_fov(delta_t_ps, cm_per_fov).reshape(-1)
    delta = b - a
    length = np.linalg.norm(delta, axis=1)
    direction = np.zeros_like(delta)
    valid = length > 1e-12
    direction[valid] = delta[valid] / length[valid, None]
    mid = 0.5 * (a + b)
    return mid + direction * shift[:, None]


def along_line_coordinate(
    points: np.ndarray,
    hits_a: np.ndarray,
    hits_b: np.ndarray,
) -> np.ndarray:
    """Folded ``t`` of ``points`` on the lines through each endpoint pair."""
    theta, _s = endpoints_to_sinogram_coords(hits_a, hits_b)
    xy = np.asarray(points, dtype=np.float64).reshape(-1, 2)
    return -xy[:, 0] * np.sin(theta) + xy[:, 1] * np.cos(theta)


def discrete_tof_kernel(
    t_pixel: np.ndarray,
    t_centers: np.ndarray,
    sigma_fov: float,
) -> np.ndarray:
    """Detection probability of each time bin. Rows sum to at most 1.

    The kernel is ``φ(t) Δt`` with ``φ`` the normal density of width
    ``sigma_fov``. Bins outside the modeled time axis are not renormalized
    onto the edge, so a pixel whose mass falls outside the axis is less
    sensitive. A zero width puts the whole probability on the nearest bin.
    """
    pixels = np.asarray(t_pixel, dtype=np.float64).ravel()
    centers = np.asarray(t_centers, dtype=np.float64).ravel()
    n = int(pixels.size)
    n_t = int(centers.size)
    if n == 0 or n_t == 0:
        return np.zeros((n, n_t), dtype=np.float64)
    if sigma_fov <= 1e-8 or n_t == 1:
        nearest = np.argmin(np.abs(pixels[:, None] - centers[None, :]), axis=1)
        kernel = np.zeros((n, n_t), dtype=np.float64)
        kernel[np.arange(n), nearest] = 1.0
        return kernel
    step = float(centers[1] - centers[0]) if n_t > 1 else 1.0
    z = (centers[None, :] - pixels[:, None]) / float(sigma_fov)
    density = np.exp(-0.5 * z * z) / (float(sigma_fov) * np.sqrt(2.0 * np.pi))
    return density * abs(step)
