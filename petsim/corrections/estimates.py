"""Estimated random and scatter sinograms.

Neither function accepts event labels, annihilation coordinates, or tumor
geometry. Random correction sees the configured rate and the crystal ring.
Global scatter scaling sees a side-window count and the same Compton
model that generated the energies. It does not see which prompts were
scattered, and it does not build a spatial scatter map.

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
This is a simplified educational approximation of the scatter *count*,
not a spatial scatter correction. Witness pairs have one photon in
300–420 keV and the other in the acquisition photopeak. Their number
N_w is an energy measurement. A phantom-free transfer k, from the
configured scatter probability and the energy resolution, converts that
number into a photopeak scatter count:

    T = k N_w

After random subtraction, the nonnegative residual R is scaled:

    scatter_hat = min(T / sum(R), 1) * R

R is the shape because photopeak scatter is small-angle and is not
separable from unscattered prompts. The side-window sinogram is broader
and is not subtracted. T is not fit to image-quality scores. With no
witnesses, or with the side window swallowed by a low photopeak threshold,
the estimate is zero.
"""

from __future__ import annotations

import numpy as np

from petsim.corrections.energy import scatter_transfer_coefficient
from petsim.models.detector import DetectorRing
from petsim.reconstruction.sinogram import build_sinogram


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
    *,
    n_side_window: int,
    scatter_probability: float,
    photo_low_kev: float,
    photo_high_kev: float,
    energy_fwhm_percent: float = 12.0,
    scatter_angle_sigma_rad: float = 0.6,
) -> np.ndarray:
    """Multiply every nonnegative bin by one fraction.

    The fraction is the side-window transfer count divided by the sum of
    ``measured``, capped at 1. Bins keep the same ratios. The side-window
    pairs supply the count, not a spatial template: their detector pairs
    are broader than the small-angle scatter still inside the photopeak.
    ``measured`` should already have the random estimate removed.
    """
    data = np.asarray(measured, dtype=np.float64)
    if data.ndim != 2:
        raise ValueError("measured sinogram must be two-dimensional")
    empty = np.zeros_like(data)
    if int(n_side_window) <= 0 or float(scatter_probability) <= 0.0:
        return empty
    coefficient = scatter_transfer_coefficient(
        scatter_probability,
        scatter_angle_sigma_rad=scatter_angle_sigma_rad,
        energy_fwhm_percent=energy_fwhm_percent,
        photo_low_kev=photo_low_kev,
        photo_high_kev=photo_high_kev,
    )
    target = coefficient * float(n_side_window)
    nonnegative = np.maximum(data, 0.0)
    total = float(nonnegative.sum())
    if target <= 0.0 or total <= 0.0:
        return empty
    fraction = min(target / total, 1.0)
    return nonnegative * fraction


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
    n_side_window: int = 0,
    scatter_probability: float = 0.0,
    photo_low_kev: float = 435.0,
    photo_high_kev: float = 650.0,
    energy_fwhm_percent: float = 12.0,
    scatter_angle_sigma_rad: float = 0.6,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return corrected prompts, random estimate, and scatter estimate.

    Random subtraction, when enabled, happens first. Scatter then scales the
    nonnegative residual to the side-window transfer count and never exceeds
    that residual. Prompts passed to FBP or MLEM are clipped at zero.
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
        if np.asarray(s_edges).size != data.shape[1] + 1:
            raise ValueError("s_edges length must be n_s + 1")
        scatter_est = estimate_scatter_sinogram(
            residual,
            n_side_window=n_side_window,
            scatter_probability=scatter_probability,
            photo_low_kev=photo_low_kev,
            photo_high_kev=photo_high_kev,
            energy_fwhm_percent=energy_fwhm_percent,
            scatter_angle_sigma_rad=scatter_angle_sigma_rad,
        )
    else:
        scatter_est = np.zeros_like(data)
    corrected = np.maximum(data - random_est - scatter_est, 0.0)
    corrected = np.where(np.isfinite(corrected), corrected, 0.0)
    return corrected, random_est, scatter_est
