"""Photopeak scatter *count* from a lower energy side window.

Educational approximation. The count becomes one scale factor on the
prompt sinogram. It is not a map of where the scatter lies.

The Stage 7 energy window keeps only small Compton deflections in the
prompt list. Those events sit on top of the 511 keV peak and have almost
the same sinogram shape as unscattered lines, so a radial tail cannot see
them. A lower window can.

Side window
    Default 300–420 keV. An unscattered 511 keV photon, blurred by the
    usual 12% resolution, almost never lands here. A random record is
    stored at exactly 511 keV, so it does not land here either. A witness
    coincidence has one photon in this window and the other in the
    photopeak. It is a measured energy cut: no scatter label is read.

Transfer
    A phantom-free Monte Carlo of the same Compton-plus-resolution model
    gives

        k = P(both photons in the photopeak and at least one scattered)
            / P(one photon in the side window and the other in the photopeak)

    using the configured scatter probability, deflection width, and energy
    resolution. The estimated photopeak scatter count is k times the number
    of witness pairs. The Monte Carlo draws no patient activity and no
    recorded event labels.

What the count does not identify
    Small-angle scatter that remains inside the photopeak is included in
    k, but its sinogram is not separable from the unscattered prompts.
    The witness lines are a different, broader set of detector pairs.
    A mixture of the witness sinogram and the prompt sinogram does not
    fit the true photopeak-scatter sinogram better than the prompts
    alone, so the witness shape is not a scatter map. The estimator
    therefore multiplies every bin of the random-corrected prompts by
    the same fraction. Scale-free image scores can stay where they were.
    The image gets dimmer on a shared count scale because counts were
    removed.
"""

from __future__ import annotations

import functools

import numpy as np

from petsim.physics.detector_effects import fwhm_to_sigma
from petsim.physics.pet_engine import PHOTON_ENERGY_KEV

# Below the default 435 keV photopeak. The gap avoids the unscattered tail.
SIDE_WINDOW_KEV = (300.0, 420.0)
_SIDE_GAP_KEV = 5.0


def side_window_below_photopeak(
    photo_low_kev: float,
    photo_high_kev: float,
    *,
    side_low_kev: float = SIDE_WINDOW_KEV[0],
    side_high_kev: float = SIDE_WINDOW_KEV[1],
) -> tuple[float, float] | None:
    """Return the side window when it sits strictly below the photopeak."""
    low = float(side_low_kev)
    high = float(side_high_kev)
    photo_low = float(photo_low_kev)
    photo_high = float(photo_high_kev)
    if photo_high < photo_low or high <= low:
        return None
    if photo_low < high + _SIDE_GAP_KEV:
        return None
    return low, high


def count_side_window_coincidences(
    energy_a_kev,
    energy_b_kev,
    detected_a,
    detected_b,
    passed_timing,
    *,
    photo_low_kev: float,
    photo_high_kev: float,
    side_low_kev: float = SIDE_WINDOW_KEV[0],
    side_high_kev: float = SIDE_WINDOW_KEV[1],
    limit: int | None = None,
) -> int:
    """Count timing-accepted pairs with one side-window photon and one photopeak photon.

    ``limit`` keeps only the first records, so a trimmed prompt list and its
    witness pairs come from the same prefix. Labels are not an input.
    """
    window = side_window_below_photopeak(
        photo_low_kev,
        photo_high_kev,
        side_low_kev=side_low_kev,
        side_high_kev=side_high_kev,
    )
    if window is None:
        return 0
    side_low, side_high = window
    energy_a = np.asarray(energy_a_kev, dtype=np.float64)
    energy_b = np.asarray(energy_b_kev, dtype=np.float64)
    if limit is not None:
        stop = max(int(limit), 0)
        energy_a = energy_a[:stop]
        energy_b = energy_b[:stop]
    n = int(energy_a.shape[0])
    if n == 0 or energy_b.shape[0] < n:
        return 0
    energy_b = energy_b[:n]
    detected = np.asarray(detected_a, dtype=bool)[:n] & np.asarray(detected_b, dtype=bool)[:n]
    timing = np.asarray(passed_timing, dtype=bool)[:n]
    photo_a = (energy_a >= photo_low_kev) & (energy_a <= photo_high_kev)
    photo_b = (energy_b >= photo_low_kev) & (energy_b <= photo_high_kev)
    side_a = (energy_a >= side_low) & (energy_a <= side_high)
    side_b = (energy_b >= side_low) & (energy_b <= side_high)
    mixed = (photo_a & side_b) | (photo_b & side_a)
    return int(np.count_nonzero(detected & timing & mixed))


@functools.lru_cache(maxsize=32)
def _transfer_cached(
    scatter_milli: int,
    sigma_milli: int,
    fwhm_milli: int,
    photo_low_milli: int,
    photo_high_milli: int,
    side_low_milli: int,
    side_high_milli: int,
) -> float:
    """Phantom-free k. Arguments are milli-units so the cache key is exact."""
    probability = scatter_milli / 1000.0
    if probability <= 0.0:
        return 0.0
    sigma = sigma_milli / 1000.0
    fwhm = fwhm_milli / 1000.0
    photo = (photo_low_milli / 1000.0, photo_high_milli / 1000.0)
    side = (side_low_milli / 1000.0, side_high_milli / 1000.0)
    rng = np.random.default_rng(0)
    n = 120_000
    sigma_e = fwhm_to_sigma(PHOTON_ENERGY_KEV * (fwhm / 100.0))

    def photons() -> tuple[np.ndarray, np.ndarray]:
        scattered = rng.random(n) < probability
        phi = np.zeros(n, dtype=np.float64)
        n_scattered = int(scattered.sum())
        if n_scattered and sigma > 0.0:
            phi[scattered] = rng.normal(0.0, sigma, n_scattered)
        energy = PHOTON_ENERGY_KEV / (1.0 + (1.0 - np.cos(phi)))
        energy = np.where(scattered, energy, PHOTON_ENERGY_KEV)
        if sigma_e > 0.0:
            energy = energy + rng.normal(0.0, sigma_e, n)
        return np.clip(energy, 0.0, None), scattered

    energy_a, scattered_a = photons()
    energy_b, scattered_b = photons()

    def inside(energy: np.ndarray, bounds: tuple[float, float]) -> np.ndarray:
        return (energy >= bounds[0]) & (energy <= bounds[1])

    photopeak = inside(energy_a, photo) & inside(energy_b, photo)
    mixed = (inside(energy_a, photo) & inside(energy_b, side)) | (
        inside(energy_b, photo) & inside(energy_a, side)
    )
    scattered_pair = scattered_a | scattered_b
    numerator = float(np.count_nonzero(photopeak & scattered_pair))
    denominator = float(np.count_nonzero(mixed))
    if numerator <= 0.0 or denominator <= 0.0:
        return 0.0
    return numerator / denominator


def scatter_transfer_coefficient(
    scatter_probability: float,
    *,
    scatter_angle_sigma_rad: float,
    energy_fwhm_percent: float,
    photo_low_kev: float,
    photo_high_kev: float,
    side_low_kev: float = SIDE_WINDOW_KEV[0],
    side_high_kev: float = SIDE_WINDOW_KEV[1],
) -> float:
    """Return k for the configured detector, or 0 when the side window is unavailable."""
    window = side_window_below_photopeak(
        photo_low_kev,
        photo_high_kev,
        side_low_kev=side_low_kev,
        side_high_kev=side_high_kev,
    )
    if window is None or float(scatter_probability) <= 0.0:
        return 0.0
    side_low, side_high = window

    def milli(value: float) -> int:
        return int(round(float(value) * 1000.0))

    return _transfer_cached(
        milli(np.clip(scatter_probability, 0.0, 1.0)),
        milli(np.clip(scatter_angle_sigma_rad, 0.0, np.pi)),
        milli(np.clip(energy_fwhm_percent, 0.0, 40.0)),
        milli(photo_low_kev),
        milli(photo_high_kev),
        milli(side_low),
        milli(side_high),
    )
