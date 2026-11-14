"""Optional detector effects for educational PET acquisition.

Effects run only when ``DetectorPhysicsConfig.enabled`` is true. Otherwise
``simulate_batch`` does not call this module and draws no extra random numbers.

Quantities
----------
Energy is in keV. Time is in picoseconds. The normalized ring radius (0.92 FOV
units) is scaled to 15 cm so flight times have physical units. The speed of
light used here is 0.0299792458 cm/ps.

Energy resolution
    True photon energy is 511 keV, or the Compton energy after a scatter.
    Measured energy is Gaussian about that value. The resolution control is
    the full width at half maximum as a percentage of 511 keV:

        FWHM_E = 511 keV * (percent / 100)
        sigma_E = FWHM_E / (2 * sqrt(2 * ln 2))

    A coincidence candidate passes only when both measured energies lie inside
    [E_low, E_high].

Timing
    For each photon, the flight time is the geometric path length divided by
    c. The path is the straight chord, or the broken path origin → scatter
    site → detector when that photon scatters.

        Δt_flight = t_A - t_B

    Positive Δt_flight means photon A has the longer path and arrives later.
    Coincidence timing resolution is the FWHM of the detector contribution to
    the arrival-time *difference* (not the FWHM of one crystal). One residual
    is drawn:

        Δt_residual ~ Normal(0, sigma_CTR^2)
        sigma_CTR = FWHM_CTR / (2 * sqrt(2 * ln 2))
        Δt_measured = Δt_flight + Δt_residual

    Geometric correction, available only because the simulation knows the true
    path lengths, is the subtraction that isolates the residual:

        Δt_residual = Δt_measured - Δt_flight

    A real coincidence processor does not know the annihilation point, so it
    cannot apply that correction. The acceptance window is therefore applied
    to the measured difference:

        accept if |Δt_measured| <= T_window

    It is not applied to Δt_flight and not applied to Δt_residual. Timing does
    not enter reconstruction (no time-of-flight update).

Random coincidences
    ``random_fraction`` is the probability that one annihilation also appends
    one extra accepted coincidence. That coincidence joins two distinct
    crystals chosen uniformly. Its endpoints are the crystal centers. It is
    not a back-to-back line from the annihilation.

Compton scatter
    With ``scatter_probability`` independently per photon, the photon scatters
    once. The site is 20–80% of the way along the original chord. The
    in-plane deflection is Gaussian with standard deviation
    ``scatter_angle_sigma_rad`` (default 0.6 rad). Energy follows

        E' = E / (1 + (E / 511 keV) * (1 - cos φ))

    This is a single-scatter 2-D cartoon: no attenuation, no Klein–Nishina
    sampling, and no out-of-plane scatter. It is not a patient-scatter
    validation.

Event classes are a partition. Energy failure is counted first, so a pair
that fails both windows is an energy rejection and is not also a timing
rejection. Random records are extra accepted coincidences; they are not
annihilations.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.physics.geometry import hit_to_detector_index, ray_circle_intersection
from petsim.physics.pet_engine import PHOTON_ENERGY_KEV, AnnihilationEvent

SPEED_OF_LIGHT_CM_PER_PS = 2.99792458e-2
RING_RADIUS_CM = 15.0
_FWHM_TO_SIGMA = 2.0 * np.sqrt(2.0 * np.log(2.0))


@dataclass
class DetectorPhysicsConfig:
    """User controls. Disabled by default."""

    enabled: bool = False
    energy_fwhm_percent: float = 12.0
    energy_window_low_kev: float = 435.0
    energy_window_high_kev: float = 650.0
    timing_fwhm_ps: float = 400.0
    timing_window_ps: float = 4000.0
    random_fraction: float = 0.0
    scatter_probability: float = 0.0
    scatter_angle_sigma_rad: float = 0.6

    def clamp(self) -> DetectorPhysicsConfig:
        low = float(np.clip(self.energy_window_low_kev, 50.0, 511.0))
        high = float(np.clip(self.energy_window_high_kev, 511.0, 800.0))
        if high < low:
            low, high = high, low
        return DetectorPhysicsConfig(
            enabled=bool(self.enabled),
            energy_fwhm_percent=float(np.clip(self.energy_fwhm_percent, 0.0, 40.0)),
            energy_window_low_kev=low,
            energy_window_high_kev=high,
            timing_fwhm_ps=float(np.clip(self.timing_fwhm_ps, 0.0, 5000.0)),
            timing_window_ps=float(np.clip(self.timing_window_ps, 1.0, 20000.0)),
            random_fraction=float(np.clip(self.random_fraction, 0.0, 1.0)),
            scatter_probability=float(np.clip(self.scatter_probability, 0.0, 1.0)),
            scatter_angle_sigma_rad=float(np.clip(self.scatter_angle_sigma_rad, 0.0, np.pi)),
        )


@dataclass
class EventAccounting:
    """One-batch or whole-acquisition event classes. The bins do not overlap."""

    annihilations: int = 0
    detected_singles: int = 0
    undetected: int = 0
    accepted_unscattered: int = 0
    accepted_scattered: int = 0
    accepted_random: int = 0
    energy_rejected: int = 0
    timing_rejected: int = 0

    @property
    def accepted_coincidences(self) -> int:
        return (
            self.accepted_unscattered
            + self.accepted_scattered
            + self.accepted_random
        )

    def annihilation_partition_holds(self) -> bool:
        """Every annihilation is in exactly one non-random bin."""
        return self.annihilations == (
            self.accepted_unscattered
            + self.accepted_scattered
            + self.energy_rejected
            + self.timing_rejected
            + self.detected_singles
            + self.undetected
        )


def fwhm_to_sigma(fwhm: float) -> float:
    """Gaussian standard deviation from a full width at half maximum."""
    if fwhm <= 0.0:
        return 0.0
    return float(fwhm) / _FWHM_TO_SIGMA


def compton_energy_kev(energy_kev: float, deflection_rad: float) -> float:
    """Photon energy after Compton scattering through ``deflection_rad``."""
    energy = float(energy_kev)
    factor = 1.0 + (energy / PHOTON_ENERGY_KEV) * (1.0 - float(np.cos(deflection_rad)))
    return energy / factor


def timing_accepted(measured_delta_t_ps: float, window_ps: float) -> bool:
    """Accept a pair from its measured arrival difference.

    ``window_ps`` is the maximum allowed ``|Δt_measured|``. The flight-time
    difference and the post-correction residual are not the tested quantities.
    """
    return abs(float(measured_delta_t_ps)) <= float(window_ps)


def fov_distance_to_cm(distance_fov: float, ring_radius_fov: float) -> float:
    scale = RING_RADIUS_CM / max(float(ring_radius_fov), 1e-6)
    return float(distance_fov) * scale


def _rotate(direction: np.ndarray, angle: float) -> np.ndarray:
    c, s = float(np.cos(angle)), float(np.sin(angle))
    x, y = float(direction[0]), float(direction[1])
    return np.array([c * x - s * y, s * x + c * y], dtype=np.float64)


def _distance(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.hypot(float(a[0] - b[0]), float(a[1] - b[1])))


def _forward_hit(origin: np.ndarray, direction: np.ndarray, radius: float) -> np.ndarray:
    unit = np.asarray(direction, dtype=np.float64)
    norm = float(np.linalg.norm(unit))
    unit = unit / max(norm, 1e-12)
    return ray_circle_intersection(origin[None, :], unit[None, :], radius)[0]


def account_events(events: list[AnnihilationEvent]) -> EventAccounting:
    """Classify events without putting any record in two bins.

    Energy is tested before timing. A candidate that fails the energy window
    is an energy rejection even if its measured time difference is also
    outside the timing window.
    """
    totals = EventAccounting()
    for event in events:
        if event.is_random:
            if event.is_coincidence:
                totals.accepted_random += 1
            continue
        totals.annihilations += 1
        detected = int(event.detected_a) + int(event.detected_b)
        if detected == 0:
            totals.undetected += 1
        elif detected == 1:
            totals.detected_singles += 1
        elif not event.passed_energy:
            totals.energy_rejected += 1
        elif not event.passed_timing:
            totals.timing_rejected += 1
        elif event.is_scattered:
            totals.accepted_scattered += 1
        else:
            totals.accepted_unscattered += 1
    return totals


def _scatter_photon(
    origin: np.ndarray,
    direction: np.ndarray,
    ring_radius: float,
    rng: np.random.Generator,
    config: DetectorPhysicsConfig,
) -> tuple[np.ndarray, float, bool, float]:
    """Return measured hit, path length in FOV units, scatter flag, true energy."""
    direction = np.asarray(direction, dtype=np.float64)
    hit = _forward_hit(origin, direction, ring_radius)
    path = _distance(origin, hit)
    if config.scatter_probability <= 0.0 or float(rng.random()) >= config.scatter_probability:
        return hit, path, False, PHOTON_ENERGY_KEV
    frac = float(rng.uniform(0.2, 0.8))
    site = origin + frac * (hit - origin)
    phi = 0.0
    if config.scatter_angle_sigma_rad > 0.0:
        phi = float(rng.normal(0.0, config.scatter_angle_sigma_rad))
    new_dir = _rotate(direction, phi)
    new_hit = _forward_hit(site, new_dir, ring_radius)
    path = _distance(origin, site) + _distance(site, new_hit)
    energy = compton_energy_kev(PHOTON_ENERGY_KEV, phi)
    return new_hit, path, True, energy


def _measure_energy(
    true_energy: float,
    rng: np.random.Generator,
    config: DetectorPhysicsConfig,
) -> float:
    fwhm = PHOTON_ENERGY_KEV * (config.energy_fwhm_percent / 100.0)
    sigma = fwhm_to_sigma(fwhm)
    if sigma <= 0.0:
        return float(true_energy)
    measured = float(true_energy + rng.normal(0.0, sigma))
    return max(measured, 0.0)


def apply_detector_physics(
    events: list[AnnihilationEvent],
    config: DetectorPhysicsConfig,
    rng: np.random.Generator,
    *,
    ring_radius: float,
    n_detectors: int,
) -> list[AnnihilationEvent]:
    """Return degraded copies. Disabled configs return the input list unchanged."""
    cfg = config.clamp()
    if not cfg.enabled:
        return events

    crystals = None
    if cfg.random_fraction > 0.0:
        from petsim.models.detector import DetectorRing

        crystals = DetectorRing(n_detectors=n_detectors, radius=ring_radius).crystal_positions()

    output: list[AnnihilationEvent] = []
    for event in events:
        origin = np.asarray(event.annihilation_xy, dtype=np.float64)
        direction_a = np.array([np.cos(event.angle), np.sin(event.angle)], dtype=np.float64)
        hit_a, path_a, scattered_a, energy_a = _scatter_photon(
            origin, direction_a, ring_radius, rng, cfg
        )
        hit_b, path_b, scattered_b, energy_b = _scatter_photon(
            origin, -direction_a, ring_radius, rng, cfg
        )
        measured_a = _measure_energy(energy_a, rng, cfg)
        measured_b = _measure_energy(energy_b, rng, cfg)
        energy_ok = (
            cfg.energy_window_low_kev <= measured_a <= cfg.energy_window_high_kev
            and cfg.energy_window_low_kev <= measured_b <= cfg.energy_window_high_kev
        )
        flight_ps = (
            fov_distance_to_cm(path_a, ring_radius) - fov_distance_to_cm(path_b, ring_radius)
        ) / SPEED_OF_LIGHT_CM_PER_PS
        residual_ps = 0.0
        sigma_t = fwhm_to_sigma(cfg.timing_fwhm_ps)
        if sigma_t > 0.0:
            residual_ps = float(rng.normal(0.0, sigma_t))
        measured_dt = float(flight_ps + residual_ps)
        timing_ok = timing_accepted(measured_dt, cfg.timing_window_ps)
        det_a = int(hit_to_detector_index(hit_a[None, :], n_detectors)[0])
        det_b = int(hit_to_detector_index(hit_b[None, :], n_detectors)[0])
        output.append(
            AnnihilationEvent(
                annihilation_xy=event.annihilation_xy,
                angle=event.angle,
                hit_a_xy=(float(hit_a[0]), float(hit_a[1])),
                hit_b_xy=(float(hit_b[0]), float(hit_b[1])),
                detector_a=det_a,
                detector_b=det_b,
                detected_a=event.detected_a,
                detected_b=event.detected_b,
                energy_kev=PHOTON_ENERGY_KEV,
                energy_a_kev=measured_a,
                energy_b_kev=measured_b,
                ideal_hit_a_xy=event.hit_a_xy,
                ideal_hit_b_xy=event.hit_b_xy,
                flight_time_difference_ps=float(flight_ps),
                timing_residual_ps=float(residual_ps),
                measured_delta_t_ps=measured_dt,
                passed_energy=energy_ok,
                passed_timing=timing_ok,
                is_scattered=scattered_a or scattered_b,
                is_random=False,
            )
        )
        if crystals is not None and float(rng.random()) < cfg.random_fraction:
            index_a, index_b = rng.choice(n_detectors, size=2, replace=False)
            hit_random_a = crystals[int(index_a)]
            hit_random_b = crystals[int(index_b)]
            output.append(
                AnnihilationEvent(
                    annihilation_xy=(0.0, 0.0),
                    angle=0.0,
                    hit_a_xy=(float(hit_random_a[0]), float(hit_random_a[1])),
                    hit_b_xy=(float(hit_random_b[0]), float(hit_random_b[1])),
                    detector_a=int(index_a),
                    detector_b=int(index_b),
                    detected_a=True,
                    detected_b=True,
                    energy_a_kev=PHOTON_ENERGY_KEV,
                    energy_b_kev=PHOTON_ENERGY_KEV,
                    passed_energy=True,
                    passed_timing=True,
                    is_scattered=False,
                    is_random=True,
                )
            )
    return output
