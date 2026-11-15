"""511 keV attenuation for the educational brain phantom.

Coordinates stay in field-of-view units. ``cm_per_fov`` is the number of
centimeters represented by a distance of 1 (the unit-circle radius). The
default, 10 cm, makes a brain radius of 0.55 about 11 cm across.

The attenuation map is not the activity image. μ is constant inside the same
elliptical brain silhouette the phantom draws, and zero outside it. The tumor
does not change μ. That silhouette is the stand-in for a transmission map.
Reconstruction never reads tumor coordinates, event labels, or uptake.

Photon survival
---------------
A 511 keV photon traveling a tissue distance ``L`` (cm) survives with
probability ``exp(-μ L)``. The two photons are tested separately, so one can
be absorbed while the other is detected. For a straight ray the sum of the
two tissue paths equals the chord of that line of response through tissue.
Lost photons are removed from the coincidence list. They are not put back later.

Sinogram weight
---------------
Each parallel-beam bin ``(θ, s)`` uses the chord through its bin center,

    x cos θ + y sin θ = s

which is the same line the projector and the LOR converter use. The survival
weight is ``a = clip(exp(-μ L), a_min, 1)``. The floor ``a_min`` stops a
nearly stopped line from being amplified without bound. This is a simplified
parallel-beam compensation, not a clinical PET attenuation correction
(no transmission scan, no 3-D scatter, no crystal-pair system matrix).

Scaling measured counts by ``1/a`` changes their expected value. It does not
restore the absorbed photons, and it increases the variance of the bins that
remain. Empty bins stay empty.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Narrow-beam linear attenuation of soft tissue / water near 511 keV.
SOFT_TISSUE_MU_PER_CM = 0.096
DEFAULT_CM_PER_FOV = 10.0
DEFAULT_A_MIN = 1e-3


@dataclass
class AttenuationConfig:
    """Homogeneous 511 keV tissue model. Disabled leaves acquisition unchanged."""

    enabled: bool = False
    mu_per_cm: float = SOFT_TISSUE_MU_PER_CM
    cm_per_fov: float = DEFAULT_CM_PER_FOV
    brain_radius: float = 0.55
    a_min: float = DEFAULT_A_MIN

    def clamp(self) -> AttenuationConfig:
        return AttenuationConfig(
            enabled=bool(self.enabled),
            mu_per_cm=float(np.clip(self.mu_per_cm, 0.0, 1.0)),
            cm_per_fov=float(np.clip(self.cm_per_fov, 1.0, 40.0)),
            brain_radius=float(np.clip(self.brain_radius, 0.20, 0.75)),
            a_min=float(np.clip(self.a_min, 1e-6, 0.5)),
        )

    @property
    def active(self) -> bool:
        """True only when survival can differ from 1. μ = 0 draws no random numbers."""
        cfg = self.clamp()
        return bool(cfg.enabled and cfg.mu_per_cm > 0.0 and cfg.cm_per_fov > 0.0)


def ellipse_axes(brain_radius: float) -> tuple[float, float]:
    """Semi-axes ``(a, b)`` of the phantom brain silhouette, in FOV units.

    Matches ``BrainPhantom.rasterize``: ``(x / 0.85)² + y² <= brain_radius²``,
    so the ellipse is taller than it is wide.
    """
    radius = float(max(brain_radius, 1e-6))
    return 0.85 * radius, radius


def _quadratic_roots(a_coef: np.ndarray, b_coef: np.ndarray, c_coef: np.ndarray):
    """Real roots of ``A t² + B t + C = 0``. Invalid rows are marked False."""
    disc = b_coef * b_coef - 4.0 * a_coef * c_coef
    valid = (a_coef > 1e-15) & (disc >= 0.0)
    sqrt_disc = np.sqrt(np.maximum(disc, 0.0))
    denom = np.where(valid, 2.0 * a_coef, 1.0)
    t_lo = (-b_coef - sqrt_disc) / denom
    t_hi = (-b_coef + sqrt_disc) / denom
    return t_lo, t_hi, valid


def ellipse_chord_length(
    s: np.ndarray,
    theta: np.ndarray,
    semi_x: float,
    semi_y: float,
) -> np.ndarray:
    """Tissue chord of the parallel-beam line ``x cos θ + y sin θ = s``.

    Length is in field-of-view units. Lines that miss the ellipse return 0.
    """
    s_arr = np.asarray(s, dtype=np.float64)
    th = np.asarray(theta, dtype=np.float64)
    shape = np.broadcast(s_arr, th).shape
    s_arr = np.broadcast_to(s_arr, shape).astype(np.float64, copy=False)
    th = np.broadcast_to(th, shape).astype(np.float64, copy=False)
    a_axis = float(semi_x)
    b_axis = float(semi_y)
    if a_axis <= 0.0 or b_axis <= 0.0:
        return np.zeros(shape, dtype=np.float64)

    cos_t = np.cos(th)
    sin_t = np.sin(th)
    # x = s cos θ - t sin θ,  y = s sin θ + t cos θ
    a_coef = (sin_t / a_axis) ** 2 + (cos_t / b_axis) ** 2
    b_coef = 2.0 * s_arr * cos_t * sin_t * (1.0 / (b_axis**2) - 1.0 / (a_axis**2))
    c_coef = (s_arr * cos_t / a_axis) ** 2 + (s_arr * sin_t / b_axis) ** 2 - 1.0
    t_lo, t_hi, valid = _quadratic_roots(a_coef, b_coef, c_coef)
    length = np.zeros(shape, dtype=np.float64)
    length[valid] = np.maximum(t_hi[valid] - t_lo[valid], 0.0)
    return length


def ray_tissue_length(
    origins: np.ndarray,
    directions: np.ndarray,
    t_max: np.ndarray,
    semi_x: float,
    semi_y: float,
) -> np.ndarray:
    """Tissue distance along ``origin + t direction`` for ``t`` in ``[0, t_max]``.

    ``directions`` are unit vectors. The result is in field-of-view units.
    """
    origin = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
    direction = np.asarray(directions, dtype=np.float64).reshape(-1, 2)
    limit = np.asarray(t_max, dtype=np.float64).reshape(-1)
    n = int(origin.shape[0])
    if n == 0:
        return np.zeros(0, dtype=np.float64)
    a_axis = float(semi_x)
    b_axis = float(semi_y)
    if a_axis <= 0.0 or b_axis <= 0.0:
        return np.zeros(n, dtype=np.float64)

    ox = origin[:, 0]
    oy = origin[:, 1]
    dx = direction[:, 0]
    dy = direction[:, 1]
    a_coef = (dx / a_axis) ** 2 + (dy / b_axis) ** 2
    b_coef = 2.0 * (ox * dx / a_axis**2 + oy * dy / b_axis**2)
    c_coef = (ox / a_axis) ** 2 + (oy / b_axis) ** 2 - 1.0
    t_lo, t_hi, valid = _quadratic_roots(a_coef, b_coef, c_coef)
    enter = np.maximum(np.minimum(t_lo, t_hi), 0.0)
    leave = np.minimum(np.maximum(t_lo, t_hi), np.maximum(limit, 0.0))
    length = np.zeros(n, dtype=np.float64)
    inside = valid & (leave > enter)
    length[inside] = leave[inside] - enter[inside]
    return length


def survival_from_length(length_fov: np.ndarray, config: AttenuationConfig) -> np.ndarray:
    """``clip(exp(-μ L), a_min, 1)`` with ``L`` in centimeters."""
    cfg = config.clamp()
    length_cm = np.asarray(length_fov, dtype=np.float64) * cfg.cm_per_fov
    exponent = -cfg.mu_per_cm * np.maximum(length_cm, 0.0)
    # Keep the exponent finite so a huge μ cannot produce NaN.
    exponent = np.maximum(exponent, -80.0)
    raw = np.exp(exponent)
    if not cfg.active:
        return np.ones_like(raw)
    return np.clip(raw, cfg.a_min, 1.0)


def sinogram_survival(
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    config: AttenuationConfig,
) -> np.ndarray:
    """Survival weight for every parallel-beam bin center. Shape ``(n_θ, n_s)``."""
    cfg = config.clamp()
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    radii = np.asarray(s_centers, dtype=np.float64).ravel()
    n_theta = int(thetas.size)
    n_s = int(radii.size)
    if not cfg.active or n_theta == 0 or n_s == 0:
        return np.ones((n_theta, n_s), dtype=np.float64)
    semi_x, semi_y = ellipse_axes(cfg.brain_radius)
    theta_grid = np.broadcast_to(thetas[:, None], (n_theta, n_s))
    s_grid = np.broadcast_to(radii[None, :], (n_theta, n_s))
    chords = ellipse_chord_length(s_grid, theta_grid, semi_x, semi_y)
    return survival_from_length(chords, cfg)


def attenuation_map(grid_size: int, config: AttenuationConfig) -> np.ndarray:
    """μ in cm⁻¹ on the phantom display grid. Zero outside the brain ellipse."""
    cfg = config.clamp()
    n = int(grid_size)
    image = np.zeros((n, n), dtype=np.float64)
    if n <= 0 or not cfg.active:
        return image
    axis = np.linspace(-1.0, 1.0, n, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    semi_x, semi_y = ellipse_axes(cfg.brain_radius)
    inside = (xx / semi_x) ** 2 + (yy / semi_y) ** 2 <= 1.0
    image[inside] = cfg.mu_per_cm
    return image


def photon_path_lengths(
    origins: np.ndarray,
    directions_a: np.ndarray,
    directions_b: np.ndarray,
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    config: AttenuationConfig,
) -> tuple[np.ndarray, np.ndarray]:
    """Tissue paths of the two photons, in field-of-view units."""
    cfg = config.clamp()
    origin = np.asarray(origins, dtype=np.float64).reshape(-1, 2)
    hit_a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    hit_b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    if origin.shape[0] == 0 or not cfg.active:
        zeros = np.zeros(origin.shape[0], dtype=np.float64)
        return zeros, zeros.copy()
    semi_x, semi_y = ellipse_axes(cfg.brain_radius)
    t_a = np.linalg.norm(hit_a - origin, axis=1)
    t_b = np.linalg.norm(hit_b - origin, axis=1)
    length_a = ray_tissue_length(origin, directions_a, t_a, semi_x, semi_y)
    length_b = ray_tissue_length(origin, directions_b, t_b, semi_x, semi_y)
    return length_a, length_b


def apply_photon_survival(
    events: list,
    config: AttenuationConfig,
    rng: np.random.Generator,
) -> list:
    """Return copies whose detection flags include independent tissue survival.

    The originals are not modified. Survival random numbers come from ``rng``,
    which the acquisition engine does not use when attenuation is inactive.
    Random-coincidence records have no annihilation path and are left unchanged.
    """
    cfg = config.clamp()
    if not events or not cfg.active:
        return list(events)
    true_index = [i for i, event in enumerate(events) if not event.is_random]
    if not true_index:
        return list(events)
    origins = np.array([events[i].annihilation_xy for i in true_index], dtype=np.float64)
    hits_a = np.array([events[i].hit_a_xy for i in true_index], dtype=np.float64)
    hits_b = np.array([events[i].hit_b_xy for i in true_index], dtype=np.float64)
    angles = np.array([events[i].angle for i in true_index], dtype=np.float64)
    dir_a = np.column_stack((np.cos(angles), np.sin(angles)))
    dir_b = -dir_a
    length_a, length_b = photon_path_lengths(origins, dir_a, dir_b, hits_a, hits_b, cfg)
    # Unfloored survival is the physical probability. The reconstruction floor
    # is applied only to the sinogram weight, not to this Bernoulli trial.
    mu_l_a = cfg.mu_per_cm * cfg.cm_per_fov * length_a
    mu_l_b = cfg.mu_per_cm * cfg.cm_per_fov * length_b
    prob_a = np.exp(-np.minimum(mu_l_a, 80.0))
    prob_b = np.exp(-np.minimum(mu_l_b, 80.0))
    live_a = rng.random(len(true_index)) < prob_a
    live_b = rng.random(len(true_index)) < prob_b
    from petsim.physics.pet_engine import AnnihilationEvent

    copied: list = []
    cursor = 0
    true_set = set(true_index)
    for i, event in enumerate(events):
        if i not in true_set:
            copied.append(event)
            continue
        updated = AnnihilationEvent(
            annihilation_xy=event.annihilation_xy,
            angle=event.angle,
            hit_a_xy=event.hit_a_xy,
            hit_b_xy=event.hit_b_xy,
            detector_a=event.detector_a,
            detector_b=event.detector_b,
            detected_a=bool(event.detected_a and live_a[cursor]),
            detected_b=bool(event.detected_b and live_b[cursor]),
            energy_kev=event.energy_kev,
            energy_a_kev=event.energy_a_kev,
            energy_b_kev=event.energy_b_kev,
            ideal_hit_a_xy=event.ideal_hit_a_xy,
            ideal_hit_b_xy=event.ideal_hit_b_xy,
            flight_time_difference_ps=event.flight_time_difference_ps,
            timing_residual_ps=event.timing_residual_ps,
            measured_delta_t_ps=event.measured_delta_t_ps,
            passed_energy=event.passed_energy,
            passed_timing=event.passed_timing,
            is_scattered=event.is_scattered,
            is_random=event.is_random,
        )
        copied.append(updated)
        cursor += 1
    return copied


def compensate_sinogram(counts: np.ndarray, survival: np.ndarray, a_min: float) -> np.ndarray:
    """Reweight measured bins by ``1 / max(a, a_min)``. Zeros stay zero.

    This is the parallel-beam FBP weight. It is not applied inside MLEM.
    Multiplying a measured bin raises its expectation and its variance. It
    does not insert the photons that were absorbed.
    """
    measured = np.asarray(counts, dtype=np.float64)
    weight = np.asarray(survival, dtype=np.float64)
    if measured.shape != weight.shape:
        raise ValueError("survival must match the sinogram shape")
    floor = float(max(a_min, 1e-6))
    scale = 1.0 / np.maximum(weight, floor)
    corrected = measured * scale
    corrected[measured == 0.0] = 0.0
    return corrected
