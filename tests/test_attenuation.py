"""Stage 11 attenuation: survival, line integrals, and paired reconstruction."""

from __future__ import annotations

import numpy as np

from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.attenuation import (
    AttenuationConfig,
    attenuation_map,
    compensate_sinogram,
    ellipse_axes,
    ellipse_chord_length,
    photon_path_lengths,
    sinogram_survival,
    survival_from_length,
)
from petsim.physics.attenuation_study import run_attenuation_study
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.lor import endpoints_to_sinogram_coords
from petsim.reconstruction.mlem import mlem_reconstruct
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.reconstruction.projectors import backproject, forward_project
from petsim.reconstruction.sinogram import Sinogram, build_sinogram


def _config(**kwargs) -> AttenuationConfig:
    base = dict(enabled=True, mu_per_cm=0.096, cm_per_fov=10.0, brain_radius=0.55, a_min=1e-3)
    base.update(kwargs)
    return AttenuationConfig(**base)


def test_beer_lambert_matches_a_known_vertical_path():
    cfg = _config()
    _, semi_y = ellipse_axes(cfg.brain_radius)
    origin = np.array([[0.0, 0.0]])
    direction = np.array([[0.0, 1.0]])
    hit = np.array([[0.0, 0.92]])
    length, _ = photon_path_lengths(origin, direction, -direction, hit, -hit, cfg)
    assert np.isclose(length[0], semi_y, rtol=1e-6)
    expected = np.exp(-cfg.mu_per_cm * cfg.cm_per_fov * semi_y)
    assert np.isclose(survival_from_length(length, cfg)[0], expected)


def test_two_photon_paths_sum_to_the_sinogram_chord():
    cfg = _config()
    semi_x, semi_y = ellipse_axes(cfg.brain_radius)
    origin = np.array([[0.12, -0.08]])
    angle = 0.7
    direction = np.array([[np.cos(angle), np.sin(angle)]])
    # Ring hits are only needed for the path limit. Use a long limit past the ellipse.
    far = origin + 3.0 * direction
    back = origin - 3.0 * direction
    length_a, length_b = photon_path_lengths(origin, direction, -direction, far, back, cfg)
    theta, s = endpoints_to_sinogram_coords(far, back)
    chord = ellipse_chord_length(s, theta, semi_x, semi_y)
    assert np.isclose(length_a[0] + length_b[0], chord[0], rtol=1e-6, atol=1e-8)


def test_central_chord_is_the_ellipse_diameter():
    semi_x, semi_y = ellipse_axes(0.55)
    vertical = ellipse_chord_length(np.array([0.0]), np.array([0.0]), semi_x, semi_y)
    horizontal = ellipse_chord_length(np.array([0.0]), np.array([np.pi / 2]), semi_x, semi_y)
    assert np.isclose(vertical[0], 2.0 * semi_y)
    assert np.isclose(horizontal[0], 2.0 * semi_x)
    missed = ellipse_chord_length(np.array([2.0]), np.array([0.0]), semi_x, semi_y)
    assert missed[0] == 0.0


def test_disabled_attenuation_does_not_change_the_random_stream_or_reconstruction():
    engine = PETEngine(seed=11)
    plain = engine.simulate_batch(40, detector_efficiency=0.8)
    engine.set_seed(11)
    flagged = engine.simulate_batch(
        40,
        detector_efficiency=0.8,
        attenuation=AttenuationConfig(enabled=False, mu_per_cm=0.096),
    )
    engine.set_seed(11)
    zero_mu = engine.simulate_batch(40, detector_efficiency=0.8, attenuation=_config(mu_per_cm=0.0))
    for left, right in ((plain, flagged), (plain, zero_mu)):
        assert len(left) == len(right)
        for a, b in zip(left, right):
            assert a.annihilation_xy == b.annihilation_xy
            assert a.detected_a == b.detected_a and a.detected_b == b.detected_b

    hits_a = np.array([event.hit_a_xy for event in plain if event.is_coincidence])
    hits_b = np.array([event.hit_b_xy for event in plain if event.is_coincidence])
    baseline = reconstruct_from_endpoints(hits_a, hits_b, ReconstructionConfig(n_iterations=2, image_size=32))
    with_flag = reconstruct_from_endpoints(
        hits_a,
        hits_b,
        ReconstructionConfig(n_iterations=2, image_size=32),
        attenuation=AttenuationConfig(enabled=False),
    )
    assert np.array_equal(baseline.image, with_flag.image)


def test_survival_is_finite_and_empty_bins_stay_empty():
    cfg = _config(mu_per_cm=0.5, cm_per_fov=30.0, a_min=1e-3)
    thetas = np.linspace(0.0, np.pi, 8, endpoint=False)
    radii = np.linspace(-0.9, 0.9, 11)
    survival = sinogram_survival(thetas, radii, cfg)
    assert np.all(np.isfinite(survival))
    assert np.all(survival >= cfg.a_min - 1e-15)
    assert np.all(survival <= 1.0)
    counts = np.zeros_like(survival)
    counts[0, 0] = 4.0
    corrected = compensate_sinogram(counts, survival, cfg.a_min)
    assert corrected[0, 0] >= counts[0, 0]
    assert np.count_nonzero(corrected) == 1


def test_weighted_forward_and_adjoint_match():
    cfg = _config()
    rng = np.random.default_rng(0)
    image = rng.random((16, 16))
    sinogram = rng.random((10, 18))
    thetas = np.linspace(0.0, np.pi, 10, endpoint=False)
    radii = np.linspace(-0.9, 0.9, 18)
    weights = sinogram_survival(thetas, radii, cfg)
    forward = weights * forward_project(image, thetas, radii, fov=1.0)
    backward = backproject(weights * sinogram, thetas, radii, image_size=16, fov=1.0)
    assert np.allclose(np.sum(forward * sinogram), np.sum(image * backward), rtol=1e-8, atol=1e-8)


def test_active_attenuation_removes_true_coincidences_only():
    engine = PETEngine(BrainPhantom(PhantomState(tumor_enabled=True)), seed=5)
    engine.set_seed(5)
    ideal = engine.simulate_batch(300, detector_efficiency=1.0)
    engine.set_seed(5)
    attenuated = engine.simulate_batch(300, detector_efficiency=1.0, attenuation=_config())
    ideal_pairs = sum(event.is_coincidence for event in ideal)
    attenuated_pairs = sum(event.is_coincidence for event in attenuated)
    assert attenuated_pairs < ideal_pairs
    for left, right in zip(ideal, attenuated):
        assert left.annihilation_xy == right.annihilation_xy
        assert (not right.detected_a) or left.detected_a
        assert (not right.detected_b) or left.detected_b


def _disk_activity(size: int, radius: float = 0.28) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    image = np.zeros((size, size), dtype=np.float64)
    image[(xx - 0.15) ** 2 + (yy - 0.05) ** 2 <= radius**2] = 1.0
    return image


def test_noiseless_correction_moves_fbp_and_mlem_toward_the_unattenuated_image():
    cfg = _config()
    size = 32
    thetas = np.linspace(0.0, np.pi, 24, endpoint=False)
    radii = np.linspace(-0.92, 0.92, 32)
    activity = _disk_activity(size)
    ideal = forward_project(activity, thetas, radii, fov=1.0)
    weights = sinogram_survival(thetas, radii, cfg)
    attenuated = ideal * weights
    # Sinogram edges are built so the stored centers are the projector samples.
    ds = float(radii[1] - radii[0])
    edges_s = np.concatenate(([radii[0] - 0.5 * ds], radii[:-1] + 0.5 * ds, [radii[-1] + 0.5 * ds]))
    dth = float(thetas[1] - thetas[0])
    edges_theta = np.concatenate(([thetas[0] - 0.5 * dth], thetas[:-1] + 0.5 * dth, [thetas[-1] + 0.5 * dth]))

    def sino(data: np.ndarray) -> Sinogram:
        return Sinogram(data=data, theta_edges=edges_theta, s_edges=edges_s, n_events=1)

    assert np.allclose(sino(ideal).theta_centers, thetas)
    ideal_fbp = filtered_backprojection(sino(ideal), filter_name="ramp", image_size=size, fov=1.0)
    raw_fbp = filtered_backprojection(sino(attenuated), filter_name="ramp", image_size=size, fov=1.0)
    corrected_counts = compensate_sinogram(attenuated, weights, cfg.a_min)
    corrected_fbp = filtered_backprojection(
        sino(corrected_counts), filter_name="ramp", image_size=size, fov=1.0
    )
    ideal_mlem, _, _ = mlem_reconstruct(sino(ideal), n_iterations=8, image_size=size, fov=1.0)
    raw_mlem, _, _ = mlem_reconstruct(sino(attenuated), n_iterations=8, image_size=size, fov=1.0)
    fixed_mlem, _, _ = mlem_reconstruct(
        sino(attenuated), n_iterations=8, image_size=size, fov=1.0, survival=weights
    )
    # Pre-corrected counts must not also be given to the weighted MLEM update.
    double, _, _ = mlem_reconstruct(
        sino(corrected_counts), n_iterations=8, image_size=size, fov=1.0, survival=weights
    )

    def gap(reference: np.ndarray, candidate: np.ndarray) -> float:
        mask = activity > 0
        return float(np.mean(np.abs(np.maximum(candidate, 0.0)[mask] - np.maximum(reference, 0.0)[mask])))

    assert gap(ideal_fbp, corrected_fbp) < gap(ideal_fbp, raw_fbp)
    assert gap(ideal_mlem, fixed_mlem) < gap(ideal_mlem, raw_mlem)
    assert gap(ideal_mlem, double) > gap(ideal_mlem, fixed_mlem)


def test_noisy_recovery_over_several_seeds_does_not_claim_less_noise():
    state = PhantomState(tumor_enabled=True, tumor_x=0.18, tumor_y=0.12, tumor_radius=0.12, tumor_intensity=1.0)
    result = run_attenuation_study(
        state,
        _config(),
        seeds=(1, 2, 3),
        method="fbp",
        n_events=900,
        n_iterations=4,
        image_size=32,
        n_theta=24,
        n_s=32,
    )
    assert len(result.comparisons) == 3
    assert result.mean_corrected_recovery > result.mean_uncorrected_recovery
    # At least one seed should show that correction does not erase noise.
    noisier = [
        item for item in result.comparisons if item.corrected.background_cv + 1e-9 >= item.attenuated.background_cv
    ]
    assert noisier
    assert "noise" in result.message.lower()
    # The measured list used for correction is the attenuated one: same count.
    for item in result.comparisons:
        assert item.corrected.n_coincidences == item.attenuated.n_coincidences
        assert item.attenuated.n_coincidences < item.ideal.n_coincidences


def test_bin_center_weight_uses_the_same_line_as_binned_endpoints():
    cfg = _config()
    hits_a = np.array([[0.0, -0.5], [0.4, 0.2]])
    hits_b = np.array([[0.0, 0.5], [-0.3, -0.4]])
    theta, s = endpoints_to_sinogram_coords(hits_a, hits_b)
    semi_x, semi_y = ellipse_axes(cfg.brain_radius)
    direct = ellipse_chord_length(s, theta, semi_x, semi_y)
    for index in range(theta.shape[0]):
        weight = sinogram_survival(np.array([float(theta[index])]), np.array([float(s[index])]), cfg)
        assert np.isclose(weight[0, 0], survival_from_length(np.array([direct[index]]), cfg)[0])
    built = build_sinogram(hits_a, hits_b, n_theta=4, n_s=4, s_max=0.92)
    assert built.theta_centers.shape[0] == 4
    assert attenuation_map(16, cfg).shape == (16, 16)
    assert np.max(attenuation_map(16, cfg)) == cfg.mu_per_cm
