"""Stage 12 time of flight: timing sign, full-space sensitivity, and identity."""

from __future__ import annotations

import inspect

import numpy as np

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.attenuation import AttenuationConfig
from petsim.physics.detector_effects import SPEED_OF_LIGHT_CM_PER_PS, fwhm_to_sigma
from petsim.physics.pet_engine import PETEngine
from petsim.physics.tof import (
    TOFConfig,
    discrete_tof_kernel,
    flight_time_difference_ps,
    most_likely_points,
    signed_offset_fov,
    spatial_sigma_fov,
    time_bin_centers,
)
from petsim.physics.tof_study import run_tof_study
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.reconstruction.tof_mlem import (
    backproject_tof,
    forward_project_tof,
    sensitivity_tof,
    tof_mlem_reconstruct,
)


def _grid():
    config = TOFConfig(enabled=True, fwhm_ps=250.0, cm_per_fov=10.0, n_t=15, t_max=1.45)
    thetas = np.linspace(0.0, np.pi, 10, endpoint=False)
    radii = np.linspace(-0.9, 0.9, 14)
    times = time_bin_centers(config.n_t, config.t_max)
    sigma = spatial_sigma_fov(config.fwhm_ps, config.cm_per_fov)
    return config, thetas, radii, times, sigma


def _disk(size: int = 14) -> np.ndarray:
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return ((xx - 0.05) ** 2 + (yy + 0.02) ** 2 <= 0.22**2).astype(np.float64)


def test_timing_sign_and_half_path():
    origin = np.array([0.2, 0.0])
    radius = 0.92
    hit_a = np.array([[radius, 0.0]])
    hit_b = np.array([[-radius, 0.0]])
    path_a = float(np.linalg.norm(hit_a - origin))
    path_b = float(np.linalg.norm(hit_b - origin))
    delta_t = flight_time_difference_ps(np.array([path_a]), np.array([path_b]), 10.0)
    assert delta_t[0] < 0.0
    point = most_likely_points(hit_a, hit_b, delta_t, 10.0)[0]
    assert np.allclose(point, origin)
    half_path = 0.5 * (path_a - path_b)
    assert np.isclose(signed_offset_fov(delta_t, 10.0)[0], half_path)


def test_spatial_sigma_matches_fwhm():
    fwhm = 400.0
    expected = SPEED_OF_LIGHT_CM_PER_PS * fwhm_to_sigma(fwhm) / 2.0 / 10.0
    assert np.isclose(spatial_sigma_fov(fwhm, 10.0), expected)


def test_narrower_kernel_is_more_localized():
    centers = time_bin_centers(41, 1.45)
    narrow = discrete_tof_kernel(np.array([0.0]), centers, spatial_sigma_fov(100.0, 10.0))[0]
    wide = discrete_tof_kernel(np.array([0.0]), centers, spatial_sigma_fov(800.0, 10.0))[0]
    narrow_moment = float(np.sum(narrow * centers**2) / np.sum(narrow))
    wide_moment = float(np.sum(wide * centers**2) / np.sum(wide))
    assert narrow_moment < wide_moment
    assert narrow.sum() <= 1.0 + 1e-6
    assert wide.sum() <= 1.0 + 1e-6


def test_adjoint_and_full_space_sensitivity():
    _config, thetas, radii, times, sigma = _grid()
    rng = np.random.default_rng(0)
    image = rng.random((14, 14))
    measurement = rng.random((thetas.size, radii.size, times.size))
    survival = np.full((thetas.size, radii.size), 0.4)
    forward = forward_project_tof(image, thetas, radii, times, sigma, survival=survival)
    backward = backproject_tof(
        measurement, thetas, radii, times, sigma, image_size=14, survival=survival
    )
    assert np.isclose(np.sum(forward * measurement), np.sum(image * backward))
    ones = np.ones_like(measurement)
    sensitivity = sensitivity_tof(
        thetas, radii, times, sigma, image_size=14, survival=survival
    )
    assert np.allclose(
        sensitivity,
        backproject_tof(ones, thetas, radii, times, sigma, image_size=14, survival=survival),
    )


def test_noiseless_disk_is_unbiased_and_observed_sensitivity_is_not():
    config, thetas, radii, times, sigma = _grid()
    disk = _disk()
    measured = forward_project_tof(disk, thetas, radii, times, sigma)
    image, _history, cancelled = tof_mlem_reconstruct(
        measured, thetas, radii, times, config, n_iterations=20, image_size=14
    )
    assert not cancelled
    inside = disk > 0
    assert abs(float(image[inside].mean()) - 1.0) < 0.08
    assert float(image[~inside].mean()) < 0.01

    observed = backproject_tof(
        (measured > 1e-6).astype(np.float64),
        thetas,
        radii,
        times,
        sigma,
        image_size=14,
    )
    full = sensitivity_tof(thetas, radii, times, sigma, image_size=14)
    assert float(full.mean()) > float(observed.mean())
    guess = np.ones_like(disk)
    for _ in range(20):
        estimate = forward_project_tof(guess, thetas, radii, times, sigma)
        ratio = np.zeros_like(measured)
        positive = estimate > 1e-12
        ratio[positive] = measured[positive] / estimate[positive]
        correction = backproject_tof(ratio, thetas, radii, times, sigma, image_size=14)
        guess = np.where(observed > 1e-8, guess * correction / observed, 0.0)
    assert float(guess[~inside].mean()) > float(image[~inside].mean()) * 5.0


def test_attenuation_is_applied_once():
    config, thetas, radii, times, sigma = _grid()
    disk = _disk()
    survival = np.full((thetas.size, radii.size), 0.35)
    measured = forward_project_tof(disk, thetas, radii, times, sigma, survival=survival)
    once, _, _ = tof_mlem_reconstruct(
        measured,
        thetas,
        radii,
        times,
        config,
        n_iterations=12,
        image_size=14,
        survival=survival,
    )
    divided = measured / survival[:, :, None]
    twice, _, _ = tof_mlem_reconstruct(
        divided,
        thetas,
        radii,
        times,
        config,
        n_iterations=12,
        image_size=14,
        survival=survival,
    )
    assert np.max(np.abs(once - twice)) > 0.05
    assert abs(float(once[disk > 0].mean()) - 1.0) < 0.15


def test_disabled_tof_matches_stage11_stream_and_images():
    phantom = BrainPhantom()
    ring = DetectorRing(n_detectors=24)
    first = PETEngine(phantom, ring, seed=11)
    omitted = first.simulate_batch(30, detector_efficiency=0.8)
    second = PETEngine(phantom, ring, seed=11)
    disabled = second.simulate_batch(
        30, detector_efficiency=0.8, tof=TOFConfig(enabled=False)
    )
    assert [event.annihilation_xy for event in omitted] == [
        event.annihilation_xy for event in disabled
    ]
    assert [event.detected_a for event in omitted] == [event.detected_a for event in disabled]
    assert [event.detected_b for event in omitted] == [event.detected_b for event in disabled]
    assert all(event.measured_delta_t_ps == 0.0 for event in disabled)

    def _reconstruct(events, tof):
        pairs = [event for event in events if event.is_coincidence]
        hits_a = np.array([event.hit_a_xy for event in pairs])
        hits_b = np.array([event.hit_b_xy for event in pairs])
        config = ReconstructionConfig(
            method="mlem", n_iterations=3, image_size=24, n_theta=12, n_s=16
        )
        result = reconstruct_from_endpoints(hits_a, hits_b, config, tof=tof)
        fbp = reconstruct_from_endpoints(
            hits_a, hits_b, ReconstructionConfig(method="fbp", image_size=24, n_theta=12, n_s=16), tof=tof
        )
        return result.image, fbp.image

    mlem_off, fbp_off = _reconstruct(disabled, TOFConfig(enabled=False))
    mlem_none, fbp_none = _reconstruct(omitted, None)
    assert np.array_equal(mlem_off, mlem_none)
    assert np.array_equal(fbp_off, fbp_none)


def test_timed_acquisition_is_reproducible_and_sharper_timing_tightens_the_cloud():
    state = PhantomState(tumor_radius=0.04, tumor_intensity=1.0, brain_intensity=0.05)

    def cloud(fwhm: float, seed: int = 5):
        engine = PETEngine(BrainPhantom(state=state), DetectorRing(n_detectors=24), seed=seed)
        events = engine.simulate_batch(
            180,
            detector_efficiency=1.0,
            tof=TOFConfig(enabled=True, fwhm_ps=fwhm, cm_per_fov=10.0),
        )
        pairs = [event for event in events if event.is_coincidence and not event.is_random]
        hits_a = np.array([event.hit_a_xy for event in pairs])
        hits_b = np.array([event.hit_b_xy for event in pairs])
        delta_t = np.array([event.measured_delta_t_ps for event in pairs])
        points = most_likely_points(hits_a, hits_b, delta_t, 10.0)
        return points, delta_t

    sharp, times = cloud(80.0)
    repeat, times_again = cloud(80.0)
    broad, _times = cloud(900.0)
    assert np.allclose(times, times_again)
    assert np.allclose(sharp, repeat)
    assert float(np.var(sharp, axis=0).sum()) < float(np.var(broad, axis=0).sum())


def test_randoms_are_excluded_and_tumor_is_not_a_recon_argument():
    buffer = AcquisitionBuffer()
    true = _event(is_random=False, delta_t=40.0)
    random = _event(is_random=True, delta_t=0.0)
    buffer.extend([true, random])
    hits_a, _hits_b, delta_t = buffer.true_coincidence_tof()
    assert hits_a.shape[0] == 1
    assert np.isclose(delta_t[0], 40.0)
    assert "tumor" not in inspect.signature(tof_mlem_reconstruct).parameters


def test_matched_study_keeps_every_seed_without_requiring_every_metric():
    result = run_tof_study(
        PhantomState(),
        TOFConfig(enabled=True, fwhm_ps=300.0, cm_per_fov=10.0, n_t=11, t_max=1.45),
        seeds=(2, 3),
        n_events=60,
        n_iterations=3,
        image_size=24,
        n_theta=12,
        n_s=16,
    )
    assert len(result.comparisons) == 2
    assert "random" in result.message.lower()
    assert np.isfinite(result.mean("conventional_nrmse"))
    assert np.isfinite(result.mean("tof_nrmse"))
    # The two scores are recorded. This noisy pair is not required to favor TOF.
    assert result.comparisons[0].seed == 2


def test_attenuation_and_tof_share_one_survival_map():
    state = PhantomState()
    atten = AttenuationConfig(enabled=True, mu_per_cm=0.096, cm_per_fov=10.0, brain_radius=0.55)
    engine = PETEngine(BrainPhantom(state=state), DetectorRing(n_detectors=24), seed=8)
    events = engine.simulate_batch(
        40,
        detector_efficiency=1.0,
        attenuation=atten,
        tof=TOFConfig(enabled=True, fwhm_ps=400.0, cm_per_fov=10.0),
    )
    pairs = [event for event in events if event.is_coincidence]
    hits_a = np.array([event.hit_a_xy for event in pairs])
    hits_b = np.array([event.hit_b_xy for event in pairs])
    delta_t = np.array([event.measured_delta_t_ps for event in pairs])
    config = ReconstructionConfig(method="mlem", n_iterations=3, image_size=24, n_theta=12, n_s=16)
    corrected = reconstruct_from_endpoints(
        hits_a, hits_b, config, attenuation=atten, tof=TOFConfig(enabled=True, fwhm_ps=400.0), delta_t_ps=delta_t
    )
    assert "attenuation once" in corrected.message
    assert "randoms excluded" in corrected.message


def _event(*, is_random: bool, delta_t: float):
    from petsim.physics.pet_engine import AnnihilationEvent

    return AnnihilationEvent(
        annihilation_xy=(0.0, 0.0),
        angle=0.0,
        hit_a_xy=(0.9, 0.0),
        hit_b_xy=(-0.9, 0.0),
        detector_a=0,
        detector_b=12,
        detected_a=True,
        detected_b=True,
        measured_delta_t_ps=delta_t,
        is_random=is_random,
    )
