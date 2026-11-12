"""Quantitative Stage 3 image-quality checks (localization + tumor contrast).

These tests use Stage 2 coincidence endpoints only (no annihilation truth as
reconstruction input). Truth coordinates are used solely for scoring.
"""

from __future__ import annotations

import numpy as np
import pytest

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.lor import endpoints_to_sinogram_coords
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.reconstruction.sinogram import analytic_disk_sinogram, build_sinogram
from petsim.reconstruction.fbp import filtered_backprojection


def _coincidences(state: PhantomState, *, seed: int, n_events: int, efficiency: float = 1.0):
    eng = PETEngine(BrainPhantom(state=state), DetectorRing(), seed=seed)
    events = eng.simulate_batch(n_events, detector_efficiency=efficiency)
    ha = np.array([e.hit_a_xy for e in events if e.is_coincidence], dtype=np.float64)
    hb = np.array([e.hit_b_xy for e in events if e.is_coincidence], dtype=np.float64)
    truth = np.array(
        [e.annihilation_xy for e in events if e.is_coincidence], dtype=np.float64
    )
    return ha, hb, truth


def _roi_masks(size: int, tumor_xy, tumor_r, brain_r=0.55):
    axis = np.linspace(-1.0, 1.0, size)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    tumor = (xx - tumor_xy[0]) ** 2 + (yy - tumor_xy[1]) ** 2 <= tumor_r**2
    brain = (xx / 0.85) ** 2 + yy**2 <= brain_r**2
    return xx, yy, tumor, brain & ~tumor, brain


def test_measured_s_matches_truth_projection():
    """Coincidence→(θ,s) must be consistent with the geometric LOR (validation only)."""
    state = PhantomState(tumor_x=0.18, tumor_y=0.12)
    ha, hb, truth = _coincidences(state, seed=7, n_events=4000, efficiency=1.0)
    assert ha.shape[0] > 500
    theta, s = endpoints_to_sinogram_coords(ha, hb)
    s_from_truth = truth[:, 0] * np.cos(theta) + truth[:, 1] * np.sin(theta)
    assert np.mean(np.abs(s - s_from_truth)) < 1e-10


def test_low_count_defaults_recover_tumor_localization_and_contrast():
    """~3k coincidences with educational defaults: tumor hotter than brain background."""
    tumor_xy = (0.20, 0.12)
    state = PhantomState(
        brain_intensity=0.20,
        tumor_enabled=True,
        tumor_x=tumor_xy[0],
        tumor_y=tumor_xy[1],
        tumor_radius=0.10,
        tumor_intensity=2.0,
    )
    ha, hb, _ = _coincidences(state, seed=42, n_events=5000, efficiency=0.8)
    assert 2500 <= ha.shape[0] <= 4500  # ~ε²·N with ε=0.8 → ~3200

    result = reconstruct_from_endpoints(ha, hb, ReconstructionConfig())  # defaults
    img = result.image_display
    xx, yy, tumor, background, brain = _roi_masks(128, tumor_xy, 0.10)

    contrast = float(img[tumor].mean() / max(img[background].mean(), 1e-12))
    assert contrast > 3.0

    # Hot-spot COM of the brightest 0.5% pixels should land near the tumor.
    thr = np.percentile(img, 99.5)
    hot = img >= thr
    mass = img[hot].sum()
    cx = float((img[hot] * xx[hot]).sum() / mass)
    cy = float((img[hot] * yy[hot]).sum() / mass)
    assert np.hypot(cx - tumor_xy[0], cy - tumor_xy[1]) < 0.12

    # Streak energy outside the brain should not dominate after smoothing.
    outside = float(img[~brain].sum() / max(img.sum(), 1e-12))
    assert outside < 0.45


def test_high_count_synthetic_phantom_localizes_tumor():
    """Sufficient counts → peak near the simulated tumor (Stage 2 endpoints only)."""
    tumor_xy = (-0.22, 0.15)
    state = PhantomState(
        brain_intensity=0.15,
        tumor_enabled=True,
        tumor_x=tumor_xy[0],
        tumor_y=tumor_xy[1],
        tumor_radius=0.11,
        tumor_intensity=2.0,
    )
    ha, hb, _ = _coincidences(state, seed=11, n_events=20_000, efficiency=1.0)
    assert ha.shape[0] >= 15_000

    result = reconstruct_from_endpoints(
        ha,
        hb,
        ReconstructionConfig(filter_name="hann", n_theta=96, smooth_sigma_s=1.5, smooth_sigma_theta=0.75),
    )
    img = result.image_display
    axis = np.linspace(-1.0, 1.0, 128)
    # Search a band around the true tumor y to reduce far streak influence.
    y0 = tumor_xy[1]
    rows = np.where(np.abs(axis - y0) <= 0.20)[0]
    sub = img[rows, :]
    iy, ix = np.unravel_index(np.argmax(sub), sub.shape)
    px, py = axis[ix], axis[rows[iy]]
    assert np.hypot(px - tumor_xy[0], py - tumor_xy[1]) < 0.15

    xx, yy, tumor, background, _ = _roi_masks(128, tumor_xy, 0.11)
    contrast = float(img[tumor].mean() / max(img[background].mean(), 1e-12))
    assert contrast > 4.0


def test_legacy_180_ramp_is_streakier_than_defaults():
    """Document that 180×ramp on ~3k counts is expected to look worse (not a mapping bug)."""
    state = PhantomState(
        brain_intensity=0.20,
        tumor_enabled=True,
        tumor_x=0.2,
        tumor_y=0.12,
        tumor_radius=0.10,
        tumor_intensity=2.0,
    )
    ha, hb, _ = _coincidences(state, seed=42, n_events=5000, efficiency=0.8)
    legacy = reconstruct_from_endpoints(
        ha,
        hb,
        ReconstructionConfig(
            filter_name="ramp",
            n_theta=180,
            smooth_sigma_s=0.0,
            smooth_sigma_theta=0.0,
        ),
    )
    improved = reconstruct_from_endpoints(ha, hb, ReconstructionConfig())
    xx, yy, tumor, background, brain = _roi_masks(128, (0.2, 0.12), 0.10)

    def outside_frac(img):
        pos = np.maximum(img, 0.0)
        return float(pos[~brain].sum() / max(pos.sum(), 1e-12))

    def contrast(img):
        pos = np.maximum(img, 0.0)
        return float(pos[tumor].mean() / max(pos[background].mean(), 1e-12))

    assert outside_frac(improved.image) < outside_frac(legacy.image)
    assert contrast(improved.image) > contrast(legacy.image)


def test_analytic_disk_still_centered_with_smoothing_path():
    """Independent projection fixture remains centered under FBP (no PET LORs)."""
    sino = analytic_disk_sinogram(radius=0.22, n_theta=96, n_s=128, s_max=1.0)
    # Apply the same smoother the pipeline uses.
    from petsim.reconstruction.sinogram import smooth_sinogram_data
    from petsim.reconstruction.sinogram import Sinogram

    sino = Sinogram(
        data=smooth_sinogram_data(sino.data, sigma_s=1.5, sigma_theta=0.75),
        theta_edges=sino.theta_edges,
        s_edges=sino.s_edges,
        n_events=0,
    )
    img = filtered_backprojection(sino, filter_name="hann", image_size=128, fov=1.0)
    axis = np.linspace(-1.0, 1.0, 128)
    pos = np.maximum(img, 0.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    mass = pos.sum()
    assert mass > 0
    cx = float((pos * xx).sum() / mass)
    cy = float((pos * yy).sum() / mass)
    assert abs(cx) < 0.05 and abs(cy) < 0.05


def test_sinogram_binning_preserves_event_count():
    state = PhantomState()
    ha, hb, _ = _coincidences(state, seed=1, n_events=3000, efficiency=1.0)
    sino = build_sinogram(ha, hb, n_theta=96, n_s=128, s_max=0.92)
    # Events with |s|<=s_max should all be counted (phantom is well inside ring).
    assert sino.data.sum() == pytest.approx(float(ha.shape[0]), abs=1.0)
