"""Physics engine tests: coincidences, reproducibility, uptake sampling."""

from __future__ import annotations

import numpy as np
import pytest

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.pet_engine import PETEngine


def _engine(seed: int = 0, state: PhantomState | None = None) -> PETEngine:
    phantom = BrainPhantom(state=state or PhantomState())
    return PETEngine(phantom, DetectorRing(n_detectors=48), seed=seed)


def test_detector_indices_from_simulated_events_in_range():
    eng = _engine(seed=7)
    events = eng.simulate_batch(2000, detector_efficiency=1.0)
    for ev in events:
        assert 0 <= ev.detector_a <= 47
        assert 0 <= ev.detector_b <= 47
        # Opposite photons should almost never share a crystal.
        assert ev.detector_a != ev.detector_b


def test_photon_hits_lie_on_ring():
    ring = DetectorRing(n_detectors=48)
    eng = PETEngine(BrainPhantom(), ring, seed=3)
    events = eng.simulate_batch(500, detector_efficiency=1.0)
    for ev in events:
        ra = np.hypot(*ev.hit_a_xy)
        rb = np.hypot(*ev.hit_b_xy)
        assert abs(ra - ring.radius) < 1e-8
        assert abs(rb - ring.radius) < 1e-8


def test_efficiency_100_percent_all_coincidences():
    eng = _engine(seed=11)
    events = eng.simulate_batch(3000, detector_efficiency=1.0)
    assert len(events) == 3000
    assert all(ev.is_coincidence for ev in events)


def test_efficiency_50_percent_about_25_percent_coincidences():
    # Large N so the empirical rate is close to ε² = 0.25.
    eng = _engine(seed=99)
    n = 20_000
    events = eng.simulate_batch(n, detector_efficiency=0.5)
    rate = sum(ev.is_coincidence for ev in events) / n
    assert abs(rate - 0.25) < 0.02  # within ~2 percentage points


def test_efficiency_80_percent_about_64_percent_coincidences():
    eng = _engine(seed=123)
    n = 15_000
    events = eng.simulate_batch(n, detector_efficiency=0.8)
    rate = sum(ev.is_coincidence for ev in events) / n
    assert abs(rate - 0.64) < 0.025


def test_fixed_seed_reproducible():
    a = _engine(seed=42).simulate_batch(200, detector_efficiency=0.7)
    b = _engine(seed=42).simulate_batch(200, detector_efficiency=0.7)
    assert [ev.annihilation_xy for ev in a] == [ev.annihilation_xy for ev in b]
    assert [ev.angle for ev in a] == [ev.angle for ev in b]
    assert [ev.detector_pair for ev in a] == [ev.detector_pair for ev in b]
    assert [ev.is_coincidence for ev in a] == [ev.is_coincidence for ev in b]


def test_different_seeds_differ():
    a = _engine(seed=1).simulate_batch(100, detector_efficiency=1.0)
    b = _engine(seed=2).simulate_batch(100, detector_efficiency=1.0)
    assert [ev.annihilation_xy for ev in a] != [ev.annihilation_xy for ev in b]


def test_higher_uptake_regions_sampled_more_often():
    """Tumor pixels (high uptake) should receive more annihilations than brain."""
    state = PhantomState(
        brain_radius=0.55,
        brain_intensity=0.2,
        tumor_enabled=True,
        tumor_x=0.2,
        tumor_y=0.15,
        tumor_radius=0.12,
        tumor_intensity=2.0,
    )
    eng = _engine(seed=5, state=state)
    activity = eng.phantom.rasterize()
    n = activity.shape[0]
    axis = np.linspace(-1.0, 1.0, n)
    xx, yy = np.meshgrid(axis, axis)
    tumor_mask = (xx - state.tumor_x) ** 2 + (yy - state.tumor_y) ** 2 <= state.tumor_radius**2
    brain_mask = (xx / 0.85) ** 2 + yy**2 <= state.brain_radius**2
    brain_only = brain_mask & ~tumor_mask

    positions = eng.sample_annihilation_positions(20_000)
    # Count samples falling in each region (nearest-pixel style via bounds).
    def in_mask(pts: np.ndarray, mask: np.ndarray) -> int:
        # Map continuous coords back to pixel indices.
        ix = np.clip(np.rint((pts[:, 0] + 1.0) * (n - 1) / 2.0).astype(int), 0, n - 1)
        iy = np.clip(np.rint((pts[:, 1] + 1.0) * (n - 1) / 2.0).astype(int), 0, n - 1)
        return int(mask[iy, ix].sum())

    n_tumor = in_mask(positions, tumor_mask)
    n_brain = in_mask(positions, brain_only)

    # Per-pixel mean: tumor should dominate despite smaller area.
    tumor_per_pix = n_tumor / max(int(tumor_mask.sum()), 1)
    brain_per_pix = n_brain / max(int(brain_only.sum()), 1)
    assert tumor_per_pix > brain_per_pix * 2.0


def test_expected_coincidence_fraction_formula():
    eng = _engine()
    assert eng.expected_coincidence_fraction(0.8) == pytest.approx(0.64)
    assert eng.expected_coincidence_fraction(0.5) == pytest.approx(0.25)
    assert eng.expected_coincidence_fraction(1.0) == pytest.approx(1.0)
