"""Coincidence-count comparison workflow tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.count_comparison import (
    DEFAULT_COUNT_LEVELS,
    compare_by_coincidence_count,
)
from petsim.models.phantom import PhantomState
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig


def _simulate_coincidences(n_events: int = 25000, seed: int = 7):
    phantom = BrainPhantom(
        PhantomState(
            tumor_enabled=True,
            tumor_x=0.18,
            tumor_y=0.12,
            tumor_radius=0.10,
            tumor_intensity=1.0,
            brain_radius=0.55,
            brain_intensity=0.35,
        )
    )
    ring = DetectorRing(n_detectors=48)
    engine = PETEngine(phantom, ring)
    engine.set_seed(seed)
    events = engine.simulate_batch(n_events, detector_efficiency=0.85)
    buf = AcquisitionBuffer()
    buf.extend(events)
    return buf.coincidence_endpoints(), phantom.state.clamp()


def test_empty_acquisition():
    a = np.zeros((0, 2))
    b = np.zeros((0, 2))
    out = compare_by_coincidence_count(
        a,
        b,
        ReconstructionConfig(),
        PhantomState(tumor_enabled=True),
        target_counts=(1000,),
    )
    assert out.total_available == 0
    assert not out.has_any
    assert "no coincidence" in out.message.lower()


def test_insufficient_counts_reported():
    (hits_a, hits_b), state = _simulate_coincidences(n_events=2000, seed=3)
    n = hits_a.shape[0]
    assert n > 0
    out = compare_by_coincidence_count(
        hits_a,
        hits_b,
        ReconstructionConfig(n_theta=48, n_s=64, image_size=64),
        state,
        target_counts=(1000, 100_000),
    )
    assert out.total_available == n
    by_target = {level.target_count: level for level in out.levels}
    if n >= 1000:
        assert by_target[1000].available
        assert by_target[1000].actual_count == 1000
    assert not by_target[100_000].available
    assert "insufficient" in by_target[100_000].message.lower()


def test_nested_prefixes_are_prefixes():
    (hits_a, hits_b), state = _simulate_coincidences(n_events=20000, seed=11)
    n = hits_a.shape[0]
    assert n >= 3000
    cfg = ReconstructionConfig(n_theta=48, n_s=64, image_size=64, filter_name="hann")
    out = compare_by_coincidence_count(
        hits_a,
        hits_b,
        cfg,
        state,
        target_counts=(1000, 3000),
    )
    assert out.has_any
    # Nested: level images should exist for available targets.
    available = [level for level in out.levels if level.available]
    assert len(available) >= 2
    for level in available:
        assert level.result is not None
        assert level.result.n_coincidences == level.actual_count
        assert level.analysis is not None


def test_default_levels_tuple():
    assert DEFAULT_COUNT_LEVELS == (1000, 3000, 5000, 10000)
