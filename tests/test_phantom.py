"""Unit tests for synthetic phantom rasterization (no Qt / no physics)."""

from __future__ import annotations

import numpy as np

from petsim.models.phantom import PhantomParameters, SyntheticBrainPhantom
from petsim.models.scanner import DetectorRingGeometry


def test_rasterize_shape_and_range() -> None:
    phantom = SyntheticBrainPhantom(resolution=64)
    field = phantom.rasterize(PhantomParameters())
    assert field.shape == (64, 64)
    assert field.dtype == np.float32
    assert float(field.min()) >= 0.0
    assert float(field.max()) <= 1.0


def test_tumor_increases_local_activity() -> None:
    phantom = SyntheticBrainPhantom(resolution=128)
    base = phantom.rasterize(PhantomParameters(show_tumor=False, brain_intensity=0.3))
    with_tumor = phantom.rasterize(
        PhantomParameters(
            show_tumor=True,
            brain_intensity=0.3,
            tumor_x=0.0,
            tumor_y=0.0,
            tumor_radius=0.15,
            tumor_intensity=1.0,
        )
    )
    # Center voxel should be brighter with the tumor present
    c = 128 // 2
    assert with_tumor[c, c] > base[c, c]


def test_detector_ring_has_48_elements() -> None:
    geom = DetectorRingGeometry(n_detectors=48)
    x, y = geom.detector_positions()
    assert len(x) == 48
    assert len(y) == 48
    radii = np.sqrt(x**2 + y**2)
    assert np.allclose(radii, geom.radius, atol=1e-9)
