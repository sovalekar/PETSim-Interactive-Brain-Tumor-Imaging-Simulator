"""Geometry unit tests for Stage 2 PET physics."""

from __future__ import annotations

import numpy as np

from petsim.physics.geometry import hit_to_detector_index, ray_circle_intersection


def test_ray_hits_lie_on_detector_ring():
    radius = 0.92
    origins = np.array([[0.1, -0.2], [0.0, 0.0], [-0.3, 0.15]], dtype=np.float64)
    angles = np.array([0.3, 1.1, 4.2], dtype=np.float64)
    directions = np.column_stack((np.cos(angles), np.sin(angles)))
    hits = ray_circle_intersection(origins, directions, radius)
    radii = np.linalg.norm(hits, axis=1)
    assert np.allclose(radii, radius, atol=1e-9)

    # Opposite photons also land on the ring.
    hits_b = ray_circle_intersection(origins, -directions, radius)
    radii_b = np.linalg.norm(hits_b, axis=1)
    assert np.allclose(radii_b, radius, atol=1e-9)


def test_detector_indices_always_in_range():
    n = 48
    # Sample many angles around the ring.
    theta = np.linspace(0.0, 2.0 * np.pi, 500, endpoint=False)
    hits = np.column_stack((np.cos(theta), np.sin(theta))) * 0.92
    idx = hit_to_detector_index(hits, n)
    assert idx.min() >= 0
    assert idx.max() <= n - 1
    assert idx.shape == (500,)


def test_crystal_zero_maps_near_positive_x():
    hits = np.array([[0.92, 0.0], [0.92, 1e-6]], dtype=np.float64)
    idx = hit_to_detector_index(hits, 48)
    assert idx[0] == 0
    assert idx[1] in (0, 1)
