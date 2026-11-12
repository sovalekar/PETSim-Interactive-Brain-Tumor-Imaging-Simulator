"""Tests for LOR endpoint → (θ, s) conversion."""

from __future__ import annotations

import numpy as np
import pytest

from petsim.reconstruction.lor import (
    analytic_point_source_coords,
    endpoints_to_sinogram_coords,
)


def test_horizontal_lor_normal_is_vertical():
    # Line y = 0.3 → θ = π/2, s = +0.3
    a = np.array([[-0.8, 0.3]])
    b = np.array([[0.8, 0.3]])
    theta, s = endpoints_to_sinogram_coords(a, b)
    assert theta.shape == (1,)
    assert s.shape == (1,)
    assert theta[0] == pytest.approx(np.pi / 2, abs=1e-9)
    assert s[0] == pytest.approx(0.3, abs=1e-9)


def test_vertical_lor_folds_into_zero_pi():
    # Line x = 0.2 → θ = 0, s = +0.2 after folding
    a = np.array([[0.2, -0.8]])
    b = np.array([[0.2, 0.8]])
    theta, s = endpoints_to_sinogram_coords(a, b)
    assert theta[0] == pytest.approx(0.0, abs=1e-9)
    assert s[0] == pytest.approx(0.2, abs=1e-9)


def test_pair_order_symmetry():
    a = np.array([[0.5, 0.4], [-0.3, 0.6]])
    b = np.array([[-0.5, -0.2], [0.7, -0.1]])
    t1, s1 = endpoints_to_sinogram_coords(a, b)
    t2, s2 = endpoints_to_sinogram_coords(b, a)
    assert np.allclose(t1, t2, atol=1e-9)
    assert np.allclose(s1, s2, atol=1e-9)


def test_theta_in_zero_pi_and_matches_line_equation():
    rng = np.random.default_rng(0)
    # Random chords on a circle of radius 0.92
    angles = rng.uniform(0, 2 * np.pi, size=(200, 2))
    r = 0.92
    a = np.column_stack((r * np.cos(angles[:, 0]), r * np.sin(angles[:, 0])))
    b = np.column_stack((r * np.cos(angles[:, 1]), r * np.sin(angles[:, 1])))
    # Drop near-identical endpoints
    mask = np.linalg.norm(a - b, axis=1) > 0.05
    a, b = a[mask], b[mask]
    theta, s = endpoints_to_sinogram_coords(a, b)
    assert np.all(theta >= 0.0)
    assert np.all(theta < np.pi)
    # Midpoint should satisfy x cosθ + y sinθ ≈ s
    mid = 0.5 * (a + b)
    lhs = mid[:, 0] * np.cos(theta) + mid[:, 1] * np.sin(theta)
    assert np.allclose(lhs, s, atol=1e-8)


def test_analytic_point_source_coords_independent_of_lor_converter():
    # Ground-truth projection of a point — not derived from endpoints_to_*.
    thetas = np.linspace(0, np.pi, 37, endpoint=False)
    s = analytic_point_source_coords(0.2, -0.1, thetas)
    expected = 0.2 * np.cos(thetas) + (-0.1) * np.sin(thetas)
    assert np.allclose(s, expected)


def test_empty_endpoints():
    theta, s = endpoints_to_sinogram_coords(
        np.zeros((0, 2)), np.zeros((0, 2))
    )
    assert theta.size == 0 and s.size == 0
