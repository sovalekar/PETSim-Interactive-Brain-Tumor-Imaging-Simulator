"""Sinogram binning and convention tests."""

from __future__ import annotations

import numpy as np
import pytest

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.lor import endpoints_to_sinogram_coords
from petsim.reconstruction.sinogram import analytic_disk_sinogram, build_sinogram


def test_empty_acquisition_sinogram_shape():
    sino = build_sinogram(
        np.zeros((0, 2)), np.zeros((0, 2)), n_theta=180, n_s=128, s_max=0.92
    )
    assert sino.data.shape == (180, 128)
    assert sino.data.sum() == 0
    assert sino.n_events == 0


def test_sinogram_bins_match_digitized_coords():
    a = np.array([[0.9, 0.0], [0.0, 0.9], [-0.9, 0.0]])
    b = np.array([[-0.9, 0.0], [0.0, -0.9], [0.9, 0.0]])
    sino = build_sinogram(a, b, n_theta=36, n_s=32, s_max=1.0)
    assert sino.data.sum() == pytest.approx(3.0)
    assert sino.theta_centers[0] == pytest.approx(np.pi / 72)


def test_angular_radial_edges_conventions():
    sino = build_sinogram(np.zeros((0, 2)), np.zeros((0, 2)), n_theta=180, n_s=128)
    assert sino.theta_edges[0] == pytest.approx(0.0)
    assert sino.theta_edges[-1] == pytest.approx(np.pi)
    assert sino.s_edges[0] < 0 < sino.s_edges[-1]
    assert len(sino.theta_centers) == 180
    assert len(sino.s_centers) == 128


def test_deterministic_sinogram_with_fixed_seed():
    eng = PETEngine(BrainPhantom(), DetectorRing(), seed=123)
    events = eng.simulate_batch(3000, detector_efficiency=1.0)
    hits_a = np.array([e.hit_a_xy for e in events if e.is_coincidence])
    hits_b = np.array([e.hit_b_xy for e in events if e.is_coincidence])
    s1 = build_sinogram(hits_a, hits_b, n_theta=90, n_s=64, s_max=0.92)
    s2 = build_sinogram(hits_a, hits_b, n_theta=90, n_s=64, s_max=0.92)
    assert np.array_equal(s1.data, s2.data)

    eng2 = PETEngine(BrainPhantom(), DetectorRing(), seed=123)
    events2 = eng2.simulate_batch(3000, detector_efficiency=1.0)
    ha2 = np.array([e.hit_a_xy for e in events2 if e.is_coincidence])
    hb2 = np.array([e.hit_b_xy for e in events2 if e.is_coincidence])
    s3 = build_sinogram(ha2, hb2, n_theta=90, n_s=64, s_max=0.92)
    assert np.array_equal(s1.data, s3.data)


def test_redundant_pair_order_same_bins():
    a = np.array([[0.7, 0.2], [-0.4, 0.5]])
    b = np.array([[-0.6, -0.3], [0.5, -0.55]])
    s_ab = build_sinogram(a, b, n_theta=60, n_s=40, s_max=1.0)
    s_ba = build_sinogram(b, a, n_theta=60, n_s=40, s_max=1.0)
    assert np.array_equal(s_ab.data, s_ba.data)


def test_analytic_disk_sinogram_independent_fixture():
    sino = analytic_disk_sinogram(radius=0.3, n_theta=90, n_s=64, s_max=1.0)
    # Independent of θ: every row identical
    assert np.allclose(sino.data, sino.data[0])
    assert sino.data[0].max() > 0
    # Outside radius should be zero
    s = sino.s_centers
    assert np.all(sino.data[0, np.abs(s) > 0.3] == 0)


def test_coords_then_bins_consistent():
    a = np.array([[0.92, 0.0], [0.0, 0.92]])
    b = np.array([[-0.92, 0.0], [0.0, -0.92]])
    theta, s = endpoints_to_sinogram_coords(a, b)
    sino = build_sinogram(a, b, n_theta=180, n_s=128, s_max=0.92)
    # Central LORs (through origin) → s ≈ 0 → middle radial bins
    mid = 64
    assert sino.data[:, mid - 2 : mid + 2].sum() >= 1.0
    assert np.all(np.abs(s) < 1e-8)
