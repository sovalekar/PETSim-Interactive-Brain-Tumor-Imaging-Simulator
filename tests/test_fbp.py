"""Filtered backprojection unit tests (including analytic projection fixture)."""

from __future__ import annotations

import numpy as np
import pytest

from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints
from petsim.reconstruction.sinogram import Sinogram, analytic_disk_sinogram, build_sinogram


def test_empty_sinogram_returns_zero_image():
    sino = build_sinogram(np.zeros((0, 2)), np.zeros((0, 2)), n_theta=180, n_s=128)
    img = filtered_backprojection(sino, filter_name="ramp", image_size=128, fov=1.0)
    assert img.shape == (128, 128)
    assert np.all(img == 0)


def test_fbp_output_dimensions_and_finite():
    sino = analytic_disk_sinogram(radius=0.25, n_theta=180, n_s=128, s_max=1.0)
    for filt in ("ramp", "hann"):
        img = filtered_backprojection(sino, filter_name=filt, image_size=128, fov=1.0)
        assert img.shape == (128, 128)
        assert np.all(np.isfinite(img))
        assert img.max() > 0


def test_analytic_disk_recovers_centered_activity():
    """Independent of PET LOR conversion: analytic Radon disk → FBP blob at center."""
    sino = analytic_disk_sinogram(
        radius=0.22, n_theta=180, n_s=128, s_max=1.0, amplitude=1.0
    )
    img = filtered_backprojection(sino, filter_name="ramp", image_size=128, fov=1.0)
    axis = np.linspace(-1.0, 1.0, 128)
    # Uniform disks reconstruct nearly flat inside; use center-of-mass of the
    # non-negative image instead of argmax (argmax can wander on flat plateaus).
    pos = np.maximum(img, 0.0)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    mass = float(pos.sum())
    assert mass > 0.0
    cx = float((pos * xx).sum() / mass)
    cy = float((pos * yy).sum() / mass)
    assert abs(cx) < 0.05
    assert abs(cy) < 0.05
    # Center brighter than periphery
    center = img[64 - 5 : 64 + 5, 64 - 5 : 64 + 5].mean()
    corner = img[0:10, 0:10].mean()
    assert center > corner * 5.0


def test_point_like_sinogram_peaks_at_offset():
    """Synthetic sinogram from analytic point projections (not LOR converter)."""
    n_theta, n_s = 180, 128
    s_max = 1.0
    x0, y0 = 0.25, -0.15
    theta_edges = np.linspace(0, np.pi, n_theta + 1)
    s_edges = np.linspace(-s_max, s_max, n_s + 1)
    theta_c = 0.5 * (theta_edges[:-1] + theta_edges[1:])
    s_c = 0.5 * (s_edges[:-1] + s_edges[1:])
    data = np.zeros((n_theta, n_s), dtype=np.float64)
    for i, th in enumerate(theta_c):
        s_true = x0 * np.cos(th) + y0 * np.sin(th)
        j = int(np.argmin(np.abs(s_c - s_true)))
        data[i, j] = 1.0
    sino = Sinogram(data=data, theta_edges=theta_edges, s_edges=s_edges, n_events=n_theta)
    img = filtered_backprojection(sino, filter_name="hann", image_size=128, fov=1.0)
    peak = np.unravel_index(np.argmax(img), img.shape)
    axis = np.linspace(-1.0, 1.0, 128)
    assert abs(axis[peak[1]] - x0) < 0.12
    assert abs(axis[peak[0]] - y0) < 0.12


def test_pipeline_empty_message():
    result = reconstruct_from_endpoints(
        np.zeros((0, 2)), np.zeros((0, 2)), ReconstructionConfig()
    )
    assert result.n_coincidences == 0
    assert result.image.shape == (128, 128)
    assert "No coincidence" in result.message


def test_display_clip_does_not_mutate_raw():
    sino = analytic_disk_sinogram(radius=0.2, n_theta=90, n_s=64, s_max=1.0)
    img = filtered_backprojection(sino, filter_name="ramp", image_size=64, fov=1.0)
    # Force a negative into a copy via pipeline result semantics
    from petsim.reconstruction.pipeline import ReconstructionResult

    raw = img.copy()
    raw[0, 0] = -1.0
    result = ReconstructionResult(
        sinogram=sino, image=raw, config=ReconstructionConfig(image_size=64), n_coincidences=10
    )
    disp = result.image_display
    assert result.image[0, 0] == -1.0
    assert disp[0, 0] == 0.0
