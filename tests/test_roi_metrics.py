"""ROI construction and contrast / CV metric tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.roi import (
    build_tumor_background_rois,
    circular_mask,
    compute_contrast_metrics,
)


def test_circular_mask_pixel_count_roughly_pi_r2():
    n, fov, r = 201, 1.0, 0.2
    mask = circular_mask((n, n), (0.0, 0.0), r, fov=fov)
    pixel_area = (2.0 * fov / (n - 1)) ** 2
    area = mask.sum() * pixel_area
    assert abs(area - np.pi * r * r) / (np.pi * r * r) < 0.05


def test_background_excludes_tumor():
    masks = build_tumor_background_rois(
        (128, 128),
        tumor_xy=(0.15, 0.10),
        tumor_radius=0.10,
        brain_radius=0.55,
        fov=1.0,
        exclusion_factor=1.5,
        background_width=0.12,
    )
    assert masks.tumor_n_pixels > 0
    assert masks.background_n_pixels > 0
    assert not np.any(masks.tumor & masks.background)


def test_tbr_and_contrast_known_values():
    n = 128
    img = np.full((n, n), 2.0)
    masks = build_tumor_background_rois(
        (n, n),
        tumor_xy=(0.0, 0.0),
        tumor_radius=0.12,
        brain_radius=0.6,
        fov=1.0,
    )
    img[masks.tumor] = 8.0
    result = compute_contrast_metrics(img, masks)
    assert result.valid
    assert abs(result.tumor_mean - 8.0) < 1e-9
    assert abs(result.background_mean - 2.0) < 1e-9
    assert abs(result.tbr - 4.0) < 1e-9
    assert abs(result.contrast - 3.0) < 1e-9


def test_background_cv():
    n = 128
    rng = np.random.default_rng(0)
    img = np.zeros((n, n))
    masks = build_tumor_background_rois(
        (n, n),
        tumor_xy=(0.2, 0.0),
        tumor_radius=0.08,
        brain_radius=0.55,
        fov=1.0,
    )
    img[masks.tumor] = 5.0
    bg = 2.0 + 0.5 * rng.standard_normal(masks.background_n_pixels)
    bg = np.maximum(bg, 0.05)
    img[masks.background] = bg
    result = compute_contrast_metrics(img, masks)
    assert result.valid
    expected_cv = float(np.std(bg, ddof=0) / np.mean(bg))
    assert abs(result.background_cv - expected_cv) < 1e-9


def test_zero_background_mean_invalid():
    n = 64
    img = np.zeros((n, n))
    masks = build_tumor_background_rois(
        (n, n),
        tumor_xy=(0.0, 0.0),
        tumor_radius=0.1,
        brain_radius=0.5,
        fov=1.0,
    )
    img[masks.tumor] = 1.0
    # Background stays zero.
    result = compute_contrast_metrics(img, masks)
    assert not result.valid
    assert result.tbr is None
    assert result.contrast is None
    assert result.background_cv is None
    assert "background" in result.message.lower()


def test_nan_background_handled():
    n = 64
    img = np.ones((n, n))
    masks = build_tumor_background_rois(
        (n, n),
        tumor_xy=(0.0, 0.0),
        tumor_radius=0.1,
        brain_radius=0.5,
        fov=1.0,
    )
    img[masks.background] = np.nan
    result = compute_contrast_metrics(img, masks, use_nonnegative=False)
    assert not result.valid


def test_small_roi_warning():
    masks = build_tumor_background_rois(
        (32, 32),
        tumor_xy=(0.9, 0.9),  # near corner / outside brain
        tumor_radius=0.03,
        brain_radius=0.3,
        fov=1.0,
        min_pixels=4,
    )
    assert "too small" in masks.message or "empty" in masks.message
