"""Tumor localization unit tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.coordinates import fov_to_pixel
from petsim.analysis.localization import localization_error, localize_tumor


def _blob_image(
    center_xy: tuple[float, float],
    *,
    n: int = 128,
    fov: float = 1.0,
    sigma: float = 0.05,
    amplitude: float = 1.0,
    background: float = 0.05,
) -> np.ndarray:
    axis = np.linspace(-fov, fov, n)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    cx, cy = center_xy
    img = background + amplitude * np.exp(
        -((xx - cx) ** 2 + (yy - cy) ** 2) / (2.0 * sigma**2)
    )
    return img.astype(np.float64)


def test_localization_error_formula():
    err = localization_error((0.2, -0.1), (0.0, 0.0))
    assert abs(err - np.sqrt(0.04 + 0.01)) < 1e-12


def test_localize_known_blob_center():
    true = (0.22, -0.15)
    img = _blob_image(true)
    result = localize_tumor(
        img,
        fov=1.0,
        brain_radius=0.7,
        neighborhood_radius=0.2,
        true_xy=true,
        tumor_enabled=True,
    )
    assert result.reliable
    assert result.estimated_xy is not None
    assert result.error is not None
    assert result.error < 0.03
    assert abs(result.estimated_xy[0] - true[0]) < 0.03
    assert abs(result.estimated_xy[1] - true[1]) < 0.03


def test_localize_does_not_use_true_xy_for_estimate():
    # Bright blob at A; true_xy claims B — estimate must follow the image.
    blob = (0.30, 0.10)
    claimed_true = (-0.30, -0.20)
    img = _blob_image(blob)
    result = localize_tumor(
        img,
        fov=1.0,
        brain_radius=0.75,
        true_xy=claimed_true,
        tumor_enabled=True,
    )
    assert result.reliable
    assert result.estimated_xy is not None
    assert localization_error(result.estimated_xy, blob) < 0.04
    assert localization_error(result.estimated_xy, claimed_true) > 0.2


def test_disabled_tumor():
    img = _blob_image((0.1, 0.1))
    result = localize_tumor(img, tumor_enabled=False, true_xy=(0.1, 0.1))
    assert not result.reliable
    assert result.estimated_xy is None
    assert result.error is None
    assert "disabled" in result.message.lower()


def test_empty_positive_region():
    img = np.zeros((64, 64))
    result = localize_tumor(img, brain_radius=0.5, tumor_enabled=True)
    assert not result.reliable
    assert result.estimated_xy is None


def test_peak_outside_brain_ignored():
    n, fov = 96, 1.0
    img = np.full((n, n), 0.02)
    # Bright spike far outside a small brain disk.
    row, col = fov_to_pixel(0.8, 0.8, image_size=n, fov=fov)
    img[int(round(row)), int(round(col))] = 10.0
    # Soft blob inside brain.
    inside = _blob_image((0.1, -0.05), n=n, fov=fov, amplitude=1.0, background=0.0)
    img = np.maximum(img, inside * 0.5 + 0.05)
    result = localize_tumor(
        img,
        fov=fov,
        brain_radius=0.4,
        true_xy=(0.1, -0.05),
        tumor_enabled=True,
    )
    assert result.reliable
    assert result.estimated_xy is not None
    assert abs(result.estimated_xy[0]) < 0.35
    assert abs(result.estimated_xy[1]) < 0.35
