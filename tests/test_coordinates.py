"""Pixel ↔ FOV coordinate conversion tests."""

from __future__ import annotations

import numpy as np

from petsim.analysis.coordinates import fov_to_pixel, image_axes, pixel_to_fov


def test_image_axes_endpoints():
    axis = image_axes(5, fov=1.0)
    assert axis.shape == (5,)
    assert axis[0] == -1.0
    assert axis[-1] == 1.0


def test_pixel_center_roundtrip():
    n, fov = 128, 1.0
    for row, col in [(0, 0), (64, 64), (127, 0), (0, 127), (31.5, 90.25)]:
        x, y = pixel_to_fov(row, col, image_size=n, fov=fov)
        row2, col2 = fov_to_pixel(x, y, image_size=n, fov=fov)
        assert abs(row2 - row) < 1e-9
        assert abs(col2 - col) < 1e-9


def test_known_corner_mapping():
    x, y = pixel_to_fov(0, 0, image_size=3, fov=1.0)
    assert x == -1.0 and y == -1.0
    x, y = pixel_to_fov(2, 2, image_size=3, fov=1.0)
    assert x == 1.0 and y == 1.0


def test_no_xy_swap():
    # Column increases → x increases; row increases → y increases.
    x0, y0 = pixel_to_fov(10, 20, image_size=64, fov=1.0)
    x1, y1 = pixel_to_fov(10, 40, image_size=64, fov=1.0)
    x2, y2 = pixel_to_fov(40, 20, image_size=64, fov=1.0)
    assert x1 > x0
    assert abs(y1 - y0) < 1e-12
    assert y2 > y0
    assert abs(x2 - x0) < 1e-12


def test_matches_fbp_meshgrid_centers():
    n, fov = 32, 1.0
    axis = image_axes(n, fov)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    # Pixel (r,c) must map to (xx[r,c], yy[r,c]).
    for r, c in [(0, 0), (5, 17), (31, 31)]:
        x, y = pixel_to_fov(r, c, image_size=n, fov=fov)
        assert abs(x - xx[r, c]) < 1e-12
        assert abs(y - yy[r, c]) < 1e-12
