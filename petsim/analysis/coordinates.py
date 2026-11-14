"""Pixel ↔ field-of-view coordinate conversions for reconstructed images.

Conventions match ``petsim.reconstruction.fbp.backproject`` and the
reconstruction panel display:

* Image array shape is ``(N, N)`` with row index ↔ y and column index ↔ x.
* Axis samples are ``np.linspace(-fov, fov, N)`` (pixel *centers*).
* ``imshow(..., origin="lower", extent=(-fov, fov, -fov, fov))`` matches this
  layout; do not swap x/y or flip axes when converting.
"""

from __future__ import annotations

import numpy as np


def image_axes(image_size: int, fov: float = 1.0) -> np.ndarray:
    """Return 1-D coordinate samples for pixel centers along one axis."""
    n = int(image_size)
    if n <= 0:
        return np.zeros(0, dtype=np.float64)
    return np.linspace(-float(fov), float(fov), n, dtype=np.float64)


def pixel_to_fov(
    row: float,
    col: float,
    *,
    image_size: int,
    fov: float = 1.0,
) -> tuple[float, float]:
    """Convert continuous pixel indices ``(row, col)`` → FOV ``(x, y)``.

    Non-integer indices are allowed (e.g. intensity-weighted centroids) and
    are linearly interpolated between pixel centers.
    """
    axis = image_axes(image_size, fov)
    if axis.size == 0:
        return 0.0, 0.0
    if axis.size == 1:
        return float(axis[0]), float(axis[0])
    x = float(np.interp(col, np.arange(axis.size, dtype=np.float64), axis))
    y = float(np.interp(row, np.arange(axis.size, dtype=np.float64), axis))
    return x, y


def fov_to_pixel(
    x: float,
    y: float,
    *,
    image_size: int,
    fov: float = 1.0,
) -> tuple[float, float]:
    """Convert FOV ``(x, y)`` → continuous pixel indices ``(row, col)``."""
    axis = image_axes(image_size, fov)
    if axis.size == 0:
        return 0.0, 0.0
    if axis.size == 1:
        return 0.0, 0.0
    # Invert linspace: axis[i] = -fov + i * (2*fov)/(N-1)
    span = 2.0 * float(fov)
    step = span / float(axis.size - 1)
    col = (float(x) + float(fov)) / step
    row = (float(y) + float(fov)) / step
    return float(row), float(col)
