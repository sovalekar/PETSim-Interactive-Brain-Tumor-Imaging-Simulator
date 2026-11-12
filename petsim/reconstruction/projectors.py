"""Parallel-beam forward projector and its adjoint backprojector.

The pair matches the Stage 3 sinogram geometry:

* image coordinates are pixel centers ``linspace(-fov, fov, N)``
* row index is y, column index is x (``indexing="ij"``)
* ``s = x cos θ + y sin θ``
* ``θ`` and ``s`` samples are sinogram bin centers

Each pixel is splatted into radial bins with linear interpolation, scaled by
the center-to-center spacing ``dx``. The backprojector gathers with the same
weights, so the operators are discrete adjoints:

``sum(forward(x) * y) == sum(x * backproject(y))``

This is an educational geometric model. It is not the filtered-backprojection
operator (no ramp filter and no ``π / Nθ`` scale) and it is not a dense system
matrix.
"""

from __future__ import annotations

import numpy as np


def pixel_spacing(image_size: int, fov: float) -> float:
    """Center-to-center spacing of the reconstruction grid."""
    n = int(image_size)
    if n <= 1:
        return float(2.0 * fov)
    return float((2.0 * fov) / (n - 1))


def image_coordinates(image_size: int, fov: float) -> tuple[np.ndarray, np.ndarray]:
    """Return ``(xx, yy)`` grids of pixel-center coordinates."""
    n = int(image_size)
    axis = np.linspace(-float(fov), float(fov), n, dtype=np.float64)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")
    return xx, yy


def _interpolation_index(
    s_vals: np.ndarray,
    s_centers: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map radial coordinates to bin indices and linear-interpolation fractions.

    Samples outside ``[s_centers[0], s_centers[-1]]`` are marked invalid and
    contribute nothing (same support as ``np.interp(..., left=0, right=0)``).
    """
    s_centers = np.asarray(s_centers, dtype=np.float64)
    n_s = int(s_centers.size)
    if n_s == 0:
        shape = np.asarray(s_vals).shape
        empty = np.zeros(shape, dtype=np.int64)
        return empty, np.zeros(shape, dtype=np.float64), np.zeros(shape, dtype=bool)
    if n_s == 1:
        valid = np.isclose(np.asarray(s_vals, dtype=np.float64), s_centers[0])
        return (
            np.zeros(valid.shape, dtype=np.int64),
            np.zeros(valid.shape, dtype=np.float64),
            valid,
        )

    ds = float(s_centers[1] - s_centers[0])
    t = (np.asarray(s_vals, dtype=np.float64) - float(s_centers[0])) / ds
    valid = (t >= 0.0) & (t <= float(n_s - 1))
    t_clip = np.clip(t, 0.0, float(n_s - 1))
    i0 = np.floor(t_clip).astype(np.int64)
    # The last center has fraction 0 and no right neighbor.
    at_end = i0 >= (n_s - 1)
    i0 = np.clip(i0, 0, n_s - 1)
    frac = t_clip - np.floor(t_clip)
    frac = np.where(at_end, 0.0, frac)
    return i0, frac.astype(np.float64), valid


def forward_project(
    image: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    *,
    fov: float = 1.0,
) -> np.ndarray:
    """Pixel-driven forward projection. Returns ``(n_theta, n_s)``."""
    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1]:
        raise ValueError("forward_project expects a square image")
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    s_centers = np.asarray(s_centers, dtype=np.float64).ravel()
    n = int(img.shape[0])
    n_theta = int(thetas.size)
    n_s = int(s_centers.size)
    sino = np.zeros((n_theta, n_s), dtype=np.float64)
    if n == 0 or n_theta == 0 or n_s == 0 or not np.any(img):
        return sino

    dx = pixel_spacing(n, fov)
    xx, yy = image_coordinates(n, fov)
    values = img.ravel() * dx
    for i, theta in enumerate(thetas):
        s_vals = (xx * np.cos(theta) + yy * np.sin(theta)).ravel()
        i0, frac, valid = _interpolation_index(s_vals, s_centers)
        if not np.any(valid):
            continue
        i0v = i0[valid]
        fracv = frac[valid]
        wv = values[valid]
        row = sino[i]
        np.add.at(row, i0v, wv * (1.0 - fracv))
        right = i0v + 1
        has_right = right < n_s
        if np.any(has_right):
            np.add.at(row, right[has_right], wv[has_right] * fracv[has_right])
    return sino


def backproject(
    sinogram: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    *,
    image_size: int,
    fov: float = 1.0,
) -> np.ndarray:
    """Adjoint of :func:`forward_project` (linear interpolation, same ``dx``)."""
    sino = np.asarray(sinogram, dtype=np.float64)
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    s_centers = np.asarray(s_centers, dtype=np.float64).ravel()
    n = int(image_size)
    image = np.zeros((n, n), dtype=np.float64)
    if n == 0 or sino.size == 0 or thetas.size == 0 or s_centers.size == 0:
        return image

    dx = pixel_spacing(n, fov)
    xx, yy = image_coordinates(n, fov)
    n_theta = min(int(thetas.size), int(sino.shape[0]))
    for i in range(n_theta):
        s_vals = xx * np.cos(thetas[i]) + yy * np.sin(thetas[i])
        i0, frac, valid = _interpolation_index(s_vals, s_centers)
        gathered = np.zeros_like(s_vals)
        if np.any(valid):
            i0v = i0[valid]
            left = sino[i, i0v]
            right_idx = np.clip(i0v + 1, 0, s_centers.size - 1)
            right = sino[i, right_idx]
            # Fraction is 0 on the last bin, so the clipped right sample is unused.
            gathered[valid] = (1.0 - frac[valid]) * left + frac[valid] * right
        image += gathered * dx
    return image


def sensitivity_map(
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    *,
    image_size: int,
    fov: float = 1.0,
) -> np.ndarray:
    """``Σ_i A_ij``, the backprojection of a sinogram of ones."""
    n_theta = int(np.asarray(theta_centers).size)
    n_s = int(np.asarray(s_centers).size)
    ones = np.ones((n_theta, n_s), dtype=np.float64)
    return backproject(
        ones,
        theta_centers,
        s_centers,
        image_size=image_size,
        fov=fov,
    )
