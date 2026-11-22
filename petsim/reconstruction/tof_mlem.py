"""TOF-MLEM on the parallel-beam projector.

The measurement space is every bin ``(θ, s, t)``, including bins that recorded
zero. The system matrix is

    H = a(θ, s) · F_tof

``F_tof`` splats each pixel with the geometric ray weight times the discrete
TOF kernel. ``a`` is the Stage 11 chord survival and does not depend on ``t``
for this convex ellipse. It multiplies the forward model once and the
adjoint once. Counts are not pre-divided by ``a``.

Sensitivity is ``Hᵀ 1``, the backprojection of ``a`` over the whole time
axis. It is not the sum of the kernels of the events that happened to be
detected. Empty bins still contribute to the chance of detection.

Random coincidences are not histogrammed. A flat arrival time is not treated
as the TOF response of a true pair, and this version does not add a random
background term. The caller passes true coincidences only.

With TOF disabled, callers must use :func:`mlem_reconstruct` instead of this
module so the Stage 11 update is unchanged.

The tumor center is not an input.
"""

from __future__ import annotations

import numpy as np

from petsim.physics.tof import (
    TOFConfig,
    along_line_coordinate,
    discrete_tof_kernel,
    most_likely_points,
    spatial_sigma_fov,
    time_bin_centers,
)
from petsim.reconstruction.lor import endpoints_to_sinogram_coords
from petsim.reconstruction.mlem import MLEMHistory, poisson_log_likelihood
from petsim.reconstruction.projectors import (
    _interpolation_index,
    image_coordinates,
    pixel_spacing,
)


def histogram_true_events(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    delta_t_ps: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    t_centers: np.ndarray,
    config: TOFConfig,
) -> np.ndarray:
    """Bin true-coincidence times. Shape ``(n_θ, n_s, n_t)``.

    Each event contributes its TOF kernel at the measured along-line position.
    The kernel is the same discrete Gaussian the projector uses.
    """
    cfg = config.clamp()
    theta, s = endpoints_to_sinogram_coords(hits_a, hits_b)
    points = most_likely_points(hits_a, hits_b, delta_t_ps, cfg.cm_per_fov)
    t_meas = along_line_coordinate(points, hits_a, hits_b)
    n_theta = int(np.asarray(theta_centers).size)
    n_s = int(np.asarray(s_centers).size)
    n_t = int(np.asarray(t_centers).size)
    sino = np.zeros((n_theta, n_s, n_t), dtype=np.float64)
    if theta.size == 0:
        return sino
    sigma = spatial_sigma_fov(cfg.fwhm_ps, cfg.cm_per_fov)
    kernel = discrete_tof_kernel(t_meas, t_centers, sigma)
    # Nearest angle. The projector samples the same angle centers.
    edges = np.linspace(0.0, np.pi, n_theta + 1, dtype=np.float64)
    angle_index = np.clip(np.digitize(theta, edges) - 1, 0, n_theta - 1)
    i0, frac, valid = _interpolation_index(s, s_centers)
    for event in range(theta.size):
        if not valid[event]:
            continue
        row = sino[int(angle_index[event])]
        left = int(i0[event])
        weight_left = float(1.0 - frac[event])
        row[left] += weight_left * kernel[event]
        right = left + 1
        if right < n_s and frac[event] > 0.0:
            row[right] += float(frac[event]) * kernel[event]
    return sino


def _survival_array(survival: np.ndarray | None, n_theta: int, n_s: int) -> np.ndarray | None:
    if survival is None:
        return None
    weights = np.asarray(survival, dtype=np.float64)
    if weights.shape != (n_theta, n_s):
        raise ValueError("TOF attenuation survival must have shape (n_theta, n_s)")
    return weights


def forward_project_tof(
    image: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    t_centers: np.ndarray,
    sigma_fov: float,
    *,
    fov: float = 1.0,
    survival: np.ndarray | None = None,
) -> np.ndarray:
    """``a · F_tof x``. Attenuation multiplies the finished geometric projection."""
    img = np.asarray(image, dtype=np.float64)
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    radii = np.asarray(s_centers, dtype=np.float64).ravel()
    times = np.asarray(t_centers, dtype=np.float64).ravel()
    n = int(img.shape[0])
    n_theta = int(thetas.size)
    n_s = int(radii.size)
    n_t = int(times.size)
    sino = np.zeros((n_theta, n_s, n_t), dtype=np.float64)
    if n == 0 or n_theta == 0 or n_s == 0 or n_t == 0 or not np.any(img):
        return sino
    weights = _survival_array(survival, n_theta, n_s)
    dx = pixel_spacing(n, fov)
    xx, yy = image_coordinates(n, fov)
    values = (img * dx).ravel()
    for i, theta in enumerate(thetas):
        s_vals = (xx * np.cos(theta) + yy * np.sin(theta)).ravel()
        t_vals = (-xx * np.sin(theta) + yy * np.cos(theta)).ravel()
        i0, frac, valid = _interpolation_index(s_vals, radii)
        if not np.any(valid):
            continue
        kernel = discrete_tof_kernel(t_vals[valid], times, sigma_fov)
        i0v = i0[valid]
        fracv = frac[valid]
        wv = values[valid]
        slab = sino[i]
        left = wv * (1.0 - fracv)
        right = i0v + 1
        has_right = right < n_s
        for column in range(n_t):
            np.add.at(slab[:, column], i0v, left * kernel[:, column])
            if np.any(has_right):
                np.add.at(
                    slab[:, column],
                    right[has_right],
                    wv[has_right] * fracv[has_right] * kernel[has_right, column],
                )
    if weights is not None:
        sino *= weights[:, :, None]
    return sino


def backproject_tof(
    sinogram: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    t_centers: np.ndarray,
    sigma_fov: float,
    *,
    image_size: int,
    fov: float = 1.0,
    survival: np.ndarray | None = None,
) -> np.ndarray:
    """Adjoint of :func:`forward_project_tof`. ``a`` multiplies the input once."""
    sino = np.asarray(sinogram, dtype=np.float64)
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    radii = np.asarray(s_centers, dtype=np.float64).ravel()
    times = np.asarray(t_centers, dtype=np.float64).ravel()
    n = int(image_size)
    image = np.zeros((n, n), dtype=np.float64)
    if n == 0 or sino.size == 0 or thetas.size == 0:
        return image
    n_theta = min(int(thetas.size), int(sino.shape[0]))
    n_s = int(radii.size)
    weights = _survival_array(survival, n_theta, n_s)
    data = sino[:n_theta]
    if weights is not None:
        data = data * weights[:, :, None]
    dx = pixel_spacing(n, fov)
    xx, yy = image_coordinates(n, fov)
    for i in range(n_theta):
        s_vals = xx * np.cos(thetas[i]) + yy * np.sin(thetas[i])
        t_vals = (-xx * np.sin(thetas[i]) + yy * np.cos(thetas[i])).ravel()
        i0, frac, valid = _interpolation_index(s_vals.ravel(), radii)
        kernel = discrete_tof_kernel(t_vals, times, sigma_fov)
        gathered = np.zeros(t_vals.shape[0], dtype=np.float64)
        if np.any(valid):
            i0v = i0[valid]
            fracv = frac[valid]
            # Sum over time of the interpolated sinogram times the kernel.
            left = data[i, i0v, :]
            right_idx = np.clip(i0v + 1, 0, n_s - 1)
            right = data[i, right_idx, :]
            mixed = (1.0 - fracv)[:, None] * left + fracv[:, None] * right
            gathered[valid] = np.sum(mixed * kernel[valid], axis=1)
        image += gathered.reshape(n, n) * dx
    return image


def sensitivity_tof(
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    t_centers: np.ndarray,
    sigma_fov: float,
    *,
    image_size: int,
    fov: float = 1.0,
    survival: np.ndarray | None = None,
) -> np.ndarray:
    """``Hᵀ 1`` over every ``(θ, s, t)`` bin, not only bins that contain counts."""
    n_theta = int(np.asarray(theta_centers).size)
    n_s = int(np.asarray(s_centers).size)
    n_t = int(np.asarray(t_centers).size)
    ones = np.ones((n_theta, n_s, n_t), dtype=np.float64)
    return backproject_tof(
        ones,
        theta_centers,
        s_centers,
        t_centers,
        sigma_fov,
        image_size=image_size,
        fov=fov,
        survival=survival,
    )


def tof_mlem_reconstruct(
    y: np.ndarray,
    theta_centers: np.ndarray,
    s_centers: np.ndarray,
    t_centers: np.ndarray,
    config: TOFConfig,
    *,
    n_iterations: int = 10,
    image_size: int = 128,
    fov: float = 1.0,
    survival: np.ndarray | None = None,
    progress=None,
    cancel=None,
) -> tuple[np.ndarray, MLEMHistory, bool]:
    """TOF-MLEM. ``y`` is the full time-binned histogram, zeros included."""
    cfg = config.clamp()
    n_iter = max(int(n_iterations), 1)
    n = int(image_size)
    measured = np.asarray(y, dtype=np.float64)
    thetas = np.asarray(theta_centers, dtype=np.float64).ravel()
    radii = np.asarray(s_centers, dtype=np.float64).ravel()
    times = np.asarray(t_centers, dtype=np.float64).ravel()
    sigma = spatial_sigma_fov(cfg.fwhm_ps, cfg.cm_per_fov)
    if n <= 0 or measured.size == 0:
        empty = MLEMHistory(
            iterations=np.zeros(0, dtype=np.int64),
            log_likelihood=np.zeros(0, dtype=np.float64),
            relative_change=np.zeros(0, dtype=np.float64),
        )
        return np.zeros((max(n, 0), max(n, 0)), dtype=np.float64), empty, False

    sens = sensitivity_tof(
        thetas, radii, times, sigma, image_size=n, fov=fov, survival=survival
    )
    support = sens > 0.0
    image = np.zeros((n, n), dtype=np.float64)
    image[support] = 1.0
    iterations: list[int] = []
    likelihoods: list[float] = []
    changes: list[float] = []
    cancelled = False
    for k in range(1, n_iter + 1):
        if cancel is not None and cancel():
            cancelled = True
            break
        previous = image
        y_hat = forward_project_tof(
            previous, thetas, radii, times, sigma, fov=fov, survival=survival
        )
        ratio = np.zeros_like(measured)
        predictable = y_hat > 0.0
        ratio[predictable] = measured[predictable] / y_hat[predictable]
        correction = backproject_tof(
            ratio, thetas, radii, times, sigma, image_size=n, fov=fov, survival=survival
        )
        updated = np.zeros_like(previous)
        updated[support] = previous[support] * (correction[support] / sens[support])
        np.maximum(updated, 0.0, out=updated)
        denom = float(np.linalg.norm(previous))
        rel = float(np.linalg.norm(updated - previous) / denom) if denom > 0.0 else 0.0
        image = updated
        y_hat_updated = forward_project_tof(
            image, thetas, radii, times, sigma, fov=fov, survival=survival
        )
        ll = poisson_log_likelihood(measured, y_hat_updated)
        iterations.append(k)
        likelihoods.append(ll)
        changes.append(rel)
        if progress is not None:
            progress(k, n_iter, ll, rel)
    history = MLEMHistory(
        iterations=np.asarray(iterations, dtype=np.int64),
        log_likelihood=np.asarray(likelihoods, dtype=np.float64),
        relative_change=np.asarray(changes, dtype=np.float64),
    )
    return image, history, cancelled
