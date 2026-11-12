"""Tumor localization from reconstructed PET images (educational).

Method
------
**Constrained intensity-weighted centroid** on the nonnegative reconstruction:

1. Optionally restrict the search to the circular brain disk (phantom geometry
   radius only — *not* the tumor center).
2. Find the brightest pixel inside that search region (coarse peak).
3. Form a local neighborhood disk around that peak.
4. Compute the intensity-weighted centroid of neighborhood pixels above a
   fraction of the local peak intensity.
5. Convert the centroid to FOV coordinates.

The estimated center is derived **only** from reconstructed image intensities.
Ground-truth tumor ``(x, y)`` is used solely when computing localization error
for evaluation — never to place the estimate.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.analysis.coordinates import image_axes, pixel_to_fov


@dataclass(frozen=True)
class LocalizationResult:
    """Outcome of one tumor-localization attempt."""

    estimated_xy: tuple[float, float] | None
    true_xy: tuple[float, float] | None
    error: float | None
    method: str
    reliable: bool
    message: str

    @property
    def estimated_x(self) -> float | None:
        return None if self.estimated_xy is None else self.estimated_xy[0]

    @property
    def estimated_y(self) -> float | None:
        return None if self.estimated_xy is None else self.estimated_xy[1]


def localization_error(
    estimated_xy: tuple[float, float],
    true_xy: tuple[float, float],
) -> float:
    """Euclidean localization error in FOV units."""
    dx = float(estimated_xy[0]) - float(true_xy[0])
    dy = float(estimated_xy[1]) - float(true_xy[1])
    return float(np.sqrt(dx * dx + dy * dy))


def localize_tumor(
    image: np.ndarray,
    *,
    fov: float = 1.0,
    brain_radius: float | None = None,
    neighborhood_radius: float = 0.18,
    intensity_fraction: float = 0.35,
    true_xy: tuple[float, float] | None = None,
    tumor_enabled: bool = True,
    use_nonnegative: bool = True,
) -> LocalizationResult:
    """Estimate tumor location from a reconstructed image.

    Parameters
    ----------
    image :
        Raw or display reconstruction array ``(N, N)``. Analysis uses a
        nonnegative copy when ``use_nonnegative`` is True (documented default:
        display-clipped intensities).
    brain_radius :
        If set, search is limited to the disk ``x²+y² ≤ brain_radius²``.
        This uses phantom *geometry*, not tumor ground truth.
    neighborhood_radius :
        FOV radius of the centroid neighborhood around the coarse peak.
    intensity_fraction :
        Keep neighborhood pixels with intensity ≥ fraction × local peak.
    true_xy :
        Optional ground truth for error reporting only.
    tumor_enabled :
        If False, skip localization and return an explanatory message.
    """
    method = (
        "constrained intensity-weighted centroid "
        "(nonnegative image; brain-disk search; local neighborhood)"
    )

    if not tumor_enabled:
        return LocalizationResult(
            estimated_xy=None,
            true_xy=true_xy,
            error=None,
            method=method,
            reliable=False,
            message="Tumor disabled — localization not applicable.",
        )

    img = np.asarray(image, dtype=np.float64)
    if img.ndim != 2 or img.shape[0] != img.shape[1] or img.size == 0:
        return LocalizationResult(
            estimated_xy=None,
            true_xy=true_xy,
            error=None,
            method=method,
            reliable=False,
            message="Invalid reconstructed image shape for localization.",
        )

    work = np.maximum(img, 0.0) if use_nonnegative else img.copy()
    n = int(work.shape[0])
    axis = image_axes(n, fov)
    yy, xx = np.meshgrid(axis, axis, indexing="ij")

    search = np.ones((n, n), dtype=bool)
    if brain_radius is not None and brain_radius > 0:
        search &= (xx * xx + yy * yy) <= (float(brain_radius) ** 2)

    if not np.any(search) or not np.any(work[search] > 0):
        return LocalizationResult(
            estimated_xy=None,
            true_xy=true_xy,
            error=None,
            method=method,
            reliable=False,
            message="No positive intensity in the localization search region.",
        )

    masked = np.where(search, work, -np.inf)
    peak_flat = int(np.argmax(masked))
    peak_row, peak_col = divmod(peak_flat, n)
    peak_x = float(xx[peak_row, peak_col])
    peak_y = float(yy[peak_row, peak_col])
    peak_val = float(work[peak_row, peak_col])
    if not np.isfinite(peak_val) or peak_val <= 0:
        return LocalizationResult(
            estimated_xy=None,
            true_xy=true_xy,
            error=None,
            method=method,
            reliable=False,
            message="Localization peak is non-positive or non-finite.",
        )

    neigh_r = max(float(neighborhood_radius), 1e-6)
    neighborhood = (xx - peak_x) ** 2 + (yy - peak_y) ** 2 <= neigh_r**2
    neighborhood &= search
    thresh = float(intensity_fraction) * peak_val
    weights_mask = neighborhood & (work >= thresh)
    if not np.any(weights_mask):
        # Fall back to the coarse peak itself.
        est_xy = (peak_x, peak_y)
    else:
        w = work[weights_mask]
        w_sum = float(np.sum(w))
        if w_sum <= 0 or not np.isfinite(w_sum):
            est_xy = (peak_x, peak_y)
        else:
            # Continuous centroid in pixel space, then convert to FOV.
            rows, cols = np.nonzero(weights_mask)
            row_c = float(np.sum(rows.astype(np.float64) * w) / w_sum)
            col_c = float(np.sum(cols.astype(np.float64) * w) / w_sum)
            est_xy = pixel_to_fov(row_c, col_c, image_size=n, fov=fov)

    err = None
    if true_xy is not None:
        err = localization_error(est_xy, true_xy)

    return LocalizationResult(
        estimated_xy=est_xy,
        true_xy=true_xy,
        error=err,
        method=method,
        reliable=True,
        message="Localization estimate derived from reconstructed image intensities.",
    )
