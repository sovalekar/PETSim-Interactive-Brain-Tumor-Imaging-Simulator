"""Region-of-interest contrast and background noise metrics (educational).

Intensity convention
--------------------
By default, metrics use **nonnegative (display-clipped) reconstructed
intensities** ``max(image, 0)``. Raw FBP negatives from filter ringing are
excluded so mean background is not artificially depressed. Values are
*relative reconstruction units*, not calibrated clinical activity
concentrations (Bq/mL or SUV).

ROI definitions
---------------
* **Tumor ROI**: circular disk centered at the *documented ground-truth*
  tumor center with radius ``tumor_radius`` (phantom setting). Ground truth
  defines the ROI geometry for evaluation; intensities still come from the
  reconstructed image.
* **Background ROI**: annulus centered on the same tumor center with
  ``inner = tumor_radius * exclusion_factor`` and
  ``outer = inner + background_width``, further restricted to the brain disk.
  The exclusion margin prevents tumor spillover from contaminating background.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.analysis.coordinates import image_axes


@dataclass(frozen=True)
class ROIMasks:
    """Boolean masks aligned with a reconstructed image."""

    tumor: np.ndarray
    background: np.ndarray
    tumor_n_pixels: int
    background_n_pixels: int
    message: str = ""


@dataclass(frozen=True)
class ContrastResult:
    """Tumor / background quantitative metrics."""

    tumor_mean: float | None
    background_mean: float | None
    tbr: float | None
    contrast: float | None
    background_cv: float | None
    tumor_n_pixels: int
    background_n_pixels: int
    intensity_basis: str
    valid: bool
    message: str


def circular_mask(
    shape: tuple[int, int],
    center_xy: tuple[float, float],
    radius: float,
    *,
    fov: float = 1.0,
) -> np.ndarray:
    """Return a circular boolean mask in FOV coordinates."""
    n_row, n_col = int(shape[0]), int(shape[1])
    if n_row <= 0 or n_col <= 0:
        return np.zeros((max(n_row, 0), max(n_col, 0)), dtype=bool)
    # Square images use shared axes; rectangular fallback uses each dim.
    y_axis = image_axes(n_row, fov)
    x_axis = image_axes(n_col, fov)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")
    cx, cy = float(center_xy[0]), float(center_xy[1])
    r = max(float(radius), 0.0)
    return (xx - cx) ** 2 + (yy - cy) ** 2 <= r * r


def build_tumor_background_rois(
    image_shape: tuple[int, int],
    *,
    tumor_xy: tuple[float, float],
    tumor_radius: float,
    brain_radius: float,
    fov: float = 1.0,
    exclusion_factor: float = 1.5,
    background_width: float = 0.12,
    min_pixels: int = 4,
) -> ROIMasks:
    """Construct tumor disk + nearby brain-tissue background annulus."""
    n = int(image_shape[0])
    shape = (n, int(image_shape[1]))
    y_axis = image_axes(shape[0], fov)
    x_axis = image_axes(shape[1], fov)
    yy, xx = np.meshgrid(y_axis, x_axis, indexing="ij")

    brain = (xx * xx + yy * yy) <= (float(brain_radius) ** 2)
    tumor = circular_mask(
        shape, tumor_xy, tumor_radius, fov=fov
    )
    tumor &= brain

    inner = float(tumor_radius) * float(exclusion_factor)
    outer = inner + max(float(background_width), 1e-6)
    cx, cy = float(tumor_xy[0]), float(tumor_xy[1])
    r2 = (xx - cx) ** 2 + (yy - cy) ** 2
    background = (r2 >= inner * inner) & (r2 <= outer * outer) & brain & (~tumor)

    t_n = int(np.count_nonzero(tumor))
    b_n = int(np.count_nonzero(background))
    notes: list[str] = []
    if t_n < min_pixels:
        notes.append(f"tumor ROI too small ({t_n} < {min_pixels} pixels)")
    if b_n < min_pixels:
        notes.append(f"background ROI too small ({b_n} < {min_pixels} pixels)")
    if t_n == 0:
        notes.append("tumor ROI empty or outside the brain disk")
    if b_n == 0:
        notes.append("background ROI empty after exclusion / brain clip")

    msg = "; ".join(notes) if notes else "ROIs constructed successfully."
    return ROIMasks(
        tumor=tumor,
        background=background,
        tumor_n_pixels=t_n,
        background_n_pixels=b_n,
        message=msg,
    )


def compute_contrast_metrics(
    image: np.ndarray,
    masks: ROIMasks,
    *,
    use_nonnegative: bool = True,
    min_pixels: int = 4,
) -> ContrastResult:
    """Compute TBR, relative contrast, and background CV from ROI masks.

    Definitions
    -----------
    * ``TBR = mean_tumor / mean_background``
    * ``Contrast = (mean_tumor - mean_background) / mean_background``
    * ``Background CV = std_background / mean_background``

    Background CV is a variability measure — not automatically labeled as
    tumor SNR. Invalid / non-positive background means yield ``valid=False``.
    """
    basis = (
        "nonnegative display-clipped reconstructed intensities"
        if use_nonnegative
        else "raw reconstructed intensities (may include negatives)"
    )
    img = np.asarray(image, dtype=np.float64)
    if img.shape != masks.tumor.shape:
        return ContrastResult(
            tumor_mean=None,
            background_mean=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=masks.tumor_n_pixels,
            background_n_pixels=masks.background_n_pixels,
            intensity_basis=basis,
            valid=False,
            message="Image shape does not match ROI masks.",
        )

    work = np.maximum(img, 0.0) if use_nonnegative else img
    t_n = int(np.count_nonzero(masks.tumor))
    b_n = int(np.count_nonzero(masks.background))
    if t_n < min_pixels or b_n < min_pixels:
        return ContrastResult(
            tumor_mean=None,
            background_mean=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=t_n,
            background_n_pixels=b_n,
            intensity_basis=basis,
            valid=False,
            message=masks.message or "ROI too small for reliable metrics.",
        )

    tumor_vals = work[masks.tumor]
    bg_vals = work[masks.background]
    tumor_mean = float(np.mean(tumor_vals))
    bg_mean = float(np.mean(bg_vals))
    bg_std = float(np.std(bg_vals, ddof=0))

    if not np.isfinite(bg_mean) or bg_mean <= 0:
        return ContrastResult(
            tumor_mean=tumor_mean if np.isfinite(tumor_mean) else None,
            background_mean=bg_mean if np.isfinite(bg_mean) else None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=t_n,
            background_n_pixels=b_n,
            intensity_basis=basis,
            valid=False,
            message="Background mean is zero, negative, or non-finite — ratios undefined.",
        )

    tbr = tumor_mean / bg_mean
    contrast = (tumor_mean - bg_mean) / bg_mean
    bg_cv = bg_std / bg_mean
    if not all(np.isfinite(v) for v in (tbr, contrast, bg_cv)):
        return ContrastResult(
            tumor_mean=tumor_mean,
            background_mean=bg_mean,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=t_n,
            background_n_pixels=b_n,
            intensity_basis=basis,
            valid=False,
            message="Non-finite TBR / contrast / CV after division.",
        )

    return ContrastResult(
        tumor_mean=tumor_mean,
        background_mean=bg_mean,
        tbr=float(tbr),
        contrast=float(contrast),
        background_cv=float(bg_cv),
        tumor_n_pixels=t_n,
        background_n_pixels=b_n,
        intensity_basis=basis,
        valid=True,
        message="Contrast metrics computed from reconstructed image ROIs.",
    )
