"""Image-quality metrics for corrected and uncorrected reconstructions.

Phantom geometry is used only here, after reconstruction. Intensities are
not calibrated to activity, so RMSE uses an explicit nonnegative scale.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.analysis.pipeline import analyze_reconstruction
from petsim.analysis.roi import build_tumor_background_rois
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.reconstruction.pipeline import ReconstructionResult


@dataclass(frozen=True)
class ImageQualityMetrics:
    """Educational scores. None means the metric is undefined for this image."""

    localization_error: float | None
    tbr: float | None
    contrast: float | None
    background_cv: float | None
    runtime_s: float
    rmse: float | None
    nrmse: float | None
    scale_alpha: float | None
    crc: float | None
    message: str


def scale_normalized_rmse(
    image: np.ndarray,
    activity: np.ndarray,
) -> tuple[float | None, float | None, float | None]:
    """RMSE after a nonnegative least-squares intensity scale.

    α = max(r · a / r · r, 0), with r = max(reconstruction, 0) and a the
    phantom raster on the same grid.

    RMSE = sqrt(mean((α r − a)²)), in phantom intensity units.
    NRMSE = RMSE / sqrt(mean(a²)). The denominator is the RMS of the
    activity map, not the maximum voxel.

    A global scale is required because FBP and MLEM outputs are not in
    phantom intensity units. The scale is evaluation-only.
    """
    reconstructed = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    reference = np.asarray(activity, dtype=np.float64)
    if reconstructed.shape != reference.shape or reconstructed.size == 0:
        return None, None, None
    gram = float(np.dot(reconstructed.ravel(), reconstructed.ravel()))
    alpha = 0.0
    if gram > 0.0:
        alpha = max(float(np.dot(reconstructed.ravel(), reference.ravel()) / gram), 0.0)
    residual = alpha * reconstructed - reference
    rmse = float(np.sqrt(np.mean(residual * residual)))
    reference_rms = float(np.sqrt(np.mean(reference * reference)))
    nrmse = None if reference_rms <= 1e-12 else rmse / reference_rms
    return rmse, nrmse, alpha


def contrast_recovery_coefficient(
    measured_contrast: float | None,
    activity: np.ndarray,
    phantom: PhantomState,
    *,
    fov: float,
) -> float | None:
    """CRC = measured contrast / phantom contrast on the same ROIs.

    Contrast is (μ_tumor − μ_background) / μ_background. The phantom value
    uses the rasterized activity, including the brain's soft falloff, not
    the nominal slider intensities alone. CRC is undefined when either
    contrast is missing or the phantom background mean is near zero.
    """
    if measured_contrast is None or not np.isfinite(measured_contrast):
        return None
    state = phantom.clamp()
    if not state.tumor_enabled:
        return None
    masks = build_tumor_background_rois(
        activity.shape,
        tumor_xy=(state.tumor_x, state.tumor_y),
        tumor_radius=state.tumor_radius,
        brain_radius=state.brain_radius,
        fov=fov,
    )
    if masks.tumor_n_pixels < 1 or masks.background_n_pixels < 1:
        return None
    values = np.asarray(activity, dtype=np.float64)
    tumor_mean = float(values[masks.tumor].mean())
    background_mean = float(values[masks.background].mean())
    if abs(background_mean) <= 1e-8:
        return None
    true_contrast = (tumor_mean - background_mean) / background_mean
    if abs(true_contrast) <= 1e-8:
        return None
    return float(measured_contrast) / true_contrast


def score_reconstruction(
    result: ReconstructionResult,
    phantom: PhantomState | None,
) -> ImageQualityMetrics:
    """Score one finished reconstruction. The phantom is not a recon input."""
    if phantom is None or result.image.size == 0:
        return ImageQualityMetrics(
            localization_error=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            runtime_s=float(result.runtime_s),
            rmse=None,
            nrmse=None,
            scale_alpha=None,
            crc=None,
            message="Phantom or image unavailable.",
        )
    analysis = analyze_reconstruction(result, phantom)
    activity = (
        BrainPhantom(state=phantom, grid_size=int(result.image.shape[0]))
        .rasterize()
        .astype(np.float64)
    )
    rmse, nrmse, alpha = scale_normalized_rmse(result.image, activity)
    contrast = analysis.contrast.contrast if analysis.contrast.valid else None
    crc = contrast_recovery_coefficient(
        contrast,
        activity,
        phantom,
        fov=result.config.fov,
    )
    loc = analysis.localization.error if analysis.localization.reliable else None
    return ImageQualityMetrics(
        localization_error=loc,
        tbr=analysis.contrast.tbr if analysis.contrast.valid else None,
        contrast=contrast,
        background_cv=analysis.contrast.background_cv if analysis.contrast.valid else None,
        runtime_s=float(result.runtime_s),
        rmse=rmse,
        nrmse=nrmse,
        scale_alpha=alpha,
        crc=crc,
        message=analysis.message,
    )
