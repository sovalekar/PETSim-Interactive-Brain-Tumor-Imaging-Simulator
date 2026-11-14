"""High-level Stage 4 analysis of one reconstruction + phantom state."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.analysis.localization import LocalizationResult, localize_tumor
from petsim.analysis.roi import (
    ContrastResult,
    ROIMasks,
    build_tumor_background_rois,
    compute_contrast_metrics,
)
from petsim.models.phantom import PhantomState
from petsim.reconstruction.pipeline import ReconstructionResult


@dataclass(frozen=True)
class QuantitativeAnalysisResult:
    """Bundle of localization + contrast metrics for the current reconstruction."""

    localization: LocalizationResult
    contrast: ContrastResult
    rois: ROIMasks | None
    n_coincidences: int
    message: str

    @property
    def available(self) -> bool:
        return self.localization.reliable or self.contrast.valid


def analyze_reconstruction(
    result: ReconstructionResult,
    phantom_state: PhantomState | None,
    *,
    use_nonnegative: bool = True,
) -> QuantitativeAnalysisResult:
    """Run localization and ROI metrics on a reconstructed image.

    Requires a phantom state that matches the acquisition used for
    ``result`` (ground-truth tumor geometry for evaluation / ROI placement).
    """
    if result is None or not result.has_data:
        empty_loc = localize_tumor(
            np.zeros((1, 1)),
            tumor_enabled=False,
            use_nonnegative=use_nonnegative,
        )
        empty_contrast = ContrastResult(
            tumor_mean=None,
            background_mean=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=0,
            background_n_pixels=0,
            intensity_basis="nonnegative display-clipped reconstructed intensities",
            valid=False,
            message="No reconstruction available for analysis.",
        )
        return QuantitativeAnalysisResult(
            localization=empty_loc,
            contrast=empty_contrast,
            rois=None,
            n_coincidences=0 if result is None else result.n_coincidences,
            message="No reconstruction available for analysis.",
        )

    if phantom_state is None:
        loc = localize_tumor(
            result.image,
            fov=result.config.fov,
            tumor_enabled=False,
            use_nonnegative=use_nonnegative,
        )
        # Override message for missing phantom metadata.
        loc = LocalizationResult(
            estimated_xy=None,
            true_xy=None,
            error=None,
            method=loc.method,
            reliable=False,
            message="Phantom state unavailable — cannot evaluate localization.",
        )
        contrast = ContrastResult(
            tumor_mean=None,
            background_mean=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=0,
            background_n_pixels=0,
            intensity_basis="nonnegative display-clipped reconstructed intensities",
            valid=False,
            message="Phantom state unavailable — ROI analysis requires tumor geometry.",
        )
        return QuantitativeAnalysisResult(
            localization=loc,
            contrast=contrast,
            rois=None,
            n_coincidences=result.n_coincidences,
            message="Analysis requires the phantom state frozen with the acquisition.",
        )

    state = phantom_state.clamp()
    true_xy = (state.tumor_x, state.tumor_y) if state.tumor_enabled else None
    loc = localize_tumor(
        result.image,
        fov=result.config.fov,
        brain_radius=state.brain_radius,
        neighborhood_radius=max(2.0 * state.tumor_radius, 0.12),
        true_xy=true_xy,
        tumor_enabled=state.tumor_enabled,
        use_nonnegative=use_nonnegative,
    )

    if not state.tumor_enabled:
        contrast = ContrastResult(
            tumor_mean=None,
            background_mean=None,
            tbr=None,
            contrast=None,
            background_cv=None,
            tumor_n_pixels=0,
            background_n_pixels=0,
            intensity_basis="nonnegative display-clipped reconstructed intensities",
            valid=False,
            message="Tumor disabled — contrast / TBR not applicable.",
        )
        return QuantitativeAnalysisResult(
            localization=loc,
            contrast=contrast,
            rois=None,
            n_coincidences=result.n_coincidences,
            message="Tumor disabled — quantitative tumor metrics unavailable.",
        )

    rois = build_tumor_background_rois(
        result.image.shape,
        tumor_xy=(state.tumor_x, state.tumor_y),
        tumor_radius=state.tumor_radius,
        brain_radius=state.brain_radius,
        fov=result.config.fov,
    )
    contrast = compute_contrast_metrics(
        result.image,
        rois,
        use_nonnegative=use_nonnegative,
    )
    msg = "Quantitative analysis complete."
    if not loc.reliable or not contrast.valid:
        msg = "Analysis finished with warnings — see localization / contrast messages."
    return QuantitativeAnalysisResult(
        localization=loc,
        contrast=contrast,
        rois=rois,
        n_coincidences=result.n_coincidences,
        message=msg,
    )
