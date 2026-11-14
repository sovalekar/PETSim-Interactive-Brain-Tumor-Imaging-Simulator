"""Reconstruction quality versus detected coincidence count.

Approach
--------
Given one sufficiently large measured coincidence list, form **nested
prefixes** (the first N events) for each target count level. Each subset is
rebinned into a sinogram and reconstructed with the **same** FBP settings,
then scored with the same localization / ROI definitions.

Target levels are detected coincidence counts, not annihilation counts.
Levels above the available coincidence total are reported as unavailable —
additional events are never fabricated.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from petsim.analysis.pipeline import QuantitativeAnalysisResult, analyze_reconstruction
from petsim.models.phantom import PhantomState
from petsim.reconstruction.pipeline import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_from_endpoints,
)

DEFAULT_COUNT_LEVELS: tuple[int, ...] = (1000, 3000, 5000, 10000)


@dataclass(frozen=True)
class CountLevelMetrics:
    """Metrics for one coincidence-count level."""

    target_count: int
    actual_count: int
    available: bool
    localization_error: float | None
    tbr: float | None
    background_cv: float | None
    message: str
    result: ReconstructionResult | None = None
    analysis: QuantitativeAnalysisResult | None = None


@dataclass
class CountComparisonResult:
    """Table of metrics across coincidence-count levels."""

    levels: list[CountLevelMetrics] = field(default_factory=list)
    total_available: int = 0
    message: str = ""

    @property
    def has_any(self) -> bool:
        return any(level.available for level in self.levels)

    def rows_for_table(self) -> list[dict[str, object]]:
        rows: list[dict[str, object]] = []
        for level in self.levels:
            rows.append(
                {
                    "target": level.target_count,
                    "actual": level.actual_count,
                    "available": level.available,
                    "loc_error": level.localization_error,
                    "tbr": level.tbr,
                    "bg_cv": level.background_cv,
                    "message": level.message,
                }
            )
        return rows


def compare_by_coincidence_count(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    config: ReconstructionConfig,
    phantom_state: PhantomState | None,
    *,
    target_counts: tuple[int, ...] | list[int] = DEFAULT_COUNT_LEVELS,
    use_nonnegative: bool = True,
) -> CountComparisonResult:
    """Reconstruct nested coincidence subsets and compute Stage 4 metrics."""
    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    if a.shape[0] != b.shape[0]:
        return CountComparisonResult(
            levels=[],
            total_available=0,
            message="Coincidence endpoint arrays have mismatched lengths.",
        )

    n_total = int(a.shape[0])
    if n_total == 0:
        return CountComparisonResult(
            levels=[],
            total_available=0,
            message="No coincidence endpoints available for count comparison.",
        )

    cfg = config.clamp()
    levels: list[CountLevelMetrics] = []
    sorted_targets = sorted({int(t) for t in target_counts if int(t) > 0})

    for target in sorted_targets:
        if n_total < target:
            levels.append(
                CountLevelMetrics(
                    target_count=target,
                    actual_count=n_total,
                    available=False,
                    localization_error=None,
                    tbr=None,
                    background_cv=None,
                    message=(
                        f"Insufficient coincidences: need {target:,}, "
                        f"have {n_total:,}."
                    ),
                )
            )
            continue

        # Nested prefix of the same acquisition (reproducible given event order).
        sub_a = a[:target]
        sub_b = b[:target]
        recon = reconstruct_from_endpoints(sub_a, sub_b, cfg)
        analysis = analyze_reconstruction(
            recon,
            phantom_state,
            use_nonnegative=use_nonnegative,
        )
        loc_err = analysis.localization.error if analysis.localization.reliable else None
        tbr = analysis.contrast.tbr if analysis.contrast.valid else None
        bg_cv = analysis.contrast.background_cv if analysis.contrast.valid else None
        levels.append(
            CountLevelMetrics(
                target_count=target,
                actual_count=target,
                available=True,
                localization_error=loc_err,
                tbr=tbr,
                background_cv=bg_cv,
                message="OK",
                result=recon,
                analysis=analysis,
            )
        )

    n_ok = sum(1 for level in levels if level.available)
    if n_ok == 0:
        msg = (
            f"No target count levels are reachable with {n_total:,} "
            "recorded coincidences."
        )
    else:
        msg = (
            f"Compared {n_ok} count level(s) using nested prefixes of "
            f"{n_total:,} coincidences (same FBP settings)."
        )
    return CountComparisonResult(
        levels=levels,
        total_available=n_total,
        message=msg,
    )
