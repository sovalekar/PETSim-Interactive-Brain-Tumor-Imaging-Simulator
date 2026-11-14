"""Reconstruct the same acquisition under ideal and degraded detector models.

Both calls receive only ring-hit endpoints. Tumor coordinates are not an
argument and are not read here.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

from petsim.reconstruction.pipeline import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_from_endpoints,
)

CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[int, int, float, float], None]


def reconstruct_ideal_and_degraded(
    ideal_hits_a: np.ndarray,
    ideal_hits_b: np.ndarray,
    degraded_hits_a: np.ndarray,
    degraded_hits_b: np.ndarray,
    config: ReconstructionConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    job_id: int = 0,
) -> tuple[ReconstructionResult, ReconstructionResult]:
    """Run one algorithm on ideal endpoints, then on accepted degraded endpoints."""
    cfg = config.clamp()
    ideal = reconstruct_from_endpoints(
        ideal_hits_a,
        ideal_hits_b,
        cfg,
        progress=progress,
        cancel=cancel,
        job_id=job_id,
    )
    if cancel is not None and cancel():
        empty = reconstruct_from_endpoints(
            degraded_hits_a[:0],
            degraded_hits_b[:0],
            cfg,
            job_id=job_id,
        )
        empty.cancelled = True
        empty.message = "Degraded reconstruction cancelled before it started."
        ideal.cancelled = True
        if not ideal.message:
            ideal.message = "Comparison cancelled."
        return ideal, empty
    degraded = reconstruct_from_endpoints(
        degraded_hits_a,
        degraded_hits_b,
        cfg,
        progress=progress,
        cancel=cancel,
        job_id=job_id,
    )
    return ideal, degraded
