"""One reconstruction scored inside an image-quality experiment."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class TrialResult:
    count_target: int
    actual_count: int
    trial_index: int
    method: str
    seed: int
    localization_error: float | None
    tbr: float | None
    contrast: float | None
    background_cv: float | None
    runtime_s: float
    n_iterations: int | None
    status: str
    message: str
