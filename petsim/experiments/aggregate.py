"""Mean and standard deviation across independent acquisition trials.

These statistics describe variability between repeated simulations. They are
not MLEM convergence (that curve is Stage 5) and they are not a clinical
confidence interval.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.experiments.results import TrialResult

METRIC_FIELDS = (
    "localization_error",
    "tbr",
    "contrast",
    "background_cv",
    "runtime_s",
)


@dataclass(frozen=True)
class AggregateRow:
    count_target: int
    method: str
    n_trials: int
    n_ok: int
    localization_error_mean: float | None
    localization_error_std: float | None
    tbr_mean: float | None
    tbr_std: float | None
    contrast_mean: float | None
    contrast_std: float | None
    background_cv_mean: float | None
    background_cv_std: float | None
    runtime_s_mean: float | None
    runtime_s_std: float | None


def _mean_std(values: list[float | None]) -> tuple[float | None, float | None]:
    finite = [float(v) for v in values if v is not None and np.isfinite(v)]
    if not finite:
        return None, None
    mean = float(np.mean(finite))
    if len(finite) == 1:
        return mean, 0.0
    return mean, float(np.std(finite, ddof=1))


def aggregate_trials(trials: list[TrialResult]) -> list[AggregateRow]:
    """Group successful trials by coincidence target and reconstruction method."""
    groups: dict[tuple[int, str], list[TrialResult]] = {}
    for trial in trials:
        key = (int(trial.count_target), trial.method)
        groups.setdefault(key, []).append(trial)

    rows: list[AggregateRow] = []
    for count, method in sorted(groups):
        bucket = groups[(count, method)]
        ok = [trial for trial in bucket if trial.status == "ok"]
        stats = {
            field: _mean_std([getattr(trial, field) for trial in ok]) for field in METRIC_FIELDS
        }
        rows.append(
            AggregateRow(
                count_target=count,
                method=method,
                n_trials=len(bucket),
                n_ok=len(ok),
                localization_error_mean=stats["localization_error"][0],
                localization_error_std=stats["localization_error"][1],
                tbr_mean=stats["tbr"][0],
                tbr_std=stats["tbr"][1],
                contrast_mean=stats["contrast"][0],
                contrast_std=stats["contrast"][1],
                background_cv_mean=stats["background_cv"][0],
                background_cv_std=stats["background_cv"][1],
                runtime_s_mean=stats["runtime_s"][0],
                runtime_s_std=stats["runtime_s"][1],
            )
        )
    return rows
