"""Run repeated acquisitions and reconstructions at fixed coincidence counts.

Each trial draws annihilations until the requested number of detected
coincidences is reached, then keeps that exact prefix. Reconstruction uses
only those measured endpoints. Tumor coordinates are not reconstruction
inputs; Stage 4 analysis may use them afterwards to score localization error
and to place evaluation ROIs.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from petsim.analysis.pipeline import analyze_reconstruction
from petsim.experiments.aggregate import AggregateRow, aggregate_trials
from petsim.experiments.config import ExperimentConfig, trial_seed
from petsim.experiments.results import TrialResult
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class ExperimentResult:
    config: ExperimentConfig
    trials: list[TrialResult] = field(default_factory=list)
    aggregates: list[AggregateRow] = field(default_factory=list)
    cancelled: bool = False
    message: str = ""


def collect_coincidences(
    engine: PETEngine,
    n_target: int,
    *,
    detector_efficiency: float,
    cancel: CancelCallback | None = None,
    batch_size: int = 800,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Simulate until ``n_target`` coincidences exist and return that prefix.

    Returns None when ``cancel`` requests a stop. If the efficiency makes the
    target unreachable within a finite annihilation budget, the arrays may be
    shorter than ``n_target``.
    """
    target = int(n_target)
    if target <= 0:
        empty = np.zeros((0, 2), dtype=np.float64)
        return empty, empty

    efficiency = float(np.clip(detector_efficiency, 0.0, 1.0))
    rate = max(efficiency * efficiency, 0.02)
    max_annihilations = int(target / rate * 12) + 2000
    buf = AcquisitionBuffer()
    batch = max(int(batch_size), 50)
    while int(np.count_nonzero(buf.coincidence_flags)) < target:
        if cancel is not None and cancel():
            return None
        if len(buf) >= max_annihilations:
            break
        n = min(batch, max_annihilations - len(buf))
        buf.extend(engine.simulate_batch(n, detector_efficiency=efficiency))
    hits_a, hits_b = buf.coincidence_endpoints()
    return hits_a[:target], hits_b[:target]


def _blank_trial(
    *,
    count: int,
    actual: int,
    trial_index: int,
    method: str,
    seed: int,
    status: str,
    message: str,
) -> TrialResult:
    return TrialResult(
        count_target=count,
        actual_count=actual,
        trial_index=trial_index,
        method=method,
        seed=seed,
        localization_error=None,
        tbr=None,
        contrast=None,
        background_cv=None,
        runtime_s=0.0,
        n_iterations=None,
        status=status,
        message=message,
    )


def _score_reconstruction(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    config: ExperimentConfig,
    method: str,
) -> tuple[TrialResult, bool]:
    recon_cfg = ReconstructionConfig(
        method=method,
        filter_name=config.filter_name,
        n_theta=config.n_theta,
        n_s=config.n_s,
        image_size=config.image_size,
        fov=config.fov,
        s_max=config.s_max,
        smooth_sigma_s=config.smooth_sigma_s if method == "fbp" else 0.0,
        smooth_sigma_theta=config.smooth_sigma_theta if method == "fbp" else 0.0,
        n_iterations=config.n_iterations,
    ).clamp()
    result = reconstruct_from_endpoints(hits_a, hits_b, recon_cfg)
    analysis = analyze_reconstruction(result, config.phantom)
    loc = analysis.localization.error if analysis.localization.reliable else None
    contrast = analysis.contrast
    iters = None
    if method == "mlem" and result.history is not None:
        iters = int(len(result.history.iterations))
    trial = TrialResult(
        count_target=int(hits_a.shape[0]),
        actual_count=int(hits_a.shape[0]),
        trial_index=-1,
        method=method,
        seed=-1,
        localization_error=loc,
        tbr=contrast.tbr if contrast.valid else None,
        contrast=contrast.contrast if contrast.valid else None,
        background_cv=contrast.background_cv if contrast.valid else None,
        runtime_s=float(result.runtime_s),
        n_iterations=iters,
        status="cancelled" if result.cancelled else "ok",
        message=result.message,
    )
    return trial, bool(result.cancelled)


def run_experiment(
    config: ExperimentConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> ExperimentResult:
    """Execute every count level, trial, and selected reconstruction method."""
    cfg = config.clamp()
    if not cfg.methods or not cfg.count_targets:
        return ExperimentResult(
            config=cfg,
            message="Select at least one reconstruction method and one coincidence target.",
        )

    steps = len(cfg.count_targets) * cfg.n_trials * len(cfg.methods)
    done = 0
    trials: list[TrialResult] = []
    cancelled = False

    for count_index, count in enumerate(cfg.count_targets):
        for trial_index in range(cfg.n_trials):
            if cancel is not None and cancel():
                cancelled = True
                break
            seed = trial_seed(cfg.seed, count_index, trial_index)
            phantom = BrainPhantom(cfg.phantom)
            engine = PETEngine(phantom, DetectorRing(n_detectors=cfg.n_detectors))
            engine.set_seed(seed)
            acquired = collect_coincidences(
                engine,
                count,
                detector_efficiency=cfg.detector_efficiency,
                cancel=cancel,
            )
            if acquired is None:
                cancelled = True
                for method in cfg.methods:
                    trials.append(
                        _blank_trial(
                            count=count,
                            actual=0,
                            trial_index=trial_index,
                            method=method,
                            seed=seed,
                            status="cancelled",
                            message="Cancelled while acquiring coincidences.",
                        )
                    )
                    done += 1
                break
            hits_a, hits_b = acquired
            actual = int(hits_a.shape[0])
            if actual < count:
                for method in cfg.methods:
                    trials.append(
                        _blank_trial(
                            count=count,
                            actual=actual,
                            trial_index=trial_index,
                            method=method,
                            seed=seed,
                            status="insufficient",
                            message=(
                                f"Collected {actual} coincidences, short of {count}. "
                                "Metrics were not invented for the missing events."
                            ),
                        )
                    )
                    done += 1
                    if progress is not None:
                        progress(done, steps, f"{count} coincidences · trial {trial_index + 1} · {method} insufficient")
                continue

            for method in cfg.methods:
                if cancel is not None and cancel():
                    cancelled = True
                    trials.append(
                        _blank_trial(
                            count=count,
                            actual=actual,
                            trial_index=trial_index,
                            method=method,
                            seed=seed,
                            status="cancelled",
                            message="Cancelled before this reconstruction.",
                        )
                    )
                    done += 1
                    break
                scored, was_cancelled = _score_reconstruction(hits_a, hits_b, cfg, method)
                scored.trial_index = trial_index
                scored.seed = seed
                scored.count_target = count
                trials.append(scored)
                done += 1
                if progress is not None:
                    progress(
                        done,
                        steps,
                        f"{count:,} coincidences · trial {trial_index + 1}/{cfg.n_trials} · {method.upper()}",
                    )
                if was_cancelled:
                    cancelled = True
                    break
            if cancelled:
                break
        if cancelled:
            break

    aggregates = aggregate_trials(trials)
    if cancelled:
        message = "Experiment cancelled. Completed trials are listed; remaining work was not fabricated."
    elif not any(trial.status == "ok" for trial in trials):
        message = "No trial completed successfully."
    else:
        message = (
            f"Finished {sum(trial.status == 'ok' for trial in trials)} successful "
            f"reconstruction(s) across {len(cfg.count_targets)} count level(s)."
        )
    return ExperimentResult(
        config=cfg,
        trials=trials,
        aggregates=aggregates,
        cancelled=cancelled,
        message=message,
    )
