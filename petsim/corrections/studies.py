"""Matched-acquisition studies of corrected and uncorrected reconstructions.

One simulated prompt list is reconstructed under every selected correction
mode. Ideal mode uses the pre-degradation endpoints from that same
simulation. Correction code still does not receive event labels.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable

import numpy as np

from petsim.corrections.metrics import ImageQualityMetrics, score_reconstruction
from petsim.corrections.reconstruct import (
    MODE_BOTH,
    MODE_IDEAL,
    MODE_RANDOM,
    MODE_SCATTER,
    MODE_UNCORRECTED,
    reconstruct_corrected,
)
from petsim.experiments.config import trial_seed
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import AcquisitionBuffer
from petsim.physics.detector_effects import DetectorPhysicsConfig
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig

ProgressCallback = Callable[[int, int, str], None]
CancelCallback = Callable[[], bool]


@dataclass
class CorrectionStudyConfig:
    """One factorial educational study. Defaults stay small enough to run."""

    phantom: PhantomState
    methods: tuple[str, ...] = ("fbp",)
    count_targets: tuple[int, ...] = (800,)
    random_fractions: tuple[float, ...] = (0.0, 0.2)
    scatter_probabilities: tuple[float, ...] = (0.0, 0.2)
    energy_windows: tuple[tuple[float, float], ...] = ((435.0, 650.0),)
    modes: tuple[str, ...] = (
        MODE_UNCORRECTED,
        MODE_RANDOM,
        MODE_SCATTER,
        MODE_BOTH,
    )
    n_trials: int = 1
    seed: int = 1
    n_iterations: int = 5
    detector_efficiency: float = 1.0
    energy_fwhm_percent: float = 12.0
    timing_fwhm_ps: float = 200.0
    timing_window_ps: float = 8000.0
    image_size: int = 48
    n_theta: int = 32
    n_s: int = 48
    n_detectors: int = 48

    def clamp(self) -> CorrectionStudyConfig:
        methods = tuple(m for m in ("fbp", "mlem") if m in {x.lower() for x in self.methods})
        counts = tuple(sorted({int(c) for c in self.count_targets if int(c) > 0}))
        randoms = tuple(sorted({float(np.clip(r, 0.0, 1.0)) for r in self.random_fractions}))
        scatters = tuple(sorted({float(np.clip(s, 0.0, 1.0)) for s in self.scatter_probabilities}))
        windows = []
        for low, high in self.energy_windows:
            lo = float(np.clip(low, 100.0, 511.0))
            hi = float(np.clip(high, 511.0, 800.0))
            if hi < lo:
                lo, hi = hi, lo
            windows.append((lo, hi))
        allowed = {MODE_IDEAL, MODE_UNCORRECTED, MODE_RANDOM, MODE_SCATTER, MODE_BOTH}
        modes = tuple(m for m in self.modes if m in allowed)
        return CorrectionStudyConfig(
            phantom=self.phantom.clamp(),
            methods=methods or ("fbp",),
            count_targets=counts,
            random_fractions=randoms,
            scatter_probabilities=scatters,
            energy_windows=tuple(windows) or ((435.0, 650.0),),
            modes=modes or (MODE_UNCORRECTED,),
            n_trials=int(np.clip(self.n_trials, 1, 20)),
            seed=int(self.seed) & 0xFFFFFFFF,
            n_iterations=int(np.clip(self.n_iterations, 1, 50)),
            detector_efficiency=float(np.clip(self.detector_efficiency, 0.2, 1.0)),
            energy_fwhm_percent=float(np.clip(self.energy_fwhm_percent, 0.0, 30.0)),
            timing_fwhm_ps=float(np.clip(self.timing_fwhm_ps, 0.0, 2000.0)),
            timing_window_ps=float(np.clip(self.timing_window_ps, 50.0, 20000.0)),
            image_size=int(np.clip(self.image_size, 32, 128)),
            n_theta=int(np.clip(self.n_theta, 16, 96)),
            n_s=int(np.clip(self.n_s, 16, 128)),
            n_detectors=int(np.clip(self.n_detectors, 8, 96)),
        )


@dataclass
class CorrectionTrial:
    count_target: int
    random_fraction: float
    scatter_probability: float
    energy_low_kev: float
    energy_high_kev: float
    method: str
    mode: str
    trial_index: int
    seed: int
    metrics: ImageQualityMetrics | None
    status: str
    message: str
    truth_random: int = 0
    truth_scattered: int = 0


@dataclass
class CorrectionStudyResult:
    config: CorrectionStudyConfig
    trials: list[CorrectionTrial] = field(default_factory=list)
    cancelled: bool = False
    message: str = ""
    job_id: int = 0


@dataclass
class MatchedAcquisition:
    """One prompt list plus the ideal-detector endpoints of the same run."""

    hits_a: np.ndarray
    hits_b: np.ndarray
    ideal_a: np.ndarray
    ideal_b: np.ndarray
    n_annihilations: int
    n_accepted: int
    truth_random: int
    truth_scattered: int
    n_side_window: int = 0


def acquire_matched(
    engine: PETEngine,
    n_target: int,
    physics: DetectorPhysicsConfig,
    *,
    detector_efficiency: float,
    cancel: CancelCallback | None = None,
) -> MatchedAcquisition | None:
    """Collect one accepted-coincidence prefix and its ideal endpoints."""
    target = int(n_target)
    efficiency = float(np.clip(detector_efficiency, 0.0, 1.0))
    rate = max(efficiency * efficiency * 0.25, 0.02)
    max_annihilations = int(target / rate * 8) + 1500
    buffer = AcquisitionBuffer()
    while int(np.count_nonzero(buffer.coincidence_flags)) < target:
        if cancel is not None and cancel():
            return None
        if buffer.n_annihilations() >= max_annihilations:
            break
        batch = min(400, max_annihilations - buffer.n_annihilations())
        buffer.extend(
            engine.simulate_batch(
                batch,
                detector_efficiency=efficiency,
                physics=physics,
            )
        )
    hits_a, hits_b = buffer.coincidence_endpoints()
    ideal_a, ideal_b = buffer.ideal_coincidence_endpoints()
    truth_random = 0
    truth_scattered = 0
    kept = 0
    cutoff = len(buffer.coincidence_flags)
    for index, (is_random, is_scattered, accepted) in enumerate(
        zip(
            buffer.is_random,
            buffer.is_scattered,
            buffer.coincidence_flags,
            strict=True,
        )
    ):
        if not accepted:
            continue
        kept += 1
        if is_random:
            truth_random += 1
        elif is_scattered:
            truth_scattered += 1
        if kept >= target:
            cutoff = index + 1
            break
    return MatchedAcquisition(
        hits_a=hits_a[:target],
        hits_b=hits_b[:target],
        ideal_a=ideal_a,
        ideal_b=ideal_b,
        n_annihilations=buffer.n_annihilations(),
        n_accepted=int(np.count_nonzero(buffer.coincidence_flags)),
        truth_random=truth_random,
        truth_scattered=truth_scattered,
        n_side_window=buffer.side_window_count(
            physics.energy_window_low_kev,
            physics.energy_window_high_kev,
            limit=cutoff,
        ),
    )


def run_correction_study(
    config: CorrectionStudyConfig,
    *,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
) -> CorrectionStudyResult:
    """Run the factorial study. Cancel stops between reconstructions."""
    cfg = config.clamp()
    steps = (
        len(cfg.count_targets)
        * len(cfg.random_fractions)
        * len(cfg.scatter_probabilities)
        * len(cfg.energy_windows)
        * cfg.n_trials
        * len(cfg.methods)
        * len(cfg.modes)
    )
    done = 0
    trials: list[CorrectionTrial] = []
    cancelled = False
    factor = 0
    for count in cfg.count_targets:
        for random_fraction in cfg.random_fractions:
            for scatter_probability in cfg.scatter_probabilities:
                for energy_low, energy_high in cfg.energy_windows:
                    for trial_index in range(cfg.n_trials):
                        if cancel is not None and cancel():
                            cancelled = True
                            break
                        seed = trial_seed(cfg.seed, factor, trial_index)
                        factor += 1
                        physics = DetectorPhysicsConfig(
                            enabled=True,
                            energy_fwhm_percent=cfg.energy_fwhm_percent,
                            energy_window_low_kev=energy_low,
                            energy_window_high_kev=energy_high,
                            timing_fwhm_ps=cfg.timing_fwhm_ps,
                            timing_window_ps=cfg.timing_window_ps,
                            random_fraction=random_fraction,
                            scatter_probability=scatter_probability,
                        )
                        engine = PETEngine(
                            BrainPhantom(cfg.phantom),
                            DetectorRing(n_detectors=cfg.n_detectors),
                            seed=seed,
                        )
                        acquired = acquire_matched(
                            engine,
                            count,
                            physics,
                            detector_efficiency=cfg.detector_efficiency,
                            cancel=cancel,
                        )
                        if acquired is None:
                            cancelled = True
                            break
                        for method in cfg.methods:
                            recon_cfg = ReconstructionConfig(
                                method=method,
                                n_iterations=cfg.n_iterations,
                                image_size=cfg.image_size,
                                n_theta=cfg.n_theta,
                                n_s=cfg.n_s,
                                s_max=0.92,
                                filter_name="hann",
                                smooth_sigma_s=1.0,
                                smooth_sigma_theta=0.5,
                            ).clamp()
                            for mode in cfg.modes:
                                if cancel is not None and cancel():
                                    cancelled = True
                                    break
                                hits_a, hits_b = acquired.hits_a, acquired.hits_b
                                if mode == MODE_IDEAL:
                                    hits_a, hits_b = acquired.ideal_a, acquired.ideal_b
                                kept = max(int(acquired.hits_a.shape[0]), 0)
                                scale = kept / max(int(acquired.n_accepted), 1)
                                result = reconstruct_corrected(
                                    hits_a,
                                    hits_b,
                                    recon_cfg,
                                    mode=mode,
                                    n_annihilations=acquired.n_annihilations,
                                    random_fraction=(
                                        0.0 if mode == MODE_IDEAL else random_fraction * scale
                                    ),
                                    n_detectors=cfg.n_detectors,
                                    n_side_window=0 if mode == MODE_IDEAL else acquired.n_side_window,
                                    scatter_probability=0.0 if mode == MODE_IDEAL else scatter_probability,
                                    photo_low_kev=energy_low,
                                    photo_high_kev=energy_high,
                                    energy_fwhm_percent=cfg.energy_fwhm_percent,
                                    cancel=cancel,
                                )
                                metrics = score_reconstruction(result, cfg.phantom)
                                status = "cancelled" if result.cancelled else "ok"
                                trials.append(
                                    CorrectionTrial(
                                        count_target=count,
                                        random_fraction=random_fraction,
                                        scatter_probability=scatter_probability,
                                        energy_low_kev=energy_low,
                                        energy_high_kev=energy_high,
                                        method=method,
                                        mode=mode,
                                        trial_index=trial_index,
                                        seed=seed,
                                        metrics=metrics,
                                        status=status,
                                        message=result.message,
                                        truth_random=acquired.truth_random,
                                        truth_scattered=acquired.truth_scattered,
                                    )
                                )
                                done += 1
                                if progress is not None:
                                    progress(
                                        done,
                                        steps,
                                        (
                                            f"{method.upper()} · {mode} · {count} pairs · "
                                            f"random {random_fraction:.2f} · scatter {scatter_probability:.2f}"
                                        ),
                                    )
                                if result.cancelled:
                                    cancelled = True
                                    break
                            if cancelled:
                                break
                        if cancelled:
                            break
                    if cancelled:
                        break
                if cancelled:
                    break
            if cancelled:
                break
        if cancelled:
            break
    if cancelled:
        message = "Correction study cancelled. Finished trials were kept."
    elif not trials:
        message = "No correction trials ran. Choose at least one count, method, and mode."
    else:
        ok = sum(trial.status == "ok" for trial in trials)
        message = f"Finished {ok} corrected/uncorrected reconstruction(s)."
    return CorrectionStudyResult(
        config=cfg,
        trials=trials,
        cancelled=cancelled,
        message=message,
    )
