"""Matched conventional MLEM and TOF-MLEM on one timed acquisition per seed.

Both reconstructions see the same true coincidences. Conventional MLEM ignores
the arrival times. TOF-MLEM uses them. Random coincidences are not generated
here and would be excluded if they were: this comparison has no additive
random background.

The known tumor position is used only by the existing scorer, after the images
exist. A narrower timing kernel is expected to tighten localization on
noiseless data. A noisy seed is not required to improve every score.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from petsim.corrections.metrics import score_reconstruction
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.attenuation import AttenuationConfig
from petsim.physics.pet_engine import PETEngine
from petsim.physics.tof import TOFConfig, most_likely_points
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints


@dataclass
class TOFSeedComparison:
    """One seed. Every localization error is kept, including large ones."""

    seed: int
    n_true: int
    conventional_nrmse: float
    tof_nrmse: float
    conventional_contrast: float
    tof_contrast: float
    conventional_cv: float
    tof_cv: float
    conventional_localization: float
    tof_localization: float


@dataclass
class TOFStudyResult:
    """Images from the last seed and scores from every seed."""

    conventional_image: np.ndarray
    tof_image: np.ndarray
    estimate_xy: np.ndarray
    comparisons: list[TOFSeedComparison] = field(default_factory=list)
    message: str = ""

    def _values(self, name: str) -> np.ndarray:
        return np.asarray([getattr(item, name) for item in self.comparisons], dtype=np.float64)

    def mean(self, name: str) -> float:
        values = self._values(name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(np.mean(finite))

    def median(self, name: str) -> float:
        values = self._values(name)
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return float("nan")
        return float(np.median(finite))


def _fmt(value: float) -> str:
    if not np.isfinite(value):
        return "no estimate"
    return f"{value:.3f}"


def run_tof_study(
    phantom: PhantomState,
    tof: TOFConfig,
    *,
    attenuation: AttenuationConfig | None = None,
    seeds: tuple[int, ...] = (1, 2, 3, 4, 5),
    n_events: int = 2000,
    n_iterations: int = 8,
    image_size: int = 48,
    n_theta: int = 36,
    n_s: int = 48,
) -> TOFStudyResult:
    """Reconstruct each seed twice from one TOF acquisition."""
    state = phantom.clamp()
    timing = tof.clamp()
    timing = TOFConfig(
        enabled=True,
        fwhm_ps=timing.fwhm_ps,
        cm_per_fov=timing.cm_per_fov,
        n_t=timing.n_t,
        t_max=timing.t_max,
    )
    atten = None if attenuation is None else attenuation.clamp()
    if atten is not None and not atten.active:
        atten = None
    recon = ReconstructionConfig(
        method="mlem",
        n_iterations=n_iterations,
        image_size=image_size,
        n_theta=n_theta,
        n_s=n_s,
        smooth_sigma_s=0.0,
        smooth_sigma_theta=0.0,
    ).clamp()
    comparisons: list[TOFSeedComparison] = []
    conventional_image = np.zeros((recon.image_size, recon.image_size))
    tof_image = conventional_image.copy()
    estimate_xy = np.zeros((0, 2))
    for seed in seeds:
        engine = PETEngine(
            BrainPhantom(state=state, grid_size=128),
            DetectorRing(n_detectors=48),
            seed=int(seed),
        )
        events = engine.simulate_batch(
            int(n_events),
            detector_efficiency=1.0,
            attenuation=atten,
            tof=timing,
        )
        pairs = [event for event in events if event.is_coincidence and not event.is_random]
        if not pairs:
            hits_a = np.zeros((0, 2))
            hits_b = np.zeros((0, 2))
            delta_t = np.zeros(0)
        else:
            hits_a = np.array([event.hit_a_xy for event in pairs], dtype=np.float64)
            hits_b = np.array([event.hit_b_xy for event in pairs], dtype=np.float64)
            delta_t = np.array([event.measured_delta_t_ps for event in pairs], dtype=np.float64)
        conventional = reconstruct_from_endpoints(hits_a, hits_b, recon, attenuation=atten)
        timed = reconstruct_from_endpoints(
            hits_a,
            hits_b,
            recon,
            attenuation=atten,
            tof=timing,
            delta_t_ps=delta_t,
        )
        conventional_score = score_reconstruction(conventional, state)
        tof_score = score_reconstruction(timed, state)
        comparisons.append(
            TOFSeedComparison(
                seed=int(seed),
                n_true=int(hits_a.shape[0]),
                conventional_nrmse=float("nan") if conventional_score.nrmse is None else conventional_score.nrmse,
                tof_nrmse=float("nan") if tof_score.nrmse is None else tof_score.nrmse,
                conventional_contrast=float("nan") if conventional_score.contrast is None else conventional_score.contrast,
                tof_contrast=float("nan") if tof_score.contrast is None else tof_score.contrast,
                conventional_cv=float("nan") if conventional_score.background_cv is None else conventional_score.background_cv,
                tof_cv=float("nan") if tof_score.background_cv is None else tof_score.background_cv,
                conventional_localization=(
                    float("nan")
                    if conventional_score.localization_error is None
                    else conventional_score.localization_error
                ),
                tof_localization=(
                    float("nan") if tof_score.localization_error is None else tof_score.localization_error
                ),
            )
        )
        conventional_image = conventional.image
        tof_image = timed.image
        sample = min(40, hits_a.shape[0])
        estimate_xy = most_likely_points(hits_a[:sample], hits_b[:sample], delta_t[:sample], timing.cm_per_fov)
    lines = []
    for item in comparisons:
        lines.append(
            f"seed {item.seed} localization conventional {_fmt(item.conventional_localization)} "
            f"TOF {_fmt(item.tof_localization)} "
            f"NRMSE {_fmt(item.conventional_nrmse)} → {_fmt(item.tof_nrmse)} "
            f"contrast {_fmt(item.conventional_contrast)} → {_fmt(item.tof_contrast)} "
            f"CV {_fmt(item.conventional_cv)} → {_fmt(item.tof_cv)}"
        )
    message = (
        f"TOF-MLEM · FWHM {timing.fwhm_ps:g} ps · {len(comparisons)} seed(s) · "
        f"localization median conventional {_fmt(_median(comparisons, 'conventional_localization'))} "
        f"TOF {_fmt(_median(comparisons, 'tof_localization'))} · "
        f"mean conventional {_fmt(_mean(comparisons, 'conventional_localization'))} "
        f"TOF {_fmt(_mean(comparisons, 'tof_localization'))}. "
        + " ".join(lines)
        + ". Every seed is included. A narrower kernel tightens noiseless localization. "
        "A noisy seed is not required to improve NRMSE, contrast, and CV together. "
        "Random coincidences are excluded. There is no additive random background."
    )
    return TOFStudyResult(
        conventional_image=conventional_image,
        tof_image=tof_image,
        estimate_xy=estimate_xy,
        comparisons=comparisons,
        message=message,
    )


def _mean(items: list[TOFSeedComparison], name: str) -> float:
    values = np.asarray([getattr(item, name) for item in items], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _median(items: list[TOFSeedComparison], name: str) -> float:
    values = np.asarray([getattr(item, name) for item in items], dtype=np.float64)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))
