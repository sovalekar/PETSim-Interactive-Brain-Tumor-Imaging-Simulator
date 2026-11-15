"""Paired ideal, attenuated, and attenuation-corrected reconstructions.

One seeded emission is simulated without attenuation. The same seed is then
simulated with attenuation. Because survival random numbers are drawn after
the emission and efficiency draws, the annihilations and the detection
attempts match. Attenuation only removes photons.

The corrected reconstruction uses that attenuated list once. FBP scales the
sinogram. MLEM weights the projector. The two corrections are never stacked.

Metrics use the known tumor only after reconstruction, to score the images.
They are not an input to the correction.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from petsim.analysis.localization import localize_tumor
from petsim.analysis.roi import build_tumor_background_rois
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.attenuation import (
    AttenuationConfig,
    attenuation_map,
)
from petsim.physics.pet_engine import PETEngine
from petsim.reconstruction.pipeline import ReconstructionConfig, reconstruct_from_endpoints


@dataclass
class RegionScore:
    """Tumor mean, background CV, and localization error in image units.

    ``localization_error`` is the distance from the existing estimator to the
    known tumor center. The estimator does not receive that center as a search
    guide. NaN means the estimator produced no position. Every seed is kept.
    """

    tumor_mean: float
    background_cv: float
    n_coincidences: int
    localization_error: float


@dataclass
class SeedComparison:
    """One seed, one algorithm."""

    seed: int
    method: str
    ideal: RegionScore
    attenuated: RegionScore
    corrected: RegionScore

    @property
    def uncorrected_recovery(self) -> float:
        if self.ideal.tumor_mean <= 0.0:
            return float("nan")
        return self.attenuated.tumor_mean / self.ideal.tumor_mean

    @property
    def corrected_recovery(self) -> float:
        if self.ideal.tumor_mean <= 0.0:
            return float("nan")
        return self.corrected.tumor_mean / self.ideal.tumor_mean


@dataclass
class AttenuationStudyResult:
    """Images from the last seed plus scores from every seed."""

    mu_map: np.ndarray
    method: str
    ideal_image: np.ndarray
    attenuated_image: np.ndarray
    corrected_image: np.ndarray
    comparisons: list[SeedComparison] = field(default_factory=list)
    message: str = ""

    @property
    def mean_uncorrected_recovery(self) -> float:
        return _nanmean([item.uncorrected_recovery for item in self.comparisons])

    @property
    def mean_corrected_recovery(self) -> float:
        return _nanmean([item.corrected_recovery for item in self.comparisons])

    @property
    def mean_corrected_cv(self) -> float:
        return _nanmean([item.corrected.background_cv for item in self.comparisons])

    @property
    def mean_ideal_cv(self) -> float:
        return _nanmean([item.ideal.background_cv for item in self.comparisons])

    @property
    def mean_ideal_localization(self) -> float:
        return _nanmean([item.ideal.localization_error for item in self.comparisons])

    @property
    def median_ideal_localization(self) -> float:
        return _nanmedian([item.ideal.localization_error for item in self.comparisons])

    @property
    def mean_attenuated_localization(self) -> float:
        return _nanmean([item.attenuated.localization_error for item in self.comparisons])

    @property
    def median_attenuated_localization(self) -> float:
        return _nanmedian([item.attenuated.localization_error for item in self.comparisons])

    @property
    def mean_corrected_localization(self) -> float:
        return _nanmean([item.corrected.localization_error for item in self.comparisons])

    @property
    def median_corrected_localization(self) -> float:
        return _nanmedian([item.corrected.localization_error for item in self.comparisons])


def _finite(values: list[float]) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _nanmean(values: list[float]) -> float:
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return float(np.mean(finite))


def _nanmedian(values: list[float]) -> float:
    """Median of every finite error. Large errors stay in the sample."""
    finite = _finite(values)
    if finite.size == 0:
        return float("nan")
    return float(np.median(finite))


def _fmt_error(value: float) -> str:
    if not np.isfinite(value):
        return "no estimate"
    return f"{value:.3f}"


def _endpoints(events) -> tuple[np.ndarray, np.ndarray]:
    pairs = [event for event in events if event.is_coincidence and not event.is_random]
    if not pairs:
        return np.zeros((0, 2)), np.zeros((0, 2))
    hits_a = np.array([event.hit_a_xy for event in pairs], dtype=np.float64)
    hits_b = np.array([event.hit_b_xy for event in pairs], dtype=np.float64)
    return hits_a, hits_b


def _score(image: np.ndarray, state: PhantomState, n_coincidences: int, fov: float) -> RegionScore:
    masks = build_tumor_background_rois(
        image.shape,
        tumor_xy=(state.tumor_x, state.tumor_y),
        tumor_radius=state.tumor_radius,
        brain_radius=state.brain_radius,
        fov=fov,
    )
    display = np.maximum(np.asarray(image, dtype=np.float64), 0.0)
    if masks.tumor_n_pixels == 0 or not np.any(masks.tumor):
        tumor_mean = float("nan")
    else:
        tumor_mean = float(np.mean(display[masks.tumor]))
    background = display[masks.background] if np.any(masks.background) else np.zeros(0)
    mean = float(np.mean(background)) if background.size else 0.0
    if background.size == 0 or mean <= 0.0:
        cv = float("nan")
    else:
        cv = float(np.std(background) / mean)
    # The known center is passed only so the existing function can return a
    # distance. Tests lock that this center does not move the estimate.
    located = localize_tumor(
        display,
        fov=fov,
        brain_radius=state.brain_radius,
        neighborhood_radius=max(2.0 * state.tumor_radius, 0.12),
        true_xy=(state.tumor_x, state.tumor_y) if state.tumor_enabled else None,
        tumor_enabled=state.tumor_enabled,
    )
    error = float(located.error) if located.error is not None else float("nan")
    return RegionScore(tumor_mean, cv, int(n_coincidences), error)


def run_attenuation_study(
    phantom: PhantomState,
    attenuation: AttenuationConfig,
    *,
    seeds: tuple[int, ...] = (1, 2, 3),
    method: str = "fbp",
    n_events: int = 2000,
    detector_efficiency: float = 1.0,
    n_iterations: int = 6,
    image_size: int = 48,
    n_theta: int = 48,
    n_s: int = 64,
) -> AttenuationStudyResult:
    """Repeat a paired comparison. The returned images belong to the last seed."""
    state = phantom.clamp()
    config = attenuation.clamp()
    # The ellipse follows the phantom radius. A disabled or μ = 0 study still
    # runs: the three reconstructions then match, because no photons are lost.
    active = AttenuationConfig(
        enabled=bool(config.enabled and config.mu_per_cm > 0.0),
        mu_per_cm=config.mu_per_cm,
        cm_per_fov=config.cm_per_fov,
        brain_radius=state.brain_radius,
        a_min=config.a_min,
    )
    recon = ReconstructionConfig(
        method=method,
        filter_name="hann",
        n_theta=n_theta,
        n_s=n_s,
        image_size=image_size,
        fov=1.0,
        s_max=0.92,
        smooth_sigma_s=0.0,
        smooth_sigma_theta=0.0,
        n_iterations=n_iterations,
    ).clamp()
    comparisons: list[SeedComparison] = []
    ideal_image = np.zeros((recon.image_size, recon.image_size))
    attenuated_image = ideal_image.copy()
    corrected_image = ideal_image.copy()
    for seed in seeds:
        phantom_model = BrainPhantom(state=state, grid_size=128)
        engine = PETEngine(phantom_model, DetectorRing(n_detectors=48), seed=int(seed))
        ideal_events = engine.simulate_batch(
            int(n_events),
            detector_efficiency=detector_efficiency,
        )
        engine.set_seed(int(seed))
        attenuated_events = engine.simulate_batch(
            int(n_events),
            detector_efficiency=detector_efficiency,
            attenuation=active,
        )
        ideal_a, ideal_b = _endpoints(ideal_events)
        att_a, att_b = _endpoints(attenuated_events)
        ideal = reconstruct_from_endpoints(ideal_a, ideal_b, recon)
        attenuated = reconstruct_from_endpoints(att_a, att_b, recon)
        corrected = reconstruct_from_endpoints(
            att_a, att_b, recon, attenuation=active if active.active else None
        )
        ideal_score = _score(ideal.image, state, ideal.n_coincidences, recon.fov)
        attenuated_score = _score(attenuated.image, state, attenuated.n_coincidences, recon.fov)
        corrected_score = _score(corrected.image, state, corrected.n_coincidences, recon.fov)
        comparisons.append(
            SeedComparison(
                int(seed),
                recon.method,
                ideal_score,
                attenuated_score,
                corrected_score,
            )
        )
        ideal_image = ideal.image
        attenuated_image = attenuated.image
        corrected_image = corrected.image
    mu_map = attenuation_map(recon.image_size, active if active.active else config)
    mean_gain = _nanmean(
        [item.corrected_recovery - item.uncorrected_recovery for item in comparisons]
    )
    cv_change = _nanmean(
        [item.corrected.background_cv - item.ideal.background_cv for item in comparisons]
    )
    per_seed = [
        (
            f"seed {item.seed} localization "
            f"ideal {_fmt_error(item.ideal.localization_error)}, "
            f"attenuated {_fmt_error(item.attenuated.localization_error)}, "
            f"corrected {_fmt_error(item.corrected.localization_error)}"
        )
        for item in comparisons
    ]
    produced = sum(
        int(np.isfinite(score.localization_error))
        for item in comparisons
        for score in (item.ideal, item.attenuated, item.corrected)
    )
    expected = 3 * len(comparisons)
    message = (
        f"{recon.method.upper()} · {len(comparisons)} seed(s) · "
        f"mean recovery gain {mean_gain:+.3f} · "
        f"corrected background CV minus ideal {cv_change:+.3f}. "
        "Recovery compares tumor means. The CV change is noise, not accuracy. "
        + " ".join(per_seed)
        + ". "
        f"Localization error, all seeds: "
        f"ideal mean {_fmt_error(_nanmean([item.ideal.localization_error for item in comparisons]))} "
        f"median {_fmt_error(_nanmedian([item.ideal.localization_error for item in comparisons]))}; "
        f"attenuated mean {_fmt_error(_nanmean([item.attenuated.localization_error for item in comparisons]))} "
        f"median {_fmt_error(_nanmedian([item.attenuated.localization_error for item in comparisons]))}; "
        f"corrected mean {_fmt_error(_nanmean([item.corrected.localization_error for item in comparisons]))} "
        f"median {_fmt_error(_nanmedian([item.corrected.localization_error for item in comparisons]))}. "
        f"Estimates produced {produced}/{expected}. Large errors are kept."
    )
    return AttenuationStudyResult(
        mu_map=mu_map,
        method=recon.method,
        ideal_image=ideal_image,
        attenuated_image=attenuated_image,
        corrected_image=corrected_image,
        comparisons=comparisons,
        message=message,
    )
