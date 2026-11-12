"""Settings for one image-quality experiment."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.models.phantom import PhantomState

DEFAULT_COUNT_TARGETS: tuple[int, ...] = (1000, 3000, 5000, 10000, 20000)


def trial_seed(base_seed: int, count_index: int, trial_index: int) -> int:
    """Deterministic 32-bit seed for one acquisition.

    The seed depends on the base seed, count level, and trial index. It does
    not depend on the reconstruction method, so FBP and MLEM compared at the
    same trial share one coincidence list.
    """
    mixed = (
        int(base_seed) * 1_000_003
        + int(count_index) * 97_031
        + int(trial_index) * 1_009
        + 17
    )
    return int(mixed & 0xFFFFFFFF)


@dataclass
class ExperimentConfig:
    """Fixed geometry and reconstruction settings for every trial."""

    phantom: PhantomState
    methods: tuple[str, ...] = ("fbp", "mlem")
    count_targets: tuple[int, ...] = DEFAULT_COUNT_TARGETS
    n_trials: int = 3
    seed: int = 1
    n_iterations: int = 10
    filter_name: str = "hann"
    detector_efficiency: float = 0.80
    image_size: int = 128
    n_theta: int = 96
    n_s: int = 128
    fov: float = 1.0
    s_max: float = 0.92
    smooth_sigma_s: float = 1.5
    smooth_sigma_theta: float = 0.75
    n_detectors: int = 48

    def clamp(self) -> ExperimentConfig:
        methods = tuple(
            dict(fbp="fbp", mlem="mlem").get(m.lower().strip(), "")
            for m in self.methods
        )
        methods = tuple(m for m in methods if m)
        # Preserve order, drop duplicates.
        seen: list[str] = []
        for method in methods:
            if method not in seen:
                seen.append(method)
        counts = tuple(
            sorted({int(c) for c in self.count_targets if int(c) > 0})
        )
        filt = self.filter_name.lower().strip()
        filt = "hann" if filt in {"hann", "hanning"} else "ramp"
        return ExperimentConfig(
            phantom=self.phantom.clamp(),
            methods=tuple(seen),
            count_targets=counts,
            n_trials=int(np.clip(self.n_trials, 1, 50)),
            seed=int(self.seed) & 0xFFFFFFFF,
            n_iterations=int(np.clip(self.n_iterations, 1, 200)),
            filter_name=filt,
            detector_efficiency=float(np.clip(self.detector_efficiency, 0.05, 1.0)),
            image_size=int(np.clip(self.image_size, 16, 256)),
            n_theta=int(np.clip(self.n_theta, 8, 360)),
            n_s=int(np.clip(self.n_s, 8, 256)),
            fov=float(max(self.fov, 1e-3)),
            s_max=float(max(abs(self.s_max), 1e-3)),
            smooth_sigma_s=float(max(self.smooth_sigma_s, 0.0)),
            smooth_sigma_theta=float(max(self.smooth_sigma_theta, 0.0)),
            n_detectors=int(np.clip(self.n_detectors, 8, 256)),
        )
