"""Acquisition session buffer and live statistics (Stage 2).

Stores coincidence measurements for a future reconstruction stage without
building sinograms or performing FBP / OSEM / ML-EM here.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from petsim.physics.pet_engine import AnnihilationEvent


@dataclass
class AcquisitionConfig:
    """User-facing acquisition parameters from the control panel."""

    n_events: int = 5000
    detector_efficiency: float = 0.80
    animation_speed: str = "Normal"  # Slow | Normal | Fast
    seed: int | None = None
    use_seed: bool = False

    def clamp(self) -> AcquisitionConfig:
        speed = self.animation_speed if self.animation_speed in {"Slow", "Normal", "Fast"} else "Normal"
        return AcquisitionConfig(
            n_events=int(np.clip(self.n_events, 1000, 20_000)),
            detector_efficiency=float(np.clip(self.detector_efficiency, 0.20, 1.0)),
            animation_speed=speed,
            seed=self.seed,
            use_seed=bool(self.use_seed),
        )

    def events_per_tick(self) -> int:
        """How many annihilations to simulate on each QTimer tick."""
        return {"Slow": 1, "Normal": 4, "Fast": 25}.get(self.animation_speed, 4)

    def timer_interval_ms(self) -> int:
        """QTimer interval chosen so the UI stays responsive."""
        return {"Slow": 90, "Normal": 40, "Fast": 16}.get(self.animation_speed, 40)


@dataclass
class AcquisitionStats:
    """Live counters shown in the control panel."""

    total_events: int = 0
    coincidences: int = 0
    target_events: int = 0
    detector_efficiency: float = 0.0
    running: bool = False
    paused: bool = False
    finished: bool = False

    @property
    def coincidence_percent(self) -> float:
        if self.total_events <= 0:
            return 0.0
        return 100.0 * self.coincidences / self.total_events

    @property
    def progress_percent(self) -> float:
        if self.target_events <= 0:
            return 0.0
        return 100.0 * min(self.total_events, self.target_events) / self.target_events

    @property
    def expected_coincidence_percent(self) -> float:
        """Analytic ε² × 100 for comparison with the empirical rate."""
        return 100.0 * self.detector_efficiency * self.detector_efficiency


@dataclass
class AcquisitionBuffer:
    """Append-only store of simulated events (physics arrays + recent overlay).

    Array fields grow as acquisition progresses and are the Stage-3-ready
    coincidence measurements (detector pairs, paths, flags) — not images.
    """

    annihilations: list[tuple[float, float]] = field(default_factory=list)
    angles: list[float] = field(default_factory=list)
    hits_a: list[tuple[float, float]] = field(default_factory=list)
    hits_b: list[tuple[float, float]] = field(default_factory=list)
    detectors_a: list[int] = field(default_factory=list)
    detectors_b: list[int] = field(default_factory=list)
    detected_a: list[bool] = field(default_factory=list)
    detected_b: list[bool] = field(default_factory=list)
    coincidence_flags: list[bool] = field(default_factory=list)

    def clear(self) -> None:
        self.annihilations.clear()
        self.angles.clear()
        self.hits_a.clear()
        self.hits_b.clear()
        self.detectors_a.clear()
        self.detectors_b.clear()
        self.detected_a.clear()
        self.detected_b.clear()
        self.coincidence_flags.clear()

    def __len__(self) -> int:
        return len(self.annihilations)

    def extend(self, events: list[AnnihilationEvent]) -> int:
        """Append events; return how many were coincidences."""
        n_coin = 0
        for ev in events:
            self.annihilations.append(ev.annihilation_xy)
            self.angles.append(ev.angle)
            self.hits_a.append(ev.hit_a_xy)
            self.hits_b.append(ev.hit_b_xy)
            self.detectors_a.append(ev.detector_a)
            self.detectors_b.append(ev.detector_b)
            self.detected_a.append(ev.detected_a)
            self.detected_b.append(ev.detected_b)
            is_coin = ev.is_coincidence
            self.coincidence_flags.append(is_coin)
            if is_coin:
                n_coin += 1
        return n_coin

    def coincidence_pairs(self) -> np.ndarray:
        """Return ``(M, 2)`` detector index pairs for successful coincidences."""
        pairs = [
            (min(a, b), max(a, b))
            for a, b, ok in zip(
                self.detectors_a, self.detectors_b, self.coincidence_flags, strict=True
            )
            if ok
        ]
        if not pairs:
            return np.zeros((0, 2), dtype=np.int64)
        return np.asarray(pairs, dtype=np.int64)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """NumPy snapshot suitable for future reconstruction pipelines."""
        return {
            "annihilations": np.asarray(self.annihilations, dtype=np.float64).reshape(-1, 2),
            "angles": np.asarray(self.angles, dtype=np.float64),
            "hits_a": np.asarray(self.hits_a, dtype=np.float64).reshape(-1, 2),
            "hits_b": np.asarray(self.hits_b, dtype=np.float64).reshape(-1, 2),
            "detectors_a": np.asarray(self.detectors_a, dtype=np.int64),
            "detectors_b": np.asarray(self.detectors_b, dtype=np.int64),
            "detected_a": np.asarray(self.detected_a, dtype=bool),
            "detected_b": np.asarray(self.detected_b, dtype=bool),
            "coincidence": np.asarray(self.coincidence_flags, dtype=bool),
        }
