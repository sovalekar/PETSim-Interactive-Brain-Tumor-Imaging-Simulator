"""Acquisition session buffer and live statistics (Stage 2).

Stores coincidence measurements used by Stage 3 sinogram / FBP reconstruction.
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
    effects_enabled: bool = False
    detected_singles: int = 0
    undetected: int = 0
    accepted_unscattered: int = 0
    scattered_coincidences: int = 0
    random_coincidences: int = 0
    energy_rejected: int = 0
    timing_rejected: int = 0

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
    ideal_hits_a: list[tuple[float, float]] = field(default_factory=list)
    ideal_hits_b: list[tuple[float, float]] = field(default_factory=list)
    energy_a_kev: list[float] = field(default_factory=list)
    energy_b_kev: list[float] = field(default_factory=list)
    passed_timing: list[bool] = field(default_factory=list)
    measured_delta_t_ps: list[float] = field(default_factory=list)
    is_random: list[bool] = field(default_factory=list)
    is_scattered: list[bool] = field(default_factory=list)

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
        self.ideal_hits_a.clear()
        self.ideal_hits_b.clear()
        self.energy_a_kev.clear()
        self.energy_b_kev.clear()
        self.passed_timing.clear()
        self.measured_delta_t_ps.clear()
        self.is_random.clear()
        self.is_scattered.clear()

    def __len__(self) -> int:
        return len(self.annihilations)

    def n_annihilations(self) -> int:
        """Generated annihilations, excluding appended random coincidences."""
        if not self.is_random:
            return len(self.annihilations)
        return sum(1 for flag in self.is_random if not flag)

    def extend(self, events: list[AnnihilationEvent]) -> int:
        """Append events; return how many were accepted coincidences."""
        n_coin = 0
        for ev in events:
            ideal_a, ideal_b = ev.ideal_endpoints
            self.annihilations.append(ev.annihilation_xy)
            self.angles.append(ev.angle)
            self.hits_a.append(ev.hit_a_xy)
            self.hits_b.append(ev.hit_b_xy)
            self.detectors_a.append(ev.detector_a)
            self.detectors_b.append(ev.detector_b)
            self.detected_a.append(ev.detected_a)
            self.detected_b.append(ev.detected_b)
            self.ideal_hits_a.append(ideal_a)
            self.ideal_hits_b.append(ideal_b)
            self.energy_a_kev.append(float(ev.energy_a_kev))
            self.energy_b_kev.append(float(ev.energy_b_kev))
            self.passed_timing.append(bool(ev.passed_timing))
            self.measured_delta_t_ps.append(float(ev.measured_delta_t_ps))
            self.is_random.append(bool(ev.is_random))
            self.is_scattered.append(bool(ev.is_scattered))
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

    def coincidence_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Return measured ring-hit endpoints ``(hits_a, hits_b)`` for coincidences.

        These are the Stage 3 reconstruction inputs. True annihilation
        coordinates are intentionally omitted.
        """
        pts_a = [
            ha
            for ha, ok in zip(self.hits_a, self.coincidence_flags, strict=True)
            if ok
        ]
        pts_b = [
            hb
            for hb, ok in zip(self.hits_b, self.coincidence_flags, strict=True)
            if ok
        ]
        if not pts_a:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
            )
        return (
            np.asarray(pts_a, dtype=np.float64),
            np.asarray(pts_b, dtype=np.float64),
        )

    def true_coincidence_tof(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """True-coincidence endpoints and measured ``Δt``.

        Random records are omitted. They have no annihilation time, and TOF-MLEM
        does not treat a flat clock as a true TOF kernel.
        """
        pts_a = []
        pts_b = []
        times = []
        for ha, hb, ok, is_random, delta_t in zip(
            self.hits_a,
            self.hits_b,
            self.coincidence_flags,
            self.is_random,
            self.measured_delta_t_ps,
            strict=True,
        ):
            if not ok or is_random:
                continue
            pts_a.append(ha)
            pts_b.append(hb)
            times.append(delta_t)
        if not pts_a:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
                np.zeros(0, dtype=np.float64),
            )
        return (
            np.asarray(pts_a, dtype=np.float64),
            np.asarray(pts_b, dtype=np.float64),
            np.asarray(times, dtype=np.float64),
        )

    def ideal_coincidence_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Ring hits before scatter, for efficiency-detected non-random pairs.

        These are the ideal-detector measurements of the same annihilations.
        Accepted random coincidences are omitted. Annihilation coordinates are
        not returned.
        """
        pts_a = []
        pts_b = []
        for ideal_a, ideal_b, det_a, det_b, is_random in zip(
            self.ideal_hits_a,
            self.ideal_hits_b,
            self.detected_a,
            self.detected_b,
            self.is_random,
            strict=True,
        ):
            if is_random or not (det_a and det_b):
                continue
            pts_a.append(ideal_a)
            pts_b.append(ideal_b)
        if not pts_a:
            return (
                np.zeros((0, 2), dtype=np.float64),
                np.zeros((0, 2), dtype=np.float64),
            )
        return (
            np.asarray(pts_a, dtype=np.float64),
            np.asarray(pts_b, dtype=np.float64),
        )

    def side_window_count(
        self,
        photo_low_kev: float,
        photo_high_kev: float,
        *,
        limit: int | None = None,
    ) -> int:
        """Witness pairs for scatter transfer. Event labels are not read."""
        from petsim.corrections.energy import count_side_window_coincidences

        return count_side_window_coincidences(
            self.energy_a_kev,
            self.energy_b_kev,
            self.detected_a,
            self.detected_b,
            self.passed_timing,
            photo_low_kev=photo_low_kev,
            photo_high_kev=photo_high_kev,
            limit=limit,
        )

    def measured_photon_energies_kev(self) -> np.ndarray:
        """Measured energies of annihilation photons, excluding random records."""
        values = [
            energy
            for energy, is_random in zip(
                [*self.energy_a_kev, *self.energy_b_kev],
                [*self.is_random, *self.is_random],
                strict=True,
            )
            if not is_random
        ]
        if not values:
            return np.zeros(0, dtype=np.float64)
        return np.asarray(values, dtype=np.float64)

    def as_arrays(self) -> dict[str, np.ndarray]:
        """NumPy snapshot suitable for reconstruction pipelines."""
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
