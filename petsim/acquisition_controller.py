"""QTimer-driven acquisition loop that keeps the UI responsive.

The controller owns the PET engine and buffer. Widgets only receive signals
with event batches / stats — they never call NumPy physics directly on the
GUI thread beyond consuming already-simulated results.
"""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QTimer, Signal

from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.acquisition import (
    AcquisitionBuffer,
    AcquisitionConfig,
    AcquisitionStats,
)
from petsim.physics.detector_effects import DetectorPhysicsConfig, account_events
from petsim.physics.pet_engine import AnnihilationEvent, PETEngine


class AcquisitionController(QObject):
    """Start / pause / reset acquisition and emit animation-friendly batches."""

    # list[AnnihilationEvent] for the detector overlay
    events_simulated = Signal(object)
    stats_changed = Signal(object)  # AcquisitionStats
    finished = Signal()
    reset_done = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self.phantom = BrainPhantom()
        self.ring = DetectorRing(n_detectors=48)
        self.engine = PETEngine(self.phantom, self.ring)
        self.buffer = AcquisitionBuffer()
        self.config = AcquisitionConfig().clamp()
        self.physics = DetectorPhysicsConfig()
        # Frozen copy used by the acquisition currently in the buffer.
        self._active_physics: DetectorPhysicsConfig | None = None
        # Snapshot of the uptake map used for the current acquisition (Stage 3
        # comparison). Cleared on reset; never used as a reconstruction input.
        self.acquired_activity: np.ndarray | None = None
        self.acquired_phantom_state: PhantomState | None = None

        self._stats = AcquisitionStats()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._on_tick)

    @property
    def stats(self) -> AcquisitionStats:
        return self._stats

    def set_phantom_state(self, state: PhantomState) -> None:
        """Update uptake map. Resets acquisition so configs are never mixed."""
        self.phantom.state = state
        self.engine.invalidate_activity_cache()
        self.reset()

    def apply_config(self, config: AcquisitionConfig) -> None:
        """Update parameters. Efficiency applies to future ticks while running."""
        self.config = config.clamp()
        self._stats.detector_efficiency = self.config.detector_efficiency
        self._stats.target_events = self.config.n_events
        if self._stats.running:
            self._timer.setInterval(self.config.timer_interval_ms())
        # If the target shrank below progress, stop cleanly.
        if self._stats.total_events >= self.config.n_events and self._stats.running:
            self._complete()
        else:
            self._emit_stats()

    def apply_detector_physics(self, config: DetectorPhysicsConfig) -> None:
        """Store settings for the next acquisition.

        Events already in the buffer keep the physics snapshot taken when
        that acquisition started. Slider changes do not rewrite them.
        """
        self.physics = config.clamp()
        if self._active_physics is None and self._stats.total_events == 0:
            self._stats.effects_enabled = self.physics.enabled
            if not self._stats.running:
                self._emit_stats()

    def start(self) -> None:
        cfg = self.config.clamp()
        self.config = cfg

        if self._stats.finished or self._stats.total_events >= cfg.n_events:
            # Fresh run when restarting after completion.
            self._clear_progress()

        if cfg.use_seed and self._stats.total_events == 0:
            self.engine.set_seed(cfg.seed)
        elif self._stats.total_events == 0 and not cfg.use_seed:
            # Fresh non-deterministic RNG for each new acquisition.
            self.engine.set_seed(None)

        if self._stats.total_events == 0:
            # Freeze the activity map and detector settings for this dataset.
            self.acquired_activity = self.phantom.rasterize().copy()
            self.acquired_phantom_state = self.phantom.state.clamp()
            self._active_physics = self.physics.clamp()
            self._stats.effects_enabled = self._active_physics.enabled

        self._stats.running = True
        self._stats.paused = False
        self._stats.finished = False
        self._stats.target_events = cfg.n_events
        self._stats.detector_efficiency = cfg.detector_efficiency
        self._timer.start(cfg.timer_interval_ms())
        self._emit_stats()

    def pause(self) -> None:
        if not self._stats.running or self._stats.finished:
            return
        self._stats.paused = True
        self._stats.running = False
        self._timer.stop()
        self._emit_stats()

    def reset(self) -> None:
        self._timer.stop()
        self._clear_progress()
        self._emit_stats()
        self.reset_done.emit()

    def _clear_progress(self) -> None:
        self.buffer.clear()
        self.acquired_activity = None
        self.acquired_phantom_state = None
        self._active_physics = None
        self._stats = AcquisitionStats(
            total_events=0,
            coincidences=0,
            target_events=self.config.n_events,
            detector_efficiency=self.config.detector_efficiency,
            running=False,
            paused=False,
            finished=False,
            effects_enabled=self.physics.enabled,
        )

    def coincidence_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Accepted measured endpoints. Annihilation coordinates are omitted."""
        return self.buffer.coincidence_endpoints()

    def ideal_coincidence_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Ideal-detector endpoints for the same annihilations."""
        return self.buffer.ideal_coincidence_endpoints()

    def measured_photon_energies_kev(self) -> np.ndarray:
        return self.buffer.measured_photon_energies_kev()

    def annihilation_count(self) -> int:
        """Annihilations stored for the current acquisition, excluding random records."""
        return self.buffer.n_annihilations()

    def side_window_count(self) -> int:
        """Lower-window witness pairs for the active photopeak, or 0 if effects are off."""
        physics = self.active_detector_physics()
        if not physics.enabled:
            return 0
        return self.buffer.side_window_count(
            physics.energy_window_low_kev,
            physics.energy_window_high_kev,
        )

    def active_detector_physics(self) -> DetectorPhysicsConfig:
        """Physics snapshot for the current buffer, or disabled if none was started."""
        if self._active_physics is not None:
            return self._active_physics
        return DetectorPhysicsConfig(enabled=False)

    def _on_tick(self) -> None:
        if self._stats.paused or not self._stats.running:
            return

        remaining = self.config.n_events - self._stats.total_events
        if remaining <= 0:
            self._complete()
            return

        batch_n = min(self.config.events_per_tick(), remaining)
        physics = self._active_physics if self._active_physics is not None else self.physics
        events: list[AnnihilationEvent] = self.engine.simulate_batch(
            batch_n,
            detector_efficiency=self.config.detector_efficiency,
            physics=physics,
        )
        n_coin = self.buffer.extend(events)
        accounted = account_events(events)
        self._stats.total_events += accounted.annihilations
        self._stats.coincidences += n_coin
        self._stats.detected_singles += accounted.detected_singles
        self._stats.undetected += accounted.undetected
        self._stats.accepted_unscattered += accounted.accepted_unscattered
        self._stats.scattered_coincidences += accounted.accepted_scattered
        self._stats.random_coincidences += accounted.accepted_random
        self._stats.energy_rejected += accounted.energy_rejected
        self._stats.timing_rejected += accounted.timing_rejected
        self._stats.effects_enabled = physics.enabled
        self._stats.detector_efficiency = self.config.detector_efficiency
        self._stats.target_events = self.config.n_events

        self.events_simulated.emit(events)
        self._emit_stats()

        if self._stats.total_events >= self.config.n_events:
            self._complete()

    def _complete(self) -> None:
        self._timer.stop()
        self._stats.running = False
        self._stats.paused = False
        self._stats.finished = True
        self._emit_stats()
        self.finished.emit()

    def _emit_stats(self) -> None:
        self.stats_changed.emit(self._stats)
