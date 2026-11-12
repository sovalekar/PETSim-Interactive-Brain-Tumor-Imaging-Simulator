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
            # Freeze the activity map that belongs with this dataset.
            self.acquired_activity = self.phantom.rasterize().copy()
            self.acquired_phantom_state = self.phantom.state.clamp()

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
        self._stats = AcquisitionStats(
            total_events=0,
            coincidences=0,
            target_events=self.config.n_events,
            detector_efficiency=self.config.detector_efficiency,
            running=False,
            paused=False,
            finished=False,
        )

    def coincidence_endpoints(self) -> tuple[np.ndarray, np.ndarray]:
        """Measured LOR endpoints for Stage 3 (no annihilation truth)."""
        return self.buffer.coincidence_endpoints()

    def _on_tick(self) -> None:
        if self._stats.paused or not self._stats.running:
            return

        remaining = self.config.n_events - self._stats.total_events
        if remaining <= 0:
            self._complete()
            return

        batch_n = min(self.config.events_per_tick(), remaining)
        events: list[AnnihilationEvent] = self.engine.simulate_batch(
            batch_n,
            detector_efficiency=self.config.detector_efficiency,
        )
        n_coin = self.buffer.extend(events)
        self._stats.total_events += len(events)
        self._stats.coincidences += n_coin
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
