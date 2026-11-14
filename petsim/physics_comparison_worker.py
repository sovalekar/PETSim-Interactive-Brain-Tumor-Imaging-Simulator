"""Background ideal-versus-degraded reconstruction for the physics tab."""

from __future__ import annotations

import threading
from dataclasses import dataclass

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.physics.comparison import reconstruct_ideal_and_degraded
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult


@dataclass
class ConfigurationComparison:
    ideal: ReconstructionResult
    degraded: ReconstructionResult
    job_id: int


class _ComparisonTask(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, int, float, float)

    def __init__(
        self,
        ideal_a: np.ndarray,
        ideal_b: np.ndarray,
        degraded_a: np.ndarray,
        degraded_b: np.ndarray,
        config: ReconstructionConfig,
        *,
        job_id: int,
    ) -> None:
        super().__init__()
        self._ideal_a = np.asarray(ideal_a, dtype=np.float64)
        self._ideal_b = np.asarray(ideal_b, dtype=np.float64)
        self._degraded_a = np.asarray(degraded_a, dtype=np.float64)
        self._degraded_b = np.asarray(degraded_b, dtype=np.float64)
        self._config = config
        self._job_id = int(job_id)
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def _progress(self, iteration: int, total: int, loglik: float, rel: float) -> None:
        self.progress.emit(self._job_id, int(iteration), int(total), float(loglik), float(rel))

    @Slot()
    def run(self) -> None:
        try:
            ideal, degraded = reconstruct_ideal_and_degraded(
                self._ideal_a,
                self._ideal_b,
                self._degraded_a,
                self._degraded_b,
                self._config,
                progress=self._progress,
                cancel=self._cancel.is_set,
                job_id=self._job_id,
            )
            self.finished.emit(
                ConfigurationComparison(ideal=ideal, degraded=degraded, job_id=self._job_id)
            )
        except Exception as exc:  # noqa: BLE001 — show reconstruction failures in the panel
            self.failed.emit(str(exc))


class PhysicsComparisonRunner(QObject):
    """One short-lived thread for the ideal versus degraded image pair."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, int, float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__()
        self._thread: QThread | None = None
        self._task: _ComparisonTask | None = None
        self._busy = False
        self._generation = 0

    @property
    def busy(self) -> bool:
        return self._busy

    def start(
        self,
        ideal_a: np.ndarray,
        ideal_b: np.ndarray,
        degraded_a: np.ndarray,
        degraded_b: np.ndarray,
        config: ReconstructionConfig,
    ) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._generation += 1
        job_id = self._generation
        self._thread = QThread(self)
        self._task = _ComparisonTask(
            ideal_a,
            ideal_b,
            degraded_a,
            degraded_b,
            config,
            job_id=job_id,
        )
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._task.progress.connect(self.progress)
        self._task.finished.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()
        return True

    def cancel(self) -> None:
        task = self._task
        if task is not None:
            task.request_cancel()

    def invalidate_pending(self) -> None:
        self._generation += 1
        self.cancel()

    def accepts(self, job_id: int) -> bool:
        return int(job_id) == self._generation

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        self.cancel()
        thread = self._thread
        if thread is None:
            self._busy = False
            return True
        if thread.isRunning():
            thread.quit()
            finished = thread.wait(timeout_ms)
            if not finished:
                thread.terminate()
                thread.wait(1000)
                self._busy = False
                return False
        self._busy = False
        return True

    def _on_finished(self, result: ConfigurationComparison) -> None:
        if not self.accepts(result.job_id):
            return
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        task = self._task
        job_id = getattr(task, "_job_id", None)
        if job_id is not None and not self.accepts(job_id):
            return
        self.failed.emit(message)

    def _cleanup(self) -> None:
        task = self._task
        thread = self._thread
        self._task = None
        self._thread = None
        self._busy = False
        if task is not None:
            task.deleteLater()
        if thread is not None:
            thread.deleteLater()
