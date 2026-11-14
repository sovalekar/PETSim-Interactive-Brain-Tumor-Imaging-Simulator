"""Background worker for Stage 6 image-quality experiments."""

from __future__ import annotations

import threading

from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.experiments.config import ExperimentConfig
from petsim.experiments.runner import ExperimentResult, run_experiment


class _ExperimentTask(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, config: ExperimentConfig) -> None:
        super().__init__()
        self._config = config
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def _on_progress(self, done: int, total: int, message: str) -> None:
        self.progress.emit(int(done), int(total), str(message))

    @Slot()
    def run(self) -> None:
        try:
            result = run_experiment(
                self._config,
                progress=self._on_progress,
                cancel=self._cancel.is_set,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class ExperimentRunner(QObject):
    """Owns one experiment thread and drops results after a reset."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: _ExperimentTask | None = None
        self._busy = False
        self._generation = 0
        self._active_job = 0

    @property
    def busy(self) -> bool:
        return self._busy

    def start(self, config: ExperimentConfig) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._generation += 1
        self._active_job = self._generation
        self._thread = QThread(self)
        self._task = _ExperimentTask(config)
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

    def _on_finished(self, result: ExperimentResult) -> None:
        if self._active_job != self._generation:
            return
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        if self._active_job != self._generation:
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
