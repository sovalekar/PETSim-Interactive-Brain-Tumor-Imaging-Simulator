"""Background QThread worker so FBP does not freeze the GUI."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.reconstruction.pipeline import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_from_endpoints,
)


class _ReconTask(QObject):
    """Runs inside a worker thread."""

    finished = Signal(object)  # ReconstructionResult
    failed = Signal(str)

    def __init__(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
    ) -> None:
        super().__init__()
        self._hits_a = np.asarray(hits_a, dtype=np.float64)
        self._hits_b = np.asarray(hits_b, dtype=np.float64)
        self._config = config

    @Slot()
    def run(self) -> None:
        try:
            result = reconstruct_from_endpoints(
                self._hits_a, self._hits_b, self._config
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface any recon failure to UI
            self.failed.emit(str(exc))


class ReconstructionRunner(QObject):
    """Owns a short-lived QThread for one reconstruction job."""

    finished = Signal(object)
    failed = Signal(str)
    started = Signal()

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: _ReconTask | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
    ) -> bool:
        if self._busy:
            return False
        self._busy = True
        # Parent the thread to this runner so lifetimes stay tied together.
        self._thread = QThread(self)
        self._task = _ReconTask(hits_a, hits_b, config)
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._task.finished.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self.started.emit()
        self._thread.start()
        return True

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        """Block until any worker thread finishes (safe for window close / tests).

        On macOS, destroying a ``MainWindow`` while a ``QThread`` is still
        running is a common hard crash during pytest teardown.
        """
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

    def _on_finished(self, result: ReconstructionResult) -> None:
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
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
