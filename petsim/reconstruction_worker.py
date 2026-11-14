"""Background QThread worker so FBP and MLEM do not freeze the GUI."""

from __future__ import annotations

import threading

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
    pair_finished = Signal(object, object)  # FBP result, MLEM result
    failed = Signal(str)
    progress = Signal(int, int, int, float, float)  # job, iteration, total, log L, rel

    def __init__(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
        *,
        job_id: int,
        pair: bool,
        mlem_config: ReconstructionConfig | None = None,
    ) -> None:
        super().__init__()
        self._hits_a = np.asarray(hits_a, dtype=np.float64)
        self._hits_b = np.asarray(hits_b, dtype=np.float64)
        self._config = config
        self._mlem_config = mlem_config
        self._job_id = int(job_id)
        self._pair = bool(pair)
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    def _progress(self, iteration: int, total: int, loglik: float, rel: float) -> None:
        self.progress.emit(
            self._job_id, int(iteration), int(total), float(loglik), float(rel)
        )

    @Slot()
    def run(self) -> None:
        try:
            if self._pair:
                fbp_cfg = self._config.as_method("fbp")
                mlem_cfg = (self._mlem_config or self._config).as_method("mlem")
                fbp = reconstruct_from_endpoints(
                    self._hits_a,
                    self._hits_b,
                    fbp_cfg,
                    job_id=self._job_id,
                )
                if self._cancel.is_set():
                    fbp.cancelled = True
                    fbp.message = "Comparison cancelled before MLEM."
                    empty = reconstruct_from_endpoints(
                        self._hits_a[:0],
                        self._hits_b[:0],
                        mlem_cfg,
                        job_id=self._job_id,
                    )
                    empty.cancelled = True
                    empty.message = "MLEM cancelled before it started."
                    self.pair_finished.emit(fbp, empty)
                    return
                mlem = reconstruct_from_endpoints(
                    self._hits_a,
                    self._hits_b,
                    mlem_cfg,
                    progress=self._progress,
                    cancel=self._cancel.is_set,
                    job_id=self._job_id,
                )
                self.pair_finished.emit(fbp, mlem)
                return

            result = reconstruct_from_endpoints(
                self._hits_a,
                self._hits_b,
                self._config,
                progress=self._progress,
                cancel=self._cancel.is_set,
                job_id=self._job_id,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — surface any recon failure to UI
            self.failed.emit(str(exc))


class ReconstructionRunner(QObject):
    """Owns a short-lived QThread for one reconstruction job."""

    finished = Signal(object)
    pair_finished = Signal(object, object)
    failed = Signal(str)
    started = Signal()
    progress = Signal(int, int, int, float, float)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: _ReconTask | None = None
        self._busy = False
        self._generation = 0

    @property
    def busy(self) -> bool:
        return self._busy

    @property
    def generation(self) -> int:
        return self._generation

    def start(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
        *,
        pair: bool = False,
        mlem_config: ReconstructionConfig | None = None,
    ) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._generation += 1
        job_id = self._generation
        self._thread = QThread(self)
        self._task = _ReconTask(
            hits_a,
            hits_b,
            config,
            job_id=job_id,
            pair=pair,
            mlem_config=mlem_config,
        )
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._task.finished.connect(self._on_finished)
        self._task.pair_finished.connect(self._on_pair_finished)
        self._task.failed.connect(self._on_failed)
        self._task.progress.connect(self.progress)
        self._task.finished.connect(self._thread.quit)
        self._task.pair_finished.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self.started.emit()
        self._thread.start()
        return True

    def cancel(self) -> None:
        """Ask the running job to stop between MLEM iterations."""
        task = self._task
        if task is not None:
            task.request_cancel()

    def invalidate_pending(self) -> None:
        """Ignore a result that is already running or about to be delivered."""
        self._generation += 1
        self.cancel()

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        """Block until any worker thread finishes (safe for window close / tests).

        On macOS, destroying a ``MainWindow`` while a ``QThread`` is still
        running is a common hard crash during pytest teardown.
        """
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

    def accepts(self, job_id: int) -> bool:
        return int(job_id) == self._generation

    def _on_finished(self, result: ReconstructionResult) -> None:
        if not self.accepts(result.job_id):
            return
        self.finished.emit(result)

    def _on_pair_finished(self, fbp: ReconstructionResult, mlem: ReconstructionResult) -> None:
        if not self.accepts(fbp.job_id):
            return
        self.pair_finished.emit(fbp, mlem)

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
