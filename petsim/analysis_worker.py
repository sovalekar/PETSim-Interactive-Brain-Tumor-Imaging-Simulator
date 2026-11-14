"""Background workers for Stage 4 quantitative analysis."""

from __future__ import annotations

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.analysis.count_comparison import (
    DEFAULT_COUNT_LEVELS,
    CountComparisonResult,
    compare_by_coincidence_count,
)
from petsim.analysis.pipeline import QuantitativeAnalysisResult, analyze_reconstruction
from petsim.models.phantom import PhantomState
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult


class _AnalysisTask(QObject):
    finished = Signal(object)  # QuantitativeAnalysisResult
    failed = Signal(str)

    def __init__(
        self,
        result: ReconstructionResult,
        phantom_state: PhantomState | None,
    ) -> None:
        super().__init__()
        self._result = result
        self._phantom_state = phantom_state

    @Slot()
    def run(self) -> None:
        try:
            out = analyze_reconstruction(self._result, self._phantom_state)
            self.finished.emit(out)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _CountComparisonTask(QObject):
    finished = Signal(object)  # CountComparisonResult
    failed = Signal(str)

    def __init__(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
        phantom_state: PhantomState | None,
        target_counts: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._hits_a = np.asarray(hits_a, dtype=np.float64)
        self._hits_b = np.asarray(hits_b, dtype=np.float64)
        self._config = config
        self._phantom_state = phantom_state
        self._target_counts = target_counts

    @Slot()
    def run(self) -> None:
        try:
            out = compare_by_coincidence_count(
                self._hits_a,
                self._hits_b,
                self._config,
                self._phantom_state,
                target_counts=self._target_counts,
            )
            self.finished.emit(out)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class AnalysisRunner(QObject):
    """Owns short-lived QThreads for analysis / count-comparison jobs."""

    analysis_finished = Signal(object)
    analysis_failed = Signal(str)
    comparison_finished = Signal(object)
    comparison_failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: QObject | None = None
        self._busy = False

    @property
    def busy(self) -> bool:
        return self._busy

    def start_analysis(
        self,
        result: ReconstructionResult,
        phantom_state: PhantomState | None,
    ) -> bool:
        if self._busy:
            return False
        return self._launch(_AnalysisTask(result, phantom_state), kind="analysis")

    def start_count_comparison(
        self,
        hits_a: np.ndarray,
        hits_b: np.ndarray,
        config: ReconstructionConfig,
        phantom_state: PhantomState | None,
        *,
        target_counts: tuple[int, ...] = DEFAULT_COUNT_LEVELS,
    ) -> bool:
        if self._busy:
            return False
        task = _CountComparisonTask(
            hits_a, hits_b, config, phantom_state, target_counts
        )
        return self._launch(task, kind="comparison")

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
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

    def _launch(self, task: QObject, *, kind: str) -> bool:
        self._busy = True
        self._thread = QThread(self)
        self._task = task
        task.moveToThread(self._thread)
        self._thread.started.connect(task.run)  # type: ignore[attr-defined]
        if kind == "analysis":
            task.finished.connect(self._on_analysis_finished)  # type: ignore[attr-defined]
            task.failed.connect(self._on_analysis_failed)  # type: ignore[attr-defined]
        else:
            task.finished.connect(self._on_comparison_finished)  # type: ignore[attr-defined]
            task.failed.connect(self._on_comparison_failed)  # type: ignore[attr-defined]
        task.finished.connect(self._thread.quit)  # type: ignore[attr-defined]
        task.failed.connect(self._thread.quit)  # type: ignore[attr-defined]
        self._thread.finished.connect(self._cleanup)
        self._thread.start()
        return True

    def _on_analysis_finished(self, result: QuantitativeAnalysisResult) -> None:
        self.analysis_finished.emit(result)

    def _on_analysis_failed(self, message: str) -> None:
        self.analysis_failed.emit(message)

    def _on_comparison_finished(self, result: CountComparisonResult) -> None:
        self.comparison_finished.emit(result)

    def _on_comparison_failed(self, message: str) -> None:
        self.comparison_failed.emit(message)

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
