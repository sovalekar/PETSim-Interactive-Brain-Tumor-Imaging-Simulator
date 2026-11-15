"""Background threads for correction comparisons and studies."""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import numpy as np
from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.corrections.metrics import ImageQualityMetrics, score_reconstruction
from petsim.corrections.reconstruct import CORRECTION_MODES, MODE_IDEAL, reconstruct_corrected
from petsim.corrections.studies import (
    CorrectionStudyConfig,
    CorrectionStudyResult,
    run_correction_study,
)
from petsim.models.phantom import PhantomState
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult


@dataclass
class ComparisonBundle:
    images: dict[str, ReconstructionResult] = field(default_factory=dict)
    metrics: dict[str, ImageQualityMetrics] = field(default_factory=dict)
    cancelled: bool = False
    message: str = ""
    job_id: int = 0


class _ComparisonTask(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(
        self,
        ideal_a: np.ndarray,
        ideal_b: np.ndarray,
        degraded_a: np.ndarray,
        degraded_b: np.ndarray,
        config: ReconstructionConfig,
        *,
        n_annihilations: int,
        random_fraction: float,
        phantom: PhantomState | None,
        job_id: int,
        n_side_window: int = 0,
        scatter_probability: float = 0.0,
        photo_low_kev: float = 435.0,
        photo_high_kev: float = 650.0,
        energy_fwhm_percent: float = 12.0,
        scatter_angle_sigma_rad: float = 0.6,
    ) -> None:
        super().__init__()
        self._ideal_a = np.asarray(ideal_a, dtype=np.float64)
        self._ideal_b = np.asarray(ideal_b, dtype=np.float64)
        self._degraded_a = np.asarray(degraded_a, dtype=np.float64)
        self._degraded_b = np.asarray(degraded_b, dtype=np.float64)
        self._config = config
        self._n_annihilations = int(n_annihilations)
        self._random_fraction = float(random_fraction)
        self._phantom = phantom
        self._job_id = int(job_id)
        self._n_side_window = int(n_side_window)
        self._scatter_probability = float(scatter_probability)
        self._photo_low_kev = float(photo_low_kev)
        self._photo_high_kev = float(photo_high_kev)
        self._energy_fwhm_percent = float(energy_fwhm_percent)
        self._scatter_angle_sigma_rad = float(scatter_angle_sigma_rad)
        self._cancel = threading.Event()

    def request_cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            bundle = ComparisonBundle(job_id=self._job_id)
            for index, mode in enumerate(CORRECTION_MODES):
                if self._cancel.is_set():
                    bundle.cancelled = True
                    break
                hits_a, hits_b = self._degraded_a, self._degraded_b
                fraction = self._random_fraction
                scatter_probability = self._scatter_probability
                n_side_window = self._n_side_window
                if mode == MODE_IDEAL:
                    hits_a, hits_b = self._ideal_a, self._ideal_b
                    fraction = 0.0
                    scatter_probability = 0.0
                    n_side_window = 0
                self.progress.emit(index, len(CORRECTION_MODES), mode)
                result = reconstruct_corrected(
                    hits_a,
                    hits_b,
                    self._config,
                    mode=mode,
                    n_annihilations=self._n_annihilations,
                    random_fraction=fraction,
                    n_side_window=n_side_window,
                    scatter_probability=scatter_probability,
                    photo_low_kev=self._photo_low_kev,
                    photo_high_kev=self._photo_high_kev,
                    energy_fwhm_percent=self._energy_fwhm_percent,
                    scatter_angle_sigma_rad=self._scatter_angle_sigma_rad,
                    cancel=self._cancel.is_set,
                    job_id=self._job_id,
                )
                bundle.images[mode] = result
                bundle.metrics[mode] = score_reconstruction(result, self._phantom)
                if result.cancelled:
                    bundle.cancelled = True
                    break
            bundle.message = (
                "Comparison cancelled."
                if bundle.cancelled
                else (
                    "Comparison finished. Random correction changes the sinogram shape. "
                    "Scatter scaling is one fraction from the side-window count, so "
                    "NRMSE, CRC, and localization can stay unchanged."
                )
            )
            self.finished.emit(bundle)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _StudyTask(QObject):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def __init__(self, config: CorrectionStudyConfig) -> None:
        super().__init__()
        self._config = config
        self._cancel = threading.Event()
        self._job_id = 0

    def request_cancel(self) -> None:
        self._cancel.set()

    @Slot()
    def run(self) -> None:
        try:
            result = run_correction_study(
                self._config,
                progress=lambda done, total, message: self.progress.emit(done, total, message),
                cancel=self._cancel.is_set,
            )
            result.job_id = int(self._job_id)
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001
            self.failed.emit(str(exc))


class _ThreadOwner(QObject):
    """Shared start / cancel / reset behavior for one worker thread."""

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: QObject | None = None
        self._busy = False
        self._generation = 0

    @property
    def busy(self) -> bool:
        return self._busy

    def cancel(self) -> None:
        task = self._task
        if task is not None and hasattr(task, "request_cancel"):
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
            if not thread.wait(timeout_ms):
                thread.terminate()
                thread.wait(1000)
                self._busy = False
                return False
        self._busy = False
        return True

    def _launch(self, task: QObject) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._generation += 1
        if hasattr(task, "_job_id"):
            task._job_id = self._generation
        self._task = task
        self._thread = QThread(self)
        task.moveToThread(self._thread)
        self._thread.started.connect(task.run)  # type: ignore[attr-defined]
        task.finished.connect(self._on_finished)  # type: ignore[attr-defined]
        task.failed.connect(self._on_failed)  # type: ignore[attr-defined]
        task.progress.connect(self.progress)  # type: ignore[attr-defined]
        task.finished.connect(self._thread.quit)  # type: ignore[attr-defined]
        task.failed.connect(self._thread.quit)  # type: ignore[attr-defined]
        self._thread.finished.connect(self._cleanup)
        self._thread.start()
        return True

    def _accepts_bundle(self, payload: object) -> bool:
        job_id = getattr(payload, "job_id", None)
        if job_id is None:
            return True
        return int(job_id) == self._generation

    def _on_finished(self, payload: object) -> None:
        if not self._accepts_bundle(payload):
            return
        self.finished.emit(payload)  # type: ignore[attr-defined]

    def _on_failed(self, message: str) -> None:
        task = self._task
        job_id = getattr(task, "_job_id", None)
        if job_id is not None and int(job_id) != self._generation:
            return
        self.failed.emit(message)  # type: ignore[attr-defined]

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


class CorrectionComparisonRunner(_ThreadOwner):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def start(
        self,
        ideal_a: np.ndarray,
        ideal_b: np.ndarray,
        degraded_a: np.ndarray,
        degraded_b: np.ndarray,
        config: ReconstructionConfig,
        *,
        n_annihilations: int,
        random_fraction: float,
        phantom: PhantomState | None,
        n_side_window: int = 0,
        scatter_probability: float = 0.0,
        photo_low_kev: float = 435.0,
        photo_high_kev: float = 650.0,
        energy_fwhm_percent: float = 12.0,
        scatter_angle_sigma_rad: float = 0.6,
    ) -> bool:
        task = _ComparisonTask(
            ideal_a,
            ideal_b,
            degraded_a,
            degraded_b,
            config,
            n_annihilations=n_annihilations,
            random_fraction=random_fraction,
            phantom=phantom,
            job_id=0,
            n_side_window=n_side_window,
            scatter_probability=scatter_probability,
            photo_low_kev=photo_low_kev,
            photo_high_kev=photo_high_kev,
            energy_fwhm_percent=energy_fwhm_percent,
            scatter_angle_sigma_rad=scatter_angle_sigma_rad,
        )
        return self._launch(task)


class CorrectionStudyRunner(_ThreadOwner):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, int, str)

    def start(self, config: CorrectionStudyConfig) -> bool:
        return self._launch(_StudyTask(config))
