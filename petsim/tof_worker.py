"""Background thread for the matched TOF comparison."""

from __future__ import annotations

from PySide6.QtCore import QObject, QThread, Signal, Slot

from petsim.models.phantom import PhantomState
from petsim.physics.attenuation import AttenuationConfig
from petsim.physics.tof import TOFConfig
from petsim.physics.tof_study import run_tof_study


class _StudyTask(QObject):
    finished = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        phantom: PhantomState,
        tof: TOFConfig,
        attenuation: AttenuationConfig | None,
        seeds: tuple[int, ...],
    ) -> None:
        super().__init__()
        self._phantom = phantom
        self._tof = tof
        self._attenuation = attenuation
        self._seeds = seeds

    @Slot()
    def run(self) -> None:
        try:
            result = run_tof_study(
                self._phantom,
                self._tof,
                attenuation=self._attenuation,
                seeds=self._seeds,
                n_events=700,
                n_iterations=5,
                image_size=40,
                n_theta=24,
                n_s=32,
            )
            self.finished.emit(result)
        except Exception as exc:  # noqa: BLE001 — show the failure in the status line
            self.failed.emit(str(exc))


class TOFStudyRunner(QObject):
    """One comparison at a time, off the GUI thread."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, parent: QObject | None = None) -> None:
        super().__init__(parent)
        self._thread: QThread | None = None
        self._task: _StudyTask | None = None
        self._busy = False

    def start(
        self,
        phantom: PhantomState,
        tof: TOFConfig,
        attenuation: AttenuationConfig | None,
        *,
        seeds: tuple[int, ...],
    ) -> bool:
        if self._busy:
            return False
        self._busy = True
        self._thread = QThread(self)
        self._task = _StudyTask(phantom, tof, attenuation, seeds)
        self._task.moveToThread(self._thread)
        self._thread.started.connect(self._task.run)
        self._task.finished.connect(self._on_finished)
        self._task.failed.connect(self._on_failed)
        self._task.finished.connect(self._thread.quit)
        self._task.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._cleanup)
        self._thread.start()
        return True

    def shutdown(self, timeout_ms: int = 30_000) -> bool:
        thread = self._thread
        if thread is None or not thread.isRunning():
            return True
        thread.quit()
        return bool(thread.wait(timeout_ms))

    def _on_finished(self, result: object) -> None:
        self.finished.emit(result)

    def _on_failed(self, message: str) -> None:
        self.failed.emit(message)

    def _cleanup(self) -> None:
        self._busy = False
        if self._task is not None:
            self._task.deleteLater()
            self._task = None
        if self._thread is not None:
            self._thread.deleteLater()
            self._thread = None
