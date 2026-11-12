"""Main three-panel PET imaging workstation window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QLabel,
    QMainWindow,
    QSplitter,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from petsim import __version__
from petsim.acquisition_controller import AcquisitionController
from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig, AcquisitionStats
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult
from petsim.reconstruction_worker import ReconstructionRunner
from petsim.widgets.controls_panel import ControlsPanel
from petsim.widgets.detector_view import DetectorView
from petsim.widgets.reconstruction_panel import ReconstructionPanel


class MainWindow(QMainWindow):
    """Dark-themed workstation: controls | detector FOV | reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            f"PETSim — Interactive Brain Tumor Imaging Simulator v{__version__}"
        )
        self.resize(1380, 840)
        self.setMinimumSize(1040, 660)

        self.acquisition = AcquisitionController(self)
        self.recon_runner = ReconstructionRunner(self)

        self._build_menu()
        self._build_body()
        self._build_status()
        self._connect_signals()

        # Initial phantom paint + sync acquisition PDF / config
        state = self.controls.current_state()
        self.acquisition.set_phantom_state(state)
        self.acquisition.apply_config(self.controls.acquisition_config())
        self._refresh_detector(state)
        self.statusBar().showMessage(
            "Ready — Stage 3: acquire coincidences, then run FBP reconstruction."
        )

    def _build_menu(self) -> None:
        menu = self.menuBar()
        file_menu = menu.addMenu("&File")
        quit_action = QAction("E&xit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        file_menu.addAction(quit_action)

        view_menu = menu.addMenu("&View")
        reset_action = QAction("&Reset panels", self)
        reset_action.triggered.connect(self._reset_splitter)
        view_menu.addAction(reset_action)

        help_menu = menu.addMenu("&Help")
        about_action = QAction("&About PETSim", self)
        about_action.triggered.connect(self._show_about)
        help_menu.addAction(about_action)

    def _build_body(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        outer = QVBoxLayout(central)
        outer.setContentsMargins(10, 10, 10, 10)
        outer.setSpacing(8)

        header = QLabel("PETSim Workstation")
        header.setObjectName("PanelTitle")
        outer.addWidget(header)

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)

        self.controls = ControlsPanel()
        self.detector_view = DetectorView()
        self.reconstruction = ReconstructionPanel()

        self.splitter.addWidget(self.controls)
        self.splitter.addWidget(self.detector_view)
        self.splitter.addWidget(self.reconstruction)
        self.splitter.setStretchFactor(0, 2)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 3)
        self.splitter.setSizes([300, 560, 480])

        outer.addWidget(self.splitter, stretch=1)

    def _build_status(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

    def _connect_signals(self) -> None:
        self.controls.state_changed.connect(self._on_phantom_changed)
        self.controls.display_changed.connect(self._on_display_changed)
        self.controls.acquisition_config_changed.connect(self._on_acq_config)
        self.controls.start_acquisition.connect(self._on_start)
        self.controls.pause_acquisition.connect(self.acquisition.pause)
        self.controls.reset_acquisition.connect(self._on_reset_acq)

        self.acquisition.events_simulated.connect(self.detector_view.show_events)
        self.acquisition.stats_changed.connect(self._on_stats)
        self.acquisition.finished.connect(self._on_finished)
        self.acquisition.reset_done.connect(self._on_acq_reset_done)

        self.reconstruction.run_requested.connect(self._on_run_reconstruction)
        self.reconstruction.clear_requested.connect(self._on_clear_reconstruction)
        self.recon_runner.finished.connect(self._on_recon_finished)
        self.recon_runner.failed.connect(self._on_recon_failed)

    def _refresh_detector(self, state: PhantomState | None = None) -> None:
        state = state or self.controls.current_state()
        opts = self.controls.display_options()
        self.detector_view.update_view(
            state,
            colormap=opts["colormap"],
            show_detectors=opts["show_detectors"],
            show_labels=opts["show_labels"],
        )

    def _on_phantom_changed(self, state: PhantomState) -> None:
        # Changing uptake geometry resets acquisition so configs are not mixed.
        self.acquisition.set_phantom_state(state)
        self.controls.set_acquisition_controls_enabled(True)
        self._refresh_detector(state)
        tumor = "on" if state.tumor_enabled else "off"
        self.statusBar().showMessage(
            f"Phantom updated (acquisition + reconstruction cleared) · "
            f"brain r={state.brain_radius:.2f} · tumor {tumor}"
        )

    def _on_display_changed(self) -> None:
        self._refresh_detector()

    def _on_acq_config(self, config: AcquisitionConfig) -> None:
        self.acquisition.apply_config(config)

    def _on_start(self) -> None:
        # New acquisition invalidates any previous reconstruction.
        self.reconstruction.invalidate("New acquisition started.")
        self.acquisition.apply_config(self.controls.acquisition_config())
        self.acquisition.start()
        self.controls.set_acquisition_controls_enabled(False)
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)

    def _on_reset_acq(self) -> None:
        self.acquisition.reset()
        self.controls.set_acquisition_controls_enabled(True)
        self.statusBar().showMessage("Acquisition reset.")

    def _on_acq_reset_done(self) -> None:
        self.detector_view.clear_events()
        self.reconstruction.set_acquired_phantom(None)
        self.reconstruction.invalidate("Acquisition data cleared.")

    def _on_stats(self, stats: AcquisitionStats) -> None:
        self.controls.update_stats(stats)
        self.reconstruction.update_coincidence_count(stats.coincidences)
        # If acquisition keeps growing after a mid-run recon, mark output stale.
        prior = self.reconstruction._result
        if prior is not None and stats.coincidences != prior.n_coincidences:
            self.reconstruction.invalidate(
                "Acquisition data changed — re-run reconstruction."
            )
        if stats.running:
            self.statusBar().showMessage(
                f"Acquiring… {stats.total_events}/{stats.target_events} · "
                f"coincidences {stats.coincidences} "
                f"({stats.coincidence_percent:.1f}%, expected "
                f"{stats.expected_coincidence_percent:.1f}%)"
            )
        elif stats.paused:
            self.statusBar().showMessage(
                f"Paused at {stats.total_events}/{stats.target_events} annihilations."
            )

    def _on_finished(self) -> None:
        self.controls.set_acquisition_controls_enabled(True)
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)
        stats = self.acquisition.stats
        self.statusBar().showMessage(
            f"Acquisition complete · {stats.total_events} annihilations · "
            f"{stats.coincidences} coincidences — ready for reconstruction."
        )

    def _on_run_reconstruction(self, config: ReconstructionConfig) -> None:
        hits_a, hits_b = self.acquisition.coincidence_endpoints()
        if hits_a.shape[0] == 0:
            self.reconstruction.show_error("No coincidence endpoints available.")
            return
        # Ensure comparison uses the activity map tied to this acquisition.
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)
        ok = self.recon_runner.start(hits_a, hits_b, config)
        if not ok:
            self.reconstruction.show_error("Reconstruction already in progress.")
            return
        self.statusBar().showMessage(
            f"Reconstructing {hits_a.shape[0]:,} coincidences with FBP "
            f"({config.filter_name})…"
        )

    def _on_recon_finished(self, result: ReconstructionResult) -> None:
        self.reconstruction.show_result(result)
        self.statusBar().showMessage(result.message)

    def _on_recon_failed(self, message: str) -> None:
        self.reconstruction.show_error(message)
        self.statusBar().showMessage(f"Reconstruction failed: {message}")

    def _on_clear_reconstruction(self) -> None:
        self.statusBar().showMessage("Reconstruction output cleared.")

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        """Stop timers/threads before the window is destroyed (macOS-safe)."""
        self.acquisition.reset()
        self.recon_runner.shutdown(timeout_ms=10_000)
        super().closeEvent(event)

    def _reset_splitter(self) -> None:
        self.splitter.setSizes([300, 560, 480])

    def _show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About PETSim",
            (
                f"<b>PETSim</b> v{__version__}<br><br>"
                "Educational PET brain-tumor imaging simulator.<br>"
                "Built with PySide6, NumPy, and Matplotlib.<br><br>"
                "<b>Stage 2:</b> annihilation, 511&nbsp;keV photons, coincidences.<br>"
                "<b>Stage 3:</b> measured sinogram + filtered backprojection (FBP).<br><br>"
                "Idealized model: no attenuation, scatter, or positron range.<br>"
                "Not a clinical diagnostic tool."
            ),
        )
