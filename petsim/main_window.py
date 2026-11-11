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
        self.resize(1320, 820)
        self.setMinimumSize(1000, 640)

        self.acquisition = AcquisitionController(self)

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
            "Ready — Stage 2: positron annihilation & coincidence acquisition."
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
        self.splitter.setStretchFactor(1, 4)
        self.splitter.setStretchFactor(2, 2)
        self.splitter.setSizes([320, 660, 300])

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
        self.acquisition.reset_done.connect(self.detector_view.clear_events)

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
            f"Phantom updated (acquisition reset) · brain r={state.brain_radius:.2f} · "
            f"tumor {tumor}"
        )

    def _on_display_changed(self) -> None:
        self._refresh_detector()

    def _on_acq_config(self, config: AcquisitionConfig) -> None:
        self.acquisition.apply_config(config)

    def _on_start(self) -> None:
        self.acquisition.apply_config(self.controls.acquisition_config())
        self.acquisition.start()
        self.controls.set_acquisition_controls_enabled(False)

    def _on_reset_acq(self) -> None:
        self.acquisition.reset()
        self.controls.set_acquisition_controls_enabled(True)
        self.statusBar().showMessage("Acquisition reset.")

    def _on_stats(self, stats: AcquisitionStats) -> None:
        self.controls.update_stats(stats)
        self.reconstruction.update_coincidence_count(stats.coincidences)
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
        stats = self.acquisition.stats
        self.statusBar().showMessage(
            f"Acquisition complete · {stats.total_events} annihilations · "
            f"{stats.coincidences} coincidences "
            f"({stats.coincidence_percent:.1f}% vs ε²="
            f"{stats.expected_coincidence_percent:.1f}%)"
        )

    def _reset_splitter(self) -> None:
        self.splitter.setSizes([320, 660, 300])

    def _show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About PETSim",
            (
                f"<b>PETSim</b> v{__version__}<br><br>"
                "Educational PET brain-tumor imaging simulator.<br>"
                "Built with PySide6, NumPy, and Matplotlib.<br><br>"
                "<b>Stage 2:</b> positron annihilation, 511&nbsp;keV back-to-back "
                "photons, detector efficiency, and coincidence acquisition.<br>"
                "<b>Stage 3 (not yet):</b> image reconstruction.<br><br>"
                "Idealized model: no attenuation, scatter, or positron range.<br>"
                "Not a clinical diagnostic tool."
            ),
        )
