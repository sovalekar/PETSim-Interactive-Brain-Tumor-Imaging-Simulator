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
from petsim.models.phantom import PhantomState
from petsim.widgets.controls_panel import ControlsPanel
from petsim.widgets.detector_view import DetectorView
from petsim.widgets.reconstruction_panel import ReconstructionPanel


class MainWindow(QMainWindow):
    """Dark-themed workstation shell: controls | detector FOV | reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PETSim — Interactive Brain Tumor Imaging Simulator v{__version__}")
        self.resize(1280, 780)
        self.setMinimumSize(960, 600)

        self._build_menu()
        self._build_body()
        self._build_status()
        self._connect_signals()

        # Initial paint
        self._on_state_changed(self.controls.current_state())

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
        self.splitter.setSizes([300, 640, 300])

        outer.addWidget(self.splitter, stretch=1)

    def _build_status(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)
        status.showMessage(
            "Ready — visualization scaffold only (no PET physics / reconstruction)."
        )

    def _connect_signals(self) -> None:
        self.controls.state_changed.connect(self._on_state_changed)

    def _on_state_changed(self, state: PhantomState) -> None:
        opts = self.controls.display_options()
        self.detector_view.update_view(
            state,
            colormap=opts["colormap"],
            show_detectors=opts["show_detectors"],
            show_labels=opts["show_labels"],
        )
        tumor = "on" if state.tumor_enabled else "off"
        self.statusBar().showMessage(
            f"Phantom updated · brain r={state.brain_radius:.2f} · "
            f"tumor {tumor} · colormap={opts['colormap']}"
        )

    def _reset_splitter(self) -> None:
        self.splitter.setSizes([300, 640, 300])

    def _show_about(self) -> None:
        from PySide6.QtWidgets import QMessageBox

        QMessageBox.about(
            self,
            "About PETSim",
            (
                f"<b>PETSim</b> v{__version__}<br><br>"
                "Interactive brain-tumor PET imaging workstation scaffold.<br>"
                "Built with PySide6, NumPy, and Matplotlib.<br><br>"
                "This release provides adjustable phantom visualization and a "
                "48-crystal detector ring. PET physics and reconstruction are "
                "intentionally not implemented yet."
            ),
        )
