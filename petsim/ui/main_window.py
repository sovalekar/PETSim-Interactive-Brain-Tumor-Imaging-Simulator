"""Main three-panel workstation window."""

from __future__ import annotations

from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import QHBoxLayout, QMainWindow, QSplitter, QWidget

from petsim import __version__
from petsim.models.phantom import PhantomParameters
from petsim.ui.controls_panel import ControlsPanel
from petsim.ui.detector_panel import DetectorPanel
from petsim.ui.reconstruction_panel import ReconstructionPanel


class MainWindow(QMainWindow):
    """Dark PET imaging workstation shell with controls / FOV / reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(f"PETSim — Interactive Brain Tumor Imaging Workstation v{__version__}")
        self.resize(1280, 760)
        self.setMinimumSize(960, 600)

        self.controls = ControlsPanel()
        self.detector = DetectorPanel()
        self.reconstruction = ReconstructionPanel()

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.addWidget(self.controls)
        splitter.addWidget(self.detector)
        splitter.addWidget(self.reconstruction)
        splitter.setStretchFactor(0, 2)
        splitter.setStretchFactor(1, 4)
        splitter.setStretchFactor(2, 2)
        splitter.setSizes([300, 640, 300])

        root = QWidget()
        layout = QHBoxLayout(root)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.addWidget(splitter)
        self.setCentralWidget(root)

        self.statusBar().showMessage(
            "UI shell ready · phantom visualization only · no PET physics / reconstruction"
        )

        self.controls.parameters_changed.connect(self._on_parameters_changed)
        # Initial sync
        self._on_parameters_changed(self.controls.current_parameters())

        quit_action = QAction("Quit", self)
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        self.addAction(quit_action)

    def _on_parameters_changed(self, params: PhantomParameters) -> None:
        self.detector.update_phantom(params)
        tumor = "on" if params.show_tumor else "off"
        self.statusBar().showMessage(
            f"Brain r={params.brain_radius:.2f} · "
            f"Tumor ({params.tumor_x:+.2f}, {params.tumor_y:+.2f}) "
            f"r={params.tumor_radius:.2f} I={params.tumor_intensity:.2f} [{tumor}]"
        )
