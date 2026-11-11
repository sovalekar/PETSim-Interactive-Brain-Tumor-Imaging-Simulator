"""Application bootstrap: QApplication, dark theme, main window."""

from __future__ import annotations

import sys

from PySide6.QtWidgets import QApplication

from petsim.theme import apply_dark_theme
from petsim.ui.main_window import MainWindow


def run(argv: list[str] | None = None) -> int:
    """Create the Qt application and show the workstation window."""
    args = list(sys.argv if argv is None else argv)
    app = QApplication(args)
    app.setApplicationName("PETSim")
    app.setOrganizationName("PETSim")
    apply_dark_theme(app)

    window = MainWindow()
    window.show()
    return app.exec()
