"""Application bootstrap for the PETSim workstation."""

from __future__ import annotations

import sys

from PySide6.QtCore import Qt
from PySide6.QtWidgets import QApplication

from petsim.main_window import MainWindow
from petsim.theme import DARK_STYLESHEET


def main(argv: list[str] | None = None) -> int:
    """Create the QApplication, apply the dark theme, and show the main window."""
    args = argv if argv is not None else sys.argv

    # High-DPI friendly defaults (safe no-ops on newer Qt if already default).
    QApplication.setHighDpiScaleFactorRoundingPolicy(
        Qt.HighDpiScaleFactorRoundingPolicy.PassThrough
    )

    app = QApplication(args)
    app.setApplicationName("PETSim")
    app.setOrganizationName("PETSim")
    app.setStyle("Fusion")
    app.setStyleSheet(DARK_STYLESHEET)

    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
