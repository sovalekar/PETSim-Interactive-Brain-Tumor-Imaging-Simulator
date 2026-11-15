"""Stage 9 shell: navigation, badges, and progress without changing physics."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel

from petsim.main_window import MainWindow
from petsim.theme import DARK_STYLESHEET
from petsim.widgets.chrome import KIND_HELP


def _chip_texts(root, object_name: str | None = None) -> list[str]:
    labels = []
    for label in root.findChildren(QLabel):
        if not label.objectName().startswith("Kind"):
            continue
        if object_name is not None and label.objectName() != object_name:
            continue
        labels.append(label.text())
    return labels


def test_navigation_badges_and_scatter_wording(qapp):
    QApplication.instance().setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    assert window.minimumWidth() <= 960
    assert window.minimumHeight() <= 640
    assert [window.tabs.tabText(index) for index in range(window.tabs.count())] == [
        "1  Workstation",
        "2  Detector Physics",
        "3  Image Quality",
        "4  Corrections",
    ]
    assert "global scatter scaling" in window.tabs.tabToolTip(3).lower()
    explanation = " ".join(
        label.text().lower() for label in window.corrections.findChildren(QLabel)
    )
    assert "one fraction" in explanation or "total" in explanation
    assert "complete spatial scatter correction" not in explanation
    window.tabs.setCurrentIndex(2)
    assert window.tabs.currentWidget() is window.experiments
    assert window._steps.checkedId() == 2
    assert window.splitter.widget(1) is window.detector_view
    assert window.controls._progress.maximum() == 1000
    assert window.reconstruction._recon_progress.format() == "Idle"
    assert set(KIND_HELP) == {"parameter", "measured", "estimated", "truth"}
    assert "Parameter" in _chip_texts(window.controls)
    assert "Measured" in _chip_texts(window.controls)
    assert _chip_texts(window.reconstruction, "KindTruth") == ["Ground truth"]
    assert "KindEstimated" in DARK_STYLESHEET
    window.close()
