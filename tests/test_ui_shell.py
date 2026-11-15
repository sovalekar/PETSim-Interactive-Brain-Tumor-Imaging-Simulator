"""Stage 10 shell: one header, badges, and the scatter-scaling wording."""

from __future__ import annotations

from PySide6.QtWidgets import QApplication, QLabel, QStackedWidget

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


def test_single_header_navigation_and_scatter_wording(qapp):
    QApplication.instance().setStyleSheet(DARK_STYLESHEET)
    window = MainWindow()
    assert isinstance(window.tabs, QStackedWidget)
    from PySide6.QtWidgets import QTabWidget

    assert window.findChildren(QTabWidget) == []
    assert window.minimumWidth() <= 960
    assert window.minimumHeight() <= 640
    labels = [window._steps.button(index).text() for index in range(window.tabs.count())]
    assert labels == ["Workstation", "Detector Physics", "Image Quality", "Corrections"]
    assert "global scatter scaling" in window._steps.button(3).toolTip().lower()
    assert "spatial scatter map" in window._steps.button(3).toolTip().lower()
    explanation = " ".join(label.text().lower() for label in window.corrections.findChildren(QLabel))
    assert "one fraction" in explanation or "total" in explanation
    assert "complete spatial scatter correction" not in explanation
    window.tabs.setCurrentIndex(2)
    assert window.tabs.currentWidget() is window.experiments
    assert window._steps.checkedId() == 2
    assert window.splitter.widget(1) is window.detector_view
    assert "crystal" in window.detector_view.findChildren(QLabel)[1].text()
    assert window.controls._progress.maximum() == 1000
    assert window.reconstruction._recon_progress.format() == "Idle"
    assert set(KIND_HELP) == {"parameter", "measured", "estimated", "truth"}
    assert "Parameter" in _chip_texts(window.controls)
    assert "Measured" in _chip_texts(window.controls)
    assert _chip_texts(window.reconstruction, "KindTruth") == ["Ground truth"]
    assert "KindEstimated" in DARK_STYLESHEET
    window.controls._brain_radius.setValue(40)
    assert window.controls._brain_radius.value() == 40
    window.close()
