"""Main three-panel PET imaging workstation window."""

from __future__ import annotations

import time

import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QAction
from PySide6.QtWidgets import (
    QButtonGroup,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QProgressBar,
    QPushButton,
    QSizePolicy,
    QSplitter,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
)

from petsim.widgets.chrome import show_guide

from petsim import __version__
from petsim.acquisition_controller import AcquisitionController
from petsim.attenuation_worker import AttenuationStudyRunner
from petsim.experiment_worker import ExperimentRunner
from petsim.analysis.pipeline import analyze_reconstruction
from petsim.analysis_worker import AnalysisRunner
from petsim.models.phantom import PhantomState
from petsim.physics.acquisition import AcquisitionConfig, AcquisitionStats
from petsim.corrections_worker import CorrectionComparisonRunner, CorrectionStudyRunner
from petsim.physics_comparison_worker import PhysicsComparisonRunner
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult
from petsim.reconstruction_worker import ReconstructionRunner
from petsim.widgets.controls_panel import ControlsPanel
from petsim.widgets.corrections_panel import CorrectionsPanel
from petsim.widgets.detector_physics_panel import DetectorPhysicsPanel
from petsim.widgets.experiment_panel import ExperimentPanel
from petsim.widgets.detector_view import DetectorView
from petsim.widgets.reconstruction_panel import ReconstructionPanel


def _header_state(message: str) -> str:
    """One word for the header. The full sentence stays in the status bar."""
    text = message.lower()
    if text.startswith("ready"):
        return "Ready"
    if "paused" in text:
        return "Paused"
    if "complete" in text:
        return "Complete"
    if "fail" in text or "unavailable" in text or "error" in text:
        return "Check status"
    if "reconstruct" in text or text.startswith("mlem"):
        return "Reconstructing"
    if "acquir" in text:
        return "Acquiring"
    return "Working"


class MainWindow(QMainWindow):
    """Dark-themed workstation: controls | detector FOV | reconstruction."""

    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle(
            f"PETSim — Interactive Brain Tumor Imaging Simulator v{__version__}"
        )
        self.resize(1380, 860)
        self.setMinimumSize(960, 620)

        self.acquisition = AcquisitionController(self)
        self.recon_runner = ReconstructionRunner(self)
        self.analysis_runner = AnalysisRunner(self)
        self.experiment_runner = ExperimentRunner(self)
        self.physics_runner = PhysicsComparisonRunner(self)
        self.correction_runner = CorrectionComparisonRunner(self)
        self.correction_study_runner = CorrectionStudyRunner(self)
        self.attenuation_runner = AttenuationStudyRunner(self)
        self._last_physics_draw = 0.0

        self._build_menu()
        self._build_body()
        self._build_status()
        self._connect_signals()

        # Initial phantom paint + sync acquisition PDF / config
        state = self.controls.current_state()
        self.acquisition.set_phantom_state(state)
        self.acquisition.apply_config(self.controls.acquisition_config())
        self._refresh_detector(state)
        self._note(
            "Ready — configure the phantom, acquire coincidences, then reconstruct."
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
        outer.setContentsMargins(8, 8, 8, 8)
        outer.setSpacing(8)

        self.experiments = ExperimentPanel()
        self.detector_physics = DetectorPhysicsPanel()
        self.corrections = CorrectionsPanel()

        self.splitter = QSplitter(Qt.Orientation.Horizontal)
        self.splitter.setChildrenCollapsible(False)
        self.splitter.setHandleWidth(6)

        self.controls = ControlsPanel()
        self.detector_view = DetectorView()
        self.reconstruction = ReconstructionPanel()
        self.controls.setMinimumWidth(240)
        self.detector_view.setMinimumWidth(260)
        self.reconstruction.setMinimumWidth(280)

        self.splitter.addWidget(self.controls)
        self.splitter.addWidget(self.detector_view)
        self.splitter.addWidget(self.reconstruction)
        self.splitter.setStretchFactor(0, 0)
        self.splitter.setStretchFactor(1, 3)
        self.splitter.setStretchFactor(2, 4)
        self.splitter.setSizes([260, 480, 540])

        workstation = QWidget()
        station_layout = QVBoxLayout(workstation)
        station_layout.setContentsMargins(0, 0, 0, 0)
        station_layout.addWidget(self.splitter)

        self.tabs = QStackedWidget()
        pages = (
            (workstation, "Workstation", "Configure the phantom, acquire coincidences, and reconstruct."),
            (self.detector_physics, "Detector Physics", "Optional energy, timing, randoms, and a simplified Compton cartoon."),
            (self.experiments, "Image Quality", "Repeated trials of localization, contrast, and noise versus count."),
            (
                self.corrections,
                "Corrections",
                "Random coincidence correction and educational global scatter scaling. "
                "Scatter scaling is one fraction, not a spatial scatter map.",
            ),
        )
        self._steps = QButtonGroup(self)
        self._steps.setExclusive(True)
        bar = QWidget()
        bar.setObjectName("AppBar")
        bar_layout = QHBoxLayout(bar)
        bar_layout.setContentsMargins(12, 8, 12, 8)
        bar_layout.setSpacing(10)
        logo = QLabel("⊕")
        logo.setObjectName("LogoMark")
        logo.setToolTip("PETSim")
        bar_layout.addWidget(logo)
        titles = QVBoxLayout()
        titles.setSpacing(0)
        app_title = QLabel("PETSim")
        app_title.setObjectName("AppTitle")
        app_sub = QLabel("Brain tumor PET")
        app_sub.setObjectName("AppSubtitle")
        app_title.setToolTip("Interactive Brain Tumor Imaging Simulator")
        titles.addWidget(app_title)
        titles.addWidget(app_sub)
        bar_layout.addLayout(titles)
        bar_layout.addSpacing(12)
        for index, (page, label, tip) in enumerate(pages):
            self.tabs.addWidget(page)
            step = QPushButton(label)
            step.setObjectName("StepButton")
            step.setCheckable(True)
            step.setToolTip(tip)
            step.setChecked(index == 0)
            self._steps.addButton(step, index)
            bar_layout.addWidget(step)
        bar_layout.addStretch(1)
        self._header_status = QLabel("Ready")
        self._header_status.setObjectName("HeaderStatus")
        self._header_status.setSizePolicy(QSizePolicy.Policy.Maximum, QSizePolicy.Policy.Preferred)
        self._header_progress = QProgressBar()
        self._header_progress.setRange(0, 100)
        self._header_progress.setValue(0)
        self._header_progress.setFixedWidth(96)
        self._header_progress.setFormat("Idle")
        self._header_progress.setToolTip("Progress of the current acquisition or reconstruction.")
        help_btn = QPushButton("Guide")
        help_btn.setToolTip("Short explanations of coincidences, reconstruction, and the metrics.")
        help_btn.clicked.connect(lambda: show_guide(self))
        bar_layout.addWidget(self._header_status)
        bar_layout.addWidget(self._header_progress)
        bar_layout.addWidget(help_btn)
        self._steps.idClicked.connect(self.tabs.setCurrentIndex)
        self.tabs.currentChanged.connect(self._on_tab_changed)
        self.tabs.currentChanged.connect(self._sync_step)
        outer.addWidget(bar)
        outer.addWidget(self.tabs, stretch=1)

    def _build_status(self) -> None:
        status = QStatusBar()
        self.setStatusBar(status)

    def _note(self, message: str) -> None:
        self.statusBar().showMessage(message)
        if hasattr(self, "_header_status"):
            self._header_status.setText(_header_state(message))
            self._header_status.setToolTip(message)

    def _set_header_progress(self, value: int, text: str) -> None:
        self._header_progress.setRange(0, 100)
        self._header_progress.setValue(max(0, min(100, value)))
        self._header_progress.setFormat(text)

    def _connect_signals(self) -> None:
        self.controls.state_changed.connect(self._on_phantom_changed)
        self.controls.display_changed.connect(self._on_display_changed)
        self.controls.acquisition_config_changed.connect(self._on_acq_config)
        self.controls.attenuation_changed.connect(self.acquisition.apply_attenuation)
        self.controls.attenuation_study_requested.connect(self._on_attenuation_study)
        self.controls.start_acquisition.connect(self._on_start)
        self.controls.pause_acquisition.connect(self.acquisition.pause)
        self.controls.reset_acquisition.connect(self._on_reset_acq)

        self.acquisition.events_simulated.connect(self.detector_view.show_events)
        self.detector_view.labels_toggled.connect(self._toggle_detector_labels)
        self.acquisition.stats_changed.connect(self._on_stats)
        self.acquisition.finished.connect(self._on_finished)
        self.acquisition.reset_done.connect(self._on_acq_reset_done)

        self.reconstruction.run_requested.connect(self._on_run_reconstruction)
        self.reconstruction.compare_requested.connect(self._on_compare_algorithms)
        self.reconstruction.cancel_requested.connect(self.recon_runner.cancel)
        self.reconstruction.clear_requested.connect(self._on_clear_reconstruction)
        self.reconstruction.analyze_requested.connect(self._on_run_analysis)
        self.reconstruction.count_comparison_requested.connect(
            self._on_run_count_comparison
        )

        self.recon_runner.finished.connect(self._on_recon_finished)
        self.recon_runner.pair_finished.connect(self._on_pair_finished)
        self.recon_runner.failed.connect(self._on_recon_failed)
        self.recon_runner.progress.connect(self._on_mlem_progress)
        self.attenuation_runner.finished.connect(self._on_attenuation_study_finished)
        self.attenuation_runner.failed.connect(self._on_attenuation_study_failed)

        self.analysis_runner.analysis_finished.connect(self._on_analysis_finished)
        self.analysis_runner.analysis_failed.connect(self._on_analysis_failed)
        self.analysis_runner.comparison_finished.connect(self._on_comparison_finished)
        self.analysis_runner.comparison_failed.connect(self._on_analysis_failed)

        self.experiments.start_requested.connect(self._on_start_experiment)
        self.experiments.cancel_requested.connect(self.experiment_runner.cancel)
        self.experiments.reset_requested.connect(self.experiment_runner.invalidate_pending)
        self.experiment_runner.progress.connect(self.experiments.show_progress)
        self.experiment_runner.finished.connect(self._on_experiment_finished)
        self.experiment_runner.failed.connect(self._on_experiment_failed)

        self.detector_physics.config_changed.connect(self._on_detector_physics)
        self.detector_physics.compare_requested.connect(self._on_compare_detector_configurations)
        self.detector_physics.cancel_compare_requested.connect(self.physics_runner.cancel)
        self.physics_runner.finished.connect(self._on_physics_comparison_finished)
        self.physics_runner.failed.connect(self.detector_physics.show_comparison_message)
        self.acquisition.apply_detector_physics(self.detector_physics.current_config())
        self.experiments.set_detector_physics(self.detector_physics.current_config())

        self.corrections.compare_requested.connect(self._on_compare_corrections)
        self.corrections.cancel_compare_requested.connect(self.correction_runner.cancel)
        self.corrections.study_requested.connect(self._on_start_correction_study)
        self.corrections.cancel_study_requested.connect(self.correction_study_runner.cancel)
        self.correction_runner.finished.connect(self._on_corrections_finished)
        self.correction_runner.failed.connect(self.corrections.show_comparison_message)
        self.correction_runner.progress.connect(
            lambda done, total, message: self.corrections.show_comparison_busy(
                f"Reconstructing {message} ({done + 1}/{total})…"
            )
        )
        self.correction_study_runner.finished.connect(self.corrections.show_study_result)
        self.correction_study_runner.failed.connect(self.corrections.show_study_error)
        self.correction_study_runner.progress.connect(self.corrections.show_study_progress)

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
        self._note(
            f"Phantom updated (acquisition + reconstruction + analysis cleared) · "
            f"brain r={state.brain_radius:.2f} · tumor {tumor}"
        )

    def _on_display_changed(self) -> None:
        self._refresh_detector()

    def _on_acq_config(self, config: AcquisitionConfig) -> None:
        self.acquisition.apply_config(config)

    def _physics_tab_visible(self) -> bool:
        return self.tabs.currentWidget() is self.detector_physics

    def _push_physics_stats(self, stats: AcquisitionStats, *, force: bool = False) -> None:
        """Redraw energy and class plots only while that tab is open.

        A full histogram on every acquisition tick stalls the workstation.
        Labels still update every tick. Opening the tab refreshes the figures.
        """
        visible = self._physics_tab_visible()
        now = time.monotonic()
        redraw = force or (
            visible and (stats.finished or now - self._last_physics_draw >= 0.2)
        )
        energies = None
        if redraw:
            self._last_physics_draw = now
            energies = self.acquisition.measured_photon_energies_kev()
        self.detector_physics.update_stats(stats, energies, redraw_plots=redraw)

    def _sync_step(self, index: int) -> None:
        button = self._steps.button(index)
        if button is not None:
            button.setChecked(True)

    def _on_tab_changed(self, _index: int) -> None:
        if self._physics_tab_visible():
            self._push_physics_stats(self.acquisition.stats, force=True)

    def _on_detector_physics(self, config) -> None:
        self.acquisition.apply_detector_physics(config)
        self.experiments.set_detector_physics(config)
        if self.acquisition.stats.total_events > 0:
            self._note(
                "Detector settings apply to the next acquisition. "
                "Events already stored are unchanged."
            )

    def _on_start(self) -> None:
        # New acquisition invalidates any previous reconstruction / analysis.
        self.recon_runner.invalidate_pending()
        self.physics_runner.invalidate_pending()
        self.reconstruction.invalidate("New acquisition started.")
        self.detector_physics.clear_visuals()
        self.correction_runner.invalidate_pending()
        self.corrections.clear_comparison()
        self.acquisition.apply_config(self.controls.acquisition_config())
        self.acquisition.start()
        self.controls.set_acquisition_controls_enabled(False)
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)

    def _on_compare_detector_configurations(self) -> None:
        ideal_a, ideal_b = self.acquisition.ideal_coincidence_endpoints()
        degraded_a, degraded_b = self.acquisition.coincidence_endpoints()
        if ideal_a.shape[0] == 0 and degraded_a.shape[0] == 0:
            self.detector_physics.show_comparison_message(
                "Acquire coincidences before comparing ideal and degraded reconstructions."
            )
            return
        config = ReconstructionConfig(
            method=self.detector_physics.comparison_method(),
            n_iterations=self.detector_physics.comparison_iterations(),
            image_size=96,
            n_theta=64,
            n_s=96,
            filter_name="hann",
        ).clamp()
        started = self.physics_runner.start(ideal_a, ideal_b, degraded_a, degraded_b, config)
        if not started:
            self.detector_physics.show_comparison_message("A comparison is already running.")
            return
        self.detector_physics.show_comparison_busy(
            f"Reconstructing ideal and degraded data with {config.method.upper()}…"
        )

    def _on_compare_corrections(self) -> None:
        degraded_a, degraded_b = self.acquisition.coincidence_endpoints()
        ideal_a, ideal_b = self.acquisition.ideal_coincidence_endpoints()
        if degraded_a.shape[0] == 0 and ideal_a.shape[0] == 0:
            self.corrections.show_comparison_message(
                "Acquire coincidences before comparing corrections."
            )
            return
        physics = self.acquisition.active_detector_physics()
        fraction = physics.random_fraction if physics.enabled else 0.0
        scatter_probability = physics.scatter_probability if physics.enabled else 0.0
        config = ReconstructionConfig(
            method=self.corrections.comparison_method(),
            n_iterations=self.corrections.comparison_iterations(),
            image_size=64,
            n_theta=48,
            n_s=64,
            filter_name="hann",
            smooth_sigma_s=1.0,
            smooth_sigma_theta=0.5,
        ).clamp()
        started = self.correction_runner.start(
            ideal_a,
            ideal_b,
            degraded_a,
            degraded_b,
            config,
            n_annihilations=self.acquisition.annihilation_count(),
            random_fraction=fraction,
            phantom=self.acquisition.acquired_phantom_state,
            n_side_window=self.acquisition.side_window_count(),
            scatter_probability=scatter_probability,
            photo_low_kev=physics.energy_window_low_kev,
            photo_high_kev=physics.energy_window_high_kev,
            energy_fwhm_percent=physics.energy_fwhm_percent,
            scatter_angle_sigma_rad=physics.scatter_angle_sigma_rad,
        )
        if not started:
            self.corrections.show_comparison_message("A correction comparison is already running.")
            return
        self.corrections.show_comparison_busy(
            f"Reconstructing five {config.method.upper()} images from this acquisition…"
        )

    def _on_corrections_finished(self, bundle) -> None:
        try:
            self.corrections.show_comparison(bundle)
        except Exception as exc:  # noqa: BLE001
            self.corrections.show_comparison_message(f"Could not display the comparison: {exc}")

    def _on_start_correction_study(self) -> None:
        config = self.corrections.build_study_config(self.controls.current_state())
        if config is None:
            self.corrections.show_study_error("Select at least one random probability and one scatter probability.")
            return
        if not self.correction_study_runner.start(config):
            self.corrections.show_study_error("A correction study is already running.")
            return
        self.corrections.show_study_progress(0, 1, "Starting matched correction study…")

    def _on_physics_comparison_finished(self, result) -> None:
        try:
            self.detector_physics.show_comparison(result.ideal, result.degraded)
        except Exception as exc:  # noqa: BLE001 — a display error must not leave the buttons stuck
            self.detector_physics.show_comparison_message(
                f"Could not display the reconstruction: {exc}"
            )

    def _on_reset_acq(self) -> None:
        self.acquisition.reset()
        self.controls.set_acquisition_controls_enabled(True)
        self._note("Acquisition reset.")

    def _on_acq_reset_done(self) -> None:
        self.recon_runner.invalidate_pending()
        self.physics_runner.invalidate_pending()
        self.detector_view.clear_events()
        self.reconstruction.set_acquired_phantom(None)
        self.reconstruction.invalidate("Acquisition data cleared.")
        self.detector_physics.clear_visuals()
        self.detector_physics.update_stats(self.acquisition.stats, np.zeros(0))
        self.correction_runner.invalidate_pending()
        self.corrections.clear_comparison()

    def _on_stats(self, stats: AcquisitionStats) -> None:
        self.controls.update_stats(stats)
        self._push_physics_stats(stats)
        self.reconstruction.update_coincidence_count(stats.coincidences)
        # If acquisition keeps growing after a mid-run recon, mark output stale.
        prior = self.reconstruction._result
        if prior is not None and stats.coincidences != prior.n_coincidences:
            self.reconstruction.invalidate(
                "Acquisition data changed — re-run reconstruction."
            )
        if stats.running:
            self._set_header_progress(int(stats.progress_percent), f"{stats.progress_percent:.0f}%")
            self._note(
                f"Acquiring… {stats.total_events}/{stats.target_events} · "
                f"coincidences {stats.coincidences} "
                f"({stats.coincidence_percent:.1f}%, expected "
                f"{stats.expected_coincidence_percent:.1f}%)"
            )
        elif stats.paused:
            self._set_header_progress(int(stats.progress_percent), "Paused")
            self._note(
                f"Paused at {stats.total_events}/{stats.target_events} annihilations."
            )

    def _on_finished(self) -> None:
        self.controls.set_acquisition_controls_enabled(True)
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)
        stats = self.acquisition.stats
        self._set_header_progress(100, "Complete")
        self._note(
            f"Acquisition complete · {stats.total_events} annihilations · "
            f"{stats.coincidences} coincidences — ready for reconstruction."
        )

    def _on_run_reconstruction(self, config: ReconstructionConfig) -> None:
        hits_a, hits_b = self.acquisition.coincidence_endpoints()
        if hits_a.shape[0] == 0:
            self.reconstruction.show_error(
                "Reconstruction unavailable — acquire coincidence data first."
            )
            return
        # Ensure comparison uses the activity map tied to this acquisition.
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)
        ok = self.recon_runner.start(
            hits_a,
            hits_b,
            config,
            attenuation=self.acquisition.active_attenuation(),
        )
        if not ok:
            self.reconstruction.show_error("Reconstruction already in progress.")
            return
        if config.method == "mlem":
            label = f"MLEM ({config.n_iterations} iterations)"
        else:
            label = f"FBP ({config.filter_name})"
        self._note(
            f"Reconstructing {hits_a.shape[0]:,} coincidences with {label}…"
        )

    def _on_mlem_progress(
        self, job_id: int, iteration: int, total: int, loglik: float, rel: float
    ) -> None:
        if not self.recon_runner.accepts(job_id):
            return
        self.reconstruction.show_mlem_progress(iteration, total, loglik, rel)
        done = int(round(100 * iteration / max(int(total), 1)))
        self._set_header_progress(done, f"MLEM {iteration}/{total}")

    def _on_compare_algorithms(self, fbp_config: ReconstructionConfig, mlem_config: ReconstructionConfig) -> None:
        hits_a, hits_b = self.acquisition.coincidence_endpoints()
        if hits_a.shape[0] == 0:
            self.reconstruction.show_error(
                "Reconstruction unavailable — acquire coincidence data first."
            )
            return
        self.reconstruction.set_acquired_phantom(self.acquisition.acquired_activity)
        ok = self.recon_runner.start(
            hits_a,
            hits_b,
            fbp_config,
            pair=True,
            mlem_config=mlem_config,
            attenuation=self.acquisition.active_attenuation(),
        )
        if not ok:
            self.reconstruction.show_error("Reconstruction already in progress.")
            return
        self._note(
            f"Comparing FBP and MLEM on {hits_a.shape[0]:,} coincidences "
            f"({mlem_config.n_iterations} MLEM iterations)…"
        )

    def _on_recon_finished(self, result: ReconstructionResult) -> None:
        if not self.recon_runner.accepts(result.job_id):
            return
        self.reconstruction.show_result(result)
        self._refresh_algorithm_metrics()
        self._note(result.message)

    def _on_pair_finished(self, fbp: ReconstructionResult, mlem: ReconstructionResult) -> None:
        if not self.recon_runner.accepts(fbp.job_id):
            return
        self.reconstruction.show_algorithm_pair(fbp, mlem)
        self._refresh_algorithm_metrics()
        self._note(f"{fbp.message} | {mlem.message}")

    def _refresh_algorithm_metrics(self) -> None:
        fbp = self.reconstruction._fbp_result
        mlem = self.reconstruction._mlem_result
        state = self.acquisition.acquired_phantom_state
        if (
            fbp is None
            or mlem is None
            or not fbp.has_data
            or not mlem.has_data
            or state is None
        ):
            return
        self.reconstruction.show_algorithm_metrics(
            analyze_reconstruction(fbp, state),
            analyze_reconstruction(mlem, state),
        )

    def _on_recon_failed(self, message: str) -> None:
        self.reconstruction.show_error(message)
        self._note(f"Reconstruction failed: {message}")

    def _on_run_analysis(self) -> None:
        result = self.reconstruction._result
        if result is None or not result.has_data:
            self.reconstruction.show_analysis_error(
                "No reconstruction available for analysis."
            )
            return
        phantom_state = self.acquisition.acquired_phantom_state
        ok = self.analysis_runner.start_analysis(result, phantom_state)
        if not ok:
            self.reconstruction.show_analysis_error("Analysis already in progress.")
            return
        self._note("Running quantitative analysis…")

    def _on_run_count_comparison(self) -> None:
        hits_a, hits_b = self.acquisition.coincidence_endpoints()
        if hits_a.shape[0] == 0:
            self.reconstruction.show_analysis_error(
                "No coincidence endpoints available for count comparison."
            )
            return
        config = self.reconstruction.current_config().as_method("fbp")
        phantom_state = self.acquisition.acquired_phantom_state
        ok = self.analysis_runner.start_count_comparison(
            hits_a, hits_b, config, phantom_state
        )
        if not ok:
            self.reconstruction.show_analysis_error(
                "Analysis already in progress."
            )
            return
        self._note(
            f"Comparing reconstruction quality across coincidence-count levels "
            f"({hits_a.shape[0]:,} available)…"
        )

    def _on_analysis_finished(self, analysis) -> None:
        self.reconstruction.show_analysis(analysis)
        self._note(analysis.message)

    def _on_comparison_finished(self, comparison) -> None:
        self.reconstruction.show_count_comparison(comparison)
        self._note(comparison.message)

    def _on_analysis_failed(self, message: str) -> None:
        self.reconstruction.show_analysis_error(message)
        self._note(f"Analysis failed: {message}")

    def _on_start_experiment(self) -> None:
        config = self.experiments.build_config(self.controls.current_state())
        if config is None:
            self.experiments.show_error("Select at least one method and one coincidence target.")
            return
        self.experiments.set_busy(True)
        self.experiments.show_progress(0, 1, "Starting experiment…")
        ok = self.experiment_runner.start(config)
        if not ok:
            self.experiments.show_error("An experiment is already running.")
            return
        self._note(
            "Image-quality experiment running "
            f"({', '.join(m.upper() for m in config.methods)}, "
            f"{config.n_trials} trial(s))."
        )

    def _on_experiment_finished(self, result) -> None:
        self.experiments.show_result(result)
        self._note(result.message)

    def _on_experiment_failed(self, message: str) -> None:
        self.experiments.show_error(message)
        self._note(f"Experiment failed: {message}")

    def _on_attenuation_study(self) -> None:
        config = self.controls.attenuation_config()
        if not config.enabled or config.mu_per_cm <= 0.0:
            self._note("Enable tissue attenuation before comparing the correction.")
            return
        method = "mlem" if self.reconstruction._is_mlem() else "fbp"
        seed = self.controls.study_seed()
        ok = self.attenuation_runner.start(
            self.controls.current_state(),
            config,
            method=method,
            seeds=(seed, seed + 1),
        )
        if not ok:
            self._note("An attenuation comparison is already running.")
            return
        self.controls.set_study_busy(True)
        self._note(
            f"Comparing ideal, attenuated, and corrected {method.upper()} "
            f"on seeds {seed} and {seed + 1}…"
        )

    def _on_attenuation_study_finished(self, result) -> None:
        self.controls.set_study_busy(False)
        self.reconstruction.show_attenuation_study(result)
        self._note(result.message)

    def _on_attenuation_study_failed(self, message: str) -> None:
        self.controls.set_study_busy(False)
        self._note(f"Attenuation comparison failed: {message}")

    def _on_clear_reconstruction(self) -> None:
        self.recon_runner.invalidate_pending()
        self._note("Reconstruction output cleared.")

    def closeEvent(self, event) -> None:  # noqa: N802 — Qt naming
        """Stop timers/threads before the window is destroyed (macOS-safe)."""
        self.acquisition.reset()
        self.recon_runner.shutdown(timeout_ms=10_000)
        self.analysis_runner.shutdown(timeout_ms=10_000)
        self.experiment_runner.shutdown(timeout_ms=10_000)
        self.physics_runner.shutdown(timeout_ms=10_000)
        self.correction_runner.shutdown(timeout_ms=10_000)
        self.correction_study_runner.shutdown(timeout_ms=10_000)
        self.attenuation_runner.shutdown(timeout_ms=10_000)
        super().closeEvent(event)

    def _toggle_detector_labels(self) -> None:
        box = self.controls._show_labels
        box.setChecked(not box.isChecked())

    def _reset_splitter(self) -> None:
        self.splitter.setSizes([260, 480, 540])

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
                "<b>Stage 3:</b> measured sinogram + filtered backprojection (FBP).<br>"
                "<b>Stage 4:</b> localization, contrast / TBR, count comparison.<br>"
                "<b>Stage 5:</b> MLEM iterative reconstruction vs FBP.<br>"
                "<b>Stage 6:</b> repeated image-quality experiments.<br>"
                "<b>Stage 7:</b> optional energy, timing, randoms, and simplified scatter.<br>"
                "<b>Stage 8:</b> random coincidence correction and global scatter scaling.<br>"
                "<b>Stage 9:</b> workstation layout.<br>"
                "<b>Stage 10:</b> single navigation header and metric cards. "
                "Scatter scaling is still a count scale, not a spatial scatter correction.<br><br>"
                "Detector effects are off unless you enable them.<br>"
                "Not a clinical diagnostic tool."
            ),
        )
