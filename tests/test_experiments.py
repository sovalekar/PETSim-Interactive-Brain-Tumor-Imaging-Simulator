"""Stage 6 experiment reproducibility, aggregation, export, and canvas tests."""

from __future__ import annotations

import numpy as np

from petsim.experiments.aggregate import aggregate_trials
from petsim.experiments.config import ExperimentConfig, trial_seed
from petsim.experiments.export import export_experiment
from petsim.experiments.results import TrialResult
from petsim.experiments.runner import collect_coincidences, run_experiment
from petsim.models.detector import DetectorRing
from petsim.models.phantom import BrainPhantom, PhantomState
from petsim.physics.pet_engine import PETEngine
from petsim.widgets.experiment_panel import ExperimentPanel
from petsim.widgets.reconstruction_panel import ReconstructionPanel


def _state() -> PhantomState:
    return PhantomState(
        tumor_enabled=True,
        tumor_x=0.16,
        tumor_y=0.08,
        tumor_radius=0.12,
        tumor_intensity=1.2,
        brain_radius=0.55,
        brain_intensity=0.35,
    )


def _config(**overrides) -> ExperimentConfig:
    values = dict(
        phantom=_state(),
        methods=("fbp",),
        count_targets=(40,),
        n_trials=2,
        seed=21,
        n_iterations=2,
        filter_name="hann",
        detector_efficiency=0.9,
        image_size=24,
        n_theta=12,
        n_s=24,
        smooth_sigma_s=0.4,
        smooth_sigma_theta=0.0,
    )
    values.update(overrides)
    return ExperimentConfig(**values)


def _trial(**overrides) -> TrialResult:
    values = dict(
        count_target=1000,
        actual_count=1000,
        trial_index=0,
        method="fbp",
        seed=1,
        localization_error=0.1,
        tbr=2.0,
        contrast=1.0,
        background_cv=0.4,
        runtime_s=0.2,
        n_iterations=None,
        status="ok",
        message="ok",
    )
    values.update(overrides)
    return TrialResult(**values)


def test_fixed_seed_repeats_the_same_metrics():
    first = run_experiment(_config())
    second = run_experiment(_config())
    assert first.cancelled is False
    assert len(first.trials) == len(second.trials) == 2
    for left, right in zip(first.trials, second.trials, strict=True):
        assert left.status == "ok"
        assert left.actual_count == left.count_target == 40
        assert left.seed == right.seed
        assert left.localization_error == right.localization_error
        assert left.tbr == right.tbr
        assert left.background_cv == right.background_cv


def test_repeated_trials_use_independent_acquisitions():
    engine_a = PETEngine(BrainPhantom(_state()), DetectorRing(48))
    engine_b = PETEngine(BrainPhantom(_state()), DetectorRing(48))
    engine_a.set_seed(trial_seed(5, 0, 0))
    engine_b.set_seed(trial_seed(5, 0, 1))
    first = collect_coincidences(engine_a, 25, detector_efficiency=0.9)
    second = collect_coincidences(engine_b, 25, detector_efficiency=0.9)
    assert first is not None and second is not None
    assert first[0].shape == (25, 2)
    assert not np.allclose(first[0], second[0])


def test_detected_count_matches_the_target():
    result = run_experiment(_config(count_targets=(25, 60), n_trials=1, methods=("fbp",)))
    assert [trial.actual_count for trial in result.trials] == [25, 60]
    assert all(trial.status == "ok" for trial in result.trials)


def test_both_methods_share_the_trial_seed_and_count():
    result = run_experiment(
        _config(methods=("fbp", "mlem"), count_targets=(24,), n_trials=1, n_iterations=2)
    )
    assert [trial.method for trial in result.trials] == ["fbp", "mlem"]
    assert result.trials[0].seed == result.trials[1].seed
    assert result.trials[0].actual_count == result.trials[1].actual_count == 24
    assert result.trials[1].n_iterations == 2
    assert result.trials[0].n_iterations is None
    assert all(trial.status == "ok" for trial in result.trials)


def test_aggregate_mean_and_sample_std():
    trials = [
        _trial(trial_index=0, localization_error=0.10, tbr=2.0),
        _trial(trial_index=1, localization_error=0.30, tbr=4.0),
        _trial(trial_index=2, method="mlem", localization_error=0.20, tbr=3.0, status="failed"),
        _trial(trial_index=3, method="mlem", localization_error=None, status="cancelled"),
    ]
    rows = aggregate_trials(trials)
    fbp = next(row for row in rows if row.method == "fbp")
    mlem = next(row for row in rows if row.method == "mlem")
    assert fbp.n_ok == 2 and fbp.n_trials == 2
    assert abs(fbp.localization_error_mean - 0.20) < 1e-12
    assert abs(fbp.localization_error_std - np.std([0.10, 0.30], ddof=1)) < 1e-12
    assert abs(fbp.tbr_mean - 3.0) < 1e-12
    assert mlem.n_ok == 0
    assert mlem.localization_error_mean is None


def test_cancel_stops_before_later_trials():
    calls = {"n": 0}

    def cancel() -> bool:
        calls["n"] += 1
        return calls["n"] > 3

    result = run_experiment(
        _config(count_targets=(30, 80), n_trials=2, methods=("fbp", "mlem")),
        cancel=cancel,
    )
    assert result.cancelled
    assert any(trial.status == "cancelled" for trial in result.trials)
    assert len(result.trials) < 2 * 2 * 2


def test_csv_export_includes_trials_and_metadata(tmp_path):
    result = run_experiment(_config(n_trials=1, count_targets=(30,)))
    written = export_experiment(result, tmp_path)
    trials_text = written["trials_csv"].read_text(encoding="utf-8")
    aggregate_text = written["aggregates_csv"].read_text(encoding="utf-8")
    assert "# base_seed=21" in trials_text
    assert "localization_error_fov" in trials_text
    assert "fbp" in trials_text
    assert "# mlem_uses_unsmoothed_poisson_counts=true" in aggregate_text
    assert "localization_error_mean" in aggregate_text


def test_experiment_plots_are_separate_from_stage_4_and_5(qapp):
    experiment = ExperimentPanel()
    reconstruction = ReconstructionPanel()
    figures = [
        experiment.plot_localization.figure,
        experiment.plot_tbr.figure,
        experiment.plot_cv.figure,
        experiment.plot_runtime.figure,
        reconstruction.canvas.figure,
        reconstruction.count_canvas.figure,
        reconstruction.convergence_canvas.figure,
    ]
    assert len({id(figure) for figure in figures}) == len(figures)
    recon_title = reconstruction.canvas.figure.axes[0].get_title()
    experiment.plot_localization.show_placeholder("experiment only")
    assert reconstruction.canvas.figure.axes[0].get_title() == recon_title
    assert reconstruction.count_canvas.figure is not experiment.plot_cv.figure
