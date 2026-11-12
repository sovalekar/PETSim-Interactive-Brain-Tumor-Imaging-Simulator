"""CSV export for individual trials and aggregated experiment statistics."""

from __future__ import annotations

import csv
from pathlib import Path

from petsim import __version__
from petsim.experiments.runner import ExperimentResult

TRIAL_COLUMNS = (
    "count_target",
    "actual_count",
    "trial_index",
    "method",
    "seed",
    "localization_error_fov",
    "tumor_to_background_ratio",
    "relative_contrast",
    "background_cv",
    "runtime_s",
    "mlem_iterations",
    "status",
    "message",
)

AGGREGATE_COLUMNS = (
    "count_target",
    "method",
    "n_trials",
    "n_ok",
    "localization_error_mean",
    "localization_error_std",
    "tbr_mean",
    "tbr_std",
    "contrast_mean",
    "contrast_std",
    "background_cv_mean",
    "background_cv_std",
    "runtime_s_mean",
    "runtime_s_std",
)


def _fmt(value: float | int | None) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.8g}"
    return str(value)


def metadata_lines(result: ExperimentResult) -> list[str]:
    """Comment lines that record how to reproduce the run."""
    cfg = result.config
    phantom = cfg.phantom
    lines = [
        f"# petsim_version={__version__}",
        "# educational_simulation=true",
        "# not_a_clinical_performance_claim=true",
        "# fbp_uses_optional_sinogram_smoothing=true",
        "# mlem_uses_unsmoothed_poisson_counts=true",
        "# std_is_across_independent_acquisitions_not_mlem_convergence=true",
        f"# base_seed={cfg.seed}",
        f"# n_trials={cfg.n_trials}",
        f"# methods={','.join(cfg.methods)}",
        f"# count_targets={','.join(str(c) for c in cfg.count_targets)}",
        f"# detector_efficiency={cfg.detector_efficiency}",
        f"# n_detectors={cfg.n_detectors}",
        f"# filter_name={cfg.filter_name}",
        f"# mlem_iterations={cfg.n_iterations}",
        f"# image_size={cfg.image_size}",
        f"# n_theta={cfg.n_theta}",
        f"# n_s={cfg.n_s}",
        f"# fov={cfg.fov}",
        f"# s_max={cfg.s_max}",
        f"# smooth_sigma_s={cfg.smooth_sigma_s}",
        f"# smooth_sigma_theta={cfg.smooth_sigma_theta}",
        f"# tumor_enabled={int(phantom.tumor_enabled)}",
        f"# tumor_x={phantom.tumor_x}",
        f"# tumor_y={phantom.tumor_y}",
        f"# tumor_radius={phantom.tumor_radius}",
        f"# tumor_intensity={phantom.tumor_intensity}",
        f"# brain_radius={phantom.brain_radius}",
        f"# brain_intensity={phantom.brain_intensity}",
        f"# cancelled={int(result.cancelled)}",
        "# message=" + result.message.replace("\n", " "),
    ]
    return lines


def export_trials_csv(result: ExperimentResult, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        for line in metadata_lines(result):
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=TRIAL_COLUMNS)
        writer.writeheader()
        for trial in result.trials:
            writer.writerow(
                {
                    "count_target": trial.count_target,
                    "actual_count": trial.actual_count,
                    "trial_index": trial.trial_index,
                    "method": trial.method,
                    "seed": trial.seed,
                    "localization_error_fov": _fmt(trial.localization_error),
                    "tumor_to_background_ratio": _fmt(trial.tbr),
                    "relative_contrast": _fmt(trial.contrast),
                    "background_cv": _fmt(trial.background_cv),
                    "runtime_s": _fmt(trial.runtime_s),
                    "mlem_iterations": _fmt(trial.n_iterations),
                    "status": trial.status,
                    "message": trial.message,
                }
            )
    return dest


def export_aggregate_csv(result: ExperimentResult, path: Path | str) -> Path:
    dest = Path(path)
    dest.parent.mkdir(parents=True, exist_ok=True)
    with dest.open("w", newline="", encoding="utf-8") as handle:
        for line in metadata_lines(result):
            handle.write(line + "\n")
        writer = csv.DictWriter(handle, fieldnames=AGGREGATE_COLUMNS)
        writer.writeheader()
        for row in result.aggregates:
            writer.writerow({column: _fmt(getattr(row, column)) for column in AGGREGATE_COLUMNS})
    return dest


def export_experiment(
    result: ExperimentResult,
    directory: Path | str,
    *,
    figures: dict | None = None,
) -> dict[str, Path]:
    """Write trial and aggregate CSV files, and optional PNG figures."""
    folder = Path(directory)
    folder.mkdir(parents=True, exist_ok=True)
    written = {
        "trials_csv": export_trials_csv(result, folder / "trials.csv"),
        "aggregates_csv": export_aggregate_csv(result, folder / "aggregates.csv"),
    }
    for name, figure in (figures or {}).items():
        safe = "".join(ch if ch.isalnum() or ch in "-_" else "_" for ch in name)
        dest = folder / f"{safe}.png"
        figure.savefig(dest, dpi=120, facecolor=figure.get_facecolor())
        written[f"figure_{safe}"] = dest
    return written
