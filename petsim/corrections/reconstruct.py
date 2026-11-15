"""Reconstruct from measured endpoints after optional sinogram corrections.

The functions below accept ring hits, an annihilation count, and a random
fraction. They do not accept event labels, annihilation coordinates, or
tumor coordinates.
"""

from __future__ import annotations

import time
from typing import Callable

import numpy as np

from petsim.corrections.estimates import apply_sinogram_corrections
from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.mlem import mlem_reconstruct
from petsim.reconstruction.pipeline import ReconstructionConfig, ReconstructionResult
from petsim.reconstruction.sinogram import Sinogram, build_sinogram, smooth_sinogram_data

CancelCallback = Callable[[], bool]
ProgressCallback = Callable[[int, int, float, float], None]

MODE_IDEAL = "ideal"
MODE_UNCORRECTED = "uncorrected"
MODE_RANDOM = "random"
MODE_SCATTER = "scatter"
MODE_BOTH = "both"
CORRECTION_MODES = (
    MODE_IDEAL,
    MODE_UNCORRECTED,
    MODE_RANDOM,
    MODE_SCATTER,
    MODE_BOTH,
)


def correction_flags(mode: str) -> tuple[bool, bool]:
    """Return ``(correct_random, correct_scatter)`` for a comparison mode."""
    name = str(mode).lower().strip()
    if name == MODE_RANDOM:
        return True, False
    if name == MODE_SCATTER:
        return False, True
    if name == MODE_BOTH:
        return True, True
    return False, False


def reconstruct_corrected(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    config: ReconstructionConfig | None = None,
    *,
    mode: str = MODE_UNCORRECTED,
    n_annihilations: int = 0,
    random_fraction: float = 0.0,
    n_detectors: int = 48,
    ring_radius: float = 0.92,
    progress: ProgressCallback | None = None,
    cancel: CancelCallback | None = None,
    job_id: int = 0,
) -> ReconstructionResult:
    """FBP or MLEM from endpoints, with an optional prompt correction.

    ``mode='ideal'`` is treated like uncorrected: the caller passes the
    ideal-detector endpoints. This function does not know which events were
    random or scattered.
    """
    cfg = (config or ReconstructionConfig()).clamp()
    flags = correction_flags(MODE_UNCORRECTED if mode == MODE_IDEAL else mode)
    started = time.perf_counter()
    if cancel is not None and cancel():
        empty = build_sinogram(
            np.zeros((0, 2)),
            np.zeros((0, 2)),
            n_theta=cfg.n_theta,
            n_s=cfg.n_s,
            s_max=cfg.s_max,
        )
        result = _empty_result(empty, cfg, job_id, started)
        result.cancelled = True
        result.message = "Correction reconstruction cancelled before it started."
        return result

    measured = build_sinogram(
        hits_a,
        hits_b,
        n_theta=cfg.n_theta,
        n_s=cfg.n_s,
        s_max=cfg.s_max,
    )
    corrected, _random_est, _scatter_est = apply_sinogram_corrections(
        measured.data,
        correct_random=flags[0],
        correct_scatter=flags[1],
        n_annihilations=n_annihilations,
        random_fraction=random_fraction,
        s_edges=measured.s_edges,
        n_detectors=n_detectors,
        ring_radius=ring_radius,
        s_max=cfg.s_max,
    )
    prompt = Sinogram(
        data=corrected,
        theta_edges=measured.theta_edges,
        s_edges=measured.s_edges,
        n_events=measured.n_events,
    )
    n = int(np.asarray(hits_a).reshape(-1, 2).shape[0])
    label = _mode_label(mode)
    if cfg.method == "mlem":
        image, history, cancelled = mlem_reconstruct(
            prompt,
            n_iterations=cfg.n_iterations,
            image_size=cfg.image_size,
            fov=cfg.fov,
            progress=progress,
            cancel=cancel,
        )
        runtime = time.perf_counter() - started
        note = "cancelled" if cancelled else "complete"
        return ReconstructionResult(
            sinogram=prompt,
            image=image,
            config=cfg,
            n_coincidences=n,
            runtime_s=runtime,
            history=history,
            cancelled=cancelled,
            job_id=job_id,
            message=(
                f"{label} · MLEM ({note}) · {n:,} input pairs · "
                f"prompts clipped at 0 · {runtime:.2f} s"
            ),
        )

    fbp_sino = prompt
    if cfg.smooth_sigma_s > 0.0 or cfg.smooth_sigma_theta > 0.0:
        fbp_sino = Sinogram(
            data=smooth_sinogram_data(
                prompt.data,
                sigma_s=cfg.smooth_sigma_s,
                sigma_theta=cfg.smooth_sigma_theta,
            ),
            theta_edges=prompt.theta_edges,
            s_edges=prompt.s_edges,
            n_events=prompt.n_events,
        )
    image = filtered_backprojection(
        fbp_sino,
        filter_name=cfg.filter_name,
        image_size=cfg.image_size,
        fov=cfg.fov,
    )
    runtime = time.perf_counter() - started
    return ReconstructionResult(
        sinogram=fbp_sino,
        image=image,
        config=cfg,
        n_coincidences=n,
        runtime_s=runtime,
        job_id=job_id,
        message=(
            f"{label} · FBP ({cfg.filter_name}) · {n:,} input pairs · "
            f"prompts clipped at 0 · {runtime:.2f} s"
        ),
    )


def _mode_label(mode: str) -> str:
    return {
        MODE_IDEAL: "Ideal",
        MODE_UNCORRECTED: "Degraded, uncorrected",
        MODE_RANDOM: "Random corrected",
        MODE_SCATTER: "Scatter corrected",
        MODE_BOTH: "Random and scatter corrected",
    }.get(mode, mode)


def _empty_result(
    sinogram: Sinogram,
    config: ReconstructionConfig,
    job_id: int,
    started: float,
) -> ReconstructionResult:
    return ReconstructionResult(
        sinogram=sinogram,
        image=np.zeros((config.image_size, config.image_size), dtype=np.float64),
        config=config,
        n_coincidences=0,
        runtime_s=time.perf_counter() - started,
        job_id=job_id,
        message="No coincidence events available for reconstruction.",
    )
