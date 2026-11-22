"""High-level reconstruction pipeline: coincidence endpoints → sinogram → image.

Stage 3 filtered backprojection and Stage 5 MLEM share the measured sinogram.
FBP may pre-smooth that sinogram; MLEM uses the unsmoothed counts so the
Poisson likelihood matches the binning of the acquisition.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, replace

import numpy as np

from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.mlem import MLEMHistory, mlem_reconstruct
from petsim.reconstruction.sinogram import Sinogram, build_sinogram, smooth_sinogram_data


@dataclass
class ReconstructionConfig:
    """User-facing reconstruction parameters.

    Defaults are chosen for the educational 48-crystal scanner with a few
    thousand coincidences: geometry-matched angular sampling, Hann window,
    and mild sinogram pre-smoothing so FBP is not dominated by single-LOR
    streaks. Stage 2 acquisition physics is unchanged.
    """

    method: str = "fbp"
    filter_name: str = "hann"  # ramp | hann
    n_theta: int = 96
    n_s: int = 128
    image_size: int = 128
    fov: float = 1.0
    s_max: float = 0.92  # detector-ring radius by default
    # Pre-FBP Gaussian smoothing in sinogram bins (0 disables).
    smooth_sigma_s: float = 1.5
    smooth_sigma_theta: float = 0.75
    # MLEM iterations. Ignored by FBP.
    n_iterations: int = 10

    def clamp(self) -> ReconstructionConfig:
        filt = self.filter_name.lower().strip()
        if filt in {"hann", "hanning", "hann-windowed ramp"}:
            filt = "hann"
        else:
            filt = "ramp"
        method = self.method.lower().strip().replace("-", "").replace("_", "")
        if method in {"mlem", "em", "maximumlikelihood"}:
            method = "mlem"
        else:
            method = "fbp"
        return ReconstructionConfig(
            method=method,
            filter_name=filt,
            n_theta=int(np.clip(self.n_theta, 16, 720)),
            n_s=int(np.clip(self.n_s, 16, 512)),
            image_size=int(np.clip(self.image_size, 32, 512)),
            fov=float(max(self.fov, 1e-3)),
            s_max=float(max(abs(self.s_max), 1e-3)),
            smooth_sigma_s=float(max(self.smooth_sigma_s, 0.0)),
            smooth_sigma_theta=float(max(self.smooth_sigma_theta, 0.0)),
            n_iterations=int(np.clip(self.n_iterations, 1, 200)),
        )

    def as_method(self, method: str) -> ReconstructionConfig:
        """Return a clamped copy that selects ``fbp`` or ``mlem``."""
        return replace(self, method=method).clamp()


@dataclass
class ReconstructionResult:
    """Outputs of one reconstruction run (raw arrays preserved)."""

    sinogram: Sinogram
    image: np.ndarray  # may contain negatives for FBP
    config: ReconstructionConfig
    n_coincidences: int
    message: str = ""
    runtime_s: float = 0.0
    history: MLEMHistory | None = None
    cancelled: bool = False
    job_id: int = 0
    random_estimate_counts: float = 0.0
    scatter_estimate_counts: float = 0.0

    @property
    def image_display(self) -> np.ndarray:
        """Non-negative view for hot/inferno display (does not alter ``image``)."""
        return np.maximum(self.image, 0.0)

    @property
    def has_data(self) -> bool:
        return self.n_coincidences > 0 and self.sinogram.data.sum() > 0

    def display_vmax(self, percentile: float = 99.5) -> float:
        """Robust display ceiling so a few streak spikes do not wash out the image."""
        pos = self.image_display
        if pos.size == 0 or not np.any(pos > 0):
            return 1.0
        vmax = float(np.percentile(pos[pos > 0], percentile))
        return max(vmax, 1e-12)


def reconstruct_from_endpoints(
    hits_a: np.ndarray,
    hits_b: np.ndarray,
    config: ReconstructionConfig | None = None,
    *,
    progress=None,
    cancel=None,
    job_id: int = 0,
    attenuation=None,
    tof=None,
    delta_t_ps=None,
) -> ReconstructionResult:
    """Run FBP or MLEM from measured coincidence endpoints.

    Parameters
    ----------
    hits_a, hits_b :
        Measured coincidence endpoints only (not annihilation truth).
    progress, cancel :
        Optional MLEM iteration callback and cancellation check. Ignored by FBP.
    """
    cfg = (config or ReconstructionConfig()).clamp()
    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    n = int(a.shape[0])
    started = time.perf_counter()

    if n == 0:
        empty = build_sinogram(
            a, b, n_theta=cfg.n_theta, n_s=cfg.n_s, s_max=cfg.s_max
        )
        return ReconstructionResult(
            sinogram=empty,
            image=np.zeros((cfg.image_size, cfg.image_size), dtype=np.float64),
            config=cfg,
            n_coincidences=0,
            message="No coincidence events available for reconstruction.",
            runtime_s=time.perf_counter() - started,
            job_id=job_id,
        )

    sino = build_sinogram(
        a, b, n_theta=cfg.n_theta, n_s=cfg.n_s, s_max=cfg.s_max
    )
    # One weight image, used by exactly one algorithm. FBP scales the measured
    # sinogram. MLEM keeps those counts and weights the projector instead.
    survival = None
    if attenuation is not None and getattr(attenuation, "active", False):
        from petsim.physics.attenuation import sinogram_survival

        survival = sinogram_survival(sino.theta_centers, sino.s_centers, attenuation)
    if cfg.method == "mlem" and tof is not None and getattr(tof, "active", False):
        from petsim.physics.tof import time_bin_centers
        from petsim.reconstruction.tof_mlem import histogram_true_events, tof_mlem_reconstruct

        times = time_bin_centers(tof.n_t, tof.t_max)
        measured_t = np.asarray(delta_t_ps, dtype=np.float64).reshape(-1)
        if measured_t.shape[0] != n:
            raise ValueError("TOF Δt must have one value per true coincidence")
        tof_y = histogram_true_events(
            a, b, measured_t, sino.theta_centers, sino.s_centers, times, tof
        )
        image, history, cancelled = tof_mlem_reconstruct(
            tof_y,
            sino.theta_centers,
            sino.s_centers,
            times,
            tof,
            n_iterations=cfg.n_iterations,
            image_size=cfg.image_size,
            fov=cfg.fov,
            survival=survival,
            progress=progress,
            cancel=cancel,
        )
        runtime = time.perf_counter() - started
        note = "cancelled" if cancelled else "complete"
        return ReconstructionResult(
            sinogram=sino,
            image=image,
            config=cfg,
            n_coincidences=n,
            runtime_s=runtime,
            history=history,
            cancelled=cancelled,
            job_id=job_id,
            message=(
                f"TOF-MLEM ({note}) · {image.shape[0]}×{image.shape[1]} · "
                f"{len(history.iterations)}/{cfg.n_iterations} iterations · "
                f"{n:,} true coincidences · FWHM {tof.fwhm_ps:g} ps · "
                f"time bins {times.size}"
                + (" · attenuation once" if survival is not None else "")
                + " · randoms excluded · not a clinical TOF matrix"
                + f" · {runtime:.2f} s"
            ),
        )

    if cfg.method == "mlem":
        image, history, cancelled = mlem_reconstruct(
            sino,
            n_iterations=cfg.n_iterations,
            image_size=cfg.image_size,
            fov=cfg.fov,
            progress=progress,
            cancel=cancel,
            survival=survival,
        )
        runtime = time.perf_counter() - started
        note = "cancelled" if cancelled else "complete"
        return ReconstructionResult(
            sinogram=sino,
            image=image,
            config=cfg,
            n_coincidences=n,
            runtime_s=runtime,
            history=history,
            cancelled=cancelled,
            job_id=job_id,
            message=(
                f"MLEM ({note}) · {image.shape[0]}×{image.shape[1]} · "
                f"{len(history.iterations)}/{cfg.n_iterations} iterations · "
                f"{n:,} coincidences · unsmoothed sinogram "
                f"{cfg.n_theta}×{cfg.n_s}"
                + (" · attenuation-weighted projector" if survival is not None else "")
                + f" · {runtime:.2f} s"
            ),
        )

    # FBP compensation is a per-bin scale of the measured counts, then the
    # existing smooth and ramp. It is not a clinical attenuation correction.
    attenuated_fbp = survival is not None
    if attenuated_fbp:
        from petsim.physics.attenuation import compensate_sinogram

        sino = Sinogram(
            data=compensate_sinogram(sino.data, survival, attenuation.a_min),
            theta_edges=sino.theta_edges,
            s_edges=sino.s_edges,
            n_events=sino.n_events,
        )
    if cfg.smooth_sigma_s > 0.0 or cfg.smooth_sigma_theta > 0.0:
        sino = Sinogram(
            data=smooth_sinogram_data(
                sino.data,
                sigma_s=cfg.smooth_sigma_s,
                sigma_theta=cfg.smooth_sigma_theta,
            ),
            theta_edges=sino.theta_edges,
            s_edges=sino.s_edges,
            n_events=sino.n_events,
        )

    image = filtered_backprojection(
        sino,
        filter_name=cfg.filter_name,
        image_size=cfg.image_size,
        fov=cfg.fov,
    )
    runtime = time.perf_counter() - started
    return ReconstructionResult(
        sinogram=sino,
        image=image,
        config=cfg,
        n_coincidences=n,
        runtime_s=runtime,
        job_id=job_id,
        message=(
            f"FBP ({cfg.filter_name}) · {cfg.image_size}×{cfg.image_size} · "
            f"{n:,} coincidences · sinogram {cfg.n_theta}×{cfg.n_s} · "
            f"smooth σs={cfg.smooth_sigma_s:g}, σθ={cfg.smooth_sigma_theta:g}"
            + (
                " · parallel-beam attenuation weight (not a clinical correction)"
                if attenuated_fbp
                else ""
            )
            + f" · {runtime:.2f} s"
        ),
    )
