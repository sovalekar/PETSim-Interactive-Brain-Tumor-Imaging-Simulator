"""High-level Stage 3 pipeline: coincidence endpoints → sinogram → FBP image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.reconstruction.fbp import filtered_backprojection
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

    def clamp(self) -> ReconstructionConfig:
        filt = self.filter_name.lower().strip()
        if filt in {"hann", "hanning", "hann-windowed ramp"}:
            filt = "hann"
        else:
            filt = "ramp"
        return ReconstructionConfig(
            method="fbp",
            filter_name=filt,
            n_theta=int(np.clip(self.n_theta, 16, 720)),
            n_s=int(np.clip(self.n_s, 16, 512)),
            image_size=int(np.clip(self.image_size, 32, 512)),
            fov=float(max(self.fov, 1e-3)),
            s_max=float(max(abs(self.s_max), 1e-3)),
            smooth_sigma_s=float(max(self.smooth_sigma_s, 0.0)),
            smooth_sigma_theta=float(max(self.smooth_sigma_theta, 0.0)),
        )


@dataclass
class ReconstructionResult:
    """Outputs of one reconstruction run (raw arrays preserved)."""

    sinogram: Sinogram
    image: np.ndarray  # may contain negatives
    config: ReconstructionConfig
    n_coincidences: int
    message: str = ""

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
) -> ReconstructionResult:
    """Run the full educational reconstruction pipeline.

    Parameters
    ----------
    hits_a, hits_b :
        Measured coincidence endpoints only (not annihilation truth).
    """
    cfg = (config or ReconstructionConfig()).clamp()
    a = np.asarray(hits_a, dtype=np.float64).reshape(-1, 2)
    b = np.asarray(hits_b, dtype=np.float64).reshape(-1, 2)
    n = int(a.shape[0])

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
        )

    sino = build_sinogram(
        a, b, n_theta=cfg.n_theta, n_s=cfg.n_s, s_max=cfg.s_max
    )
    # Work on a copy so the returned sinogram reflects what FBP actually used.
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
    return ReconstructionResult(
        sinogram=sino,
        image=image,
        config=cfg,
        n_coincidences=n,
        message=(
            f"FBP ({cfg.filter_name}) · {cfg.image_size}×{cfg.image_size} · "
            f"{n:,} coincidences · sinogram {cfg.n_theta}×{cfg.n_s} · "
            f"smooth σs={cfg.smooth_sigma_s:g}, σθ={cfg.smooth_sigma_theta:g}"
        ),
    )
