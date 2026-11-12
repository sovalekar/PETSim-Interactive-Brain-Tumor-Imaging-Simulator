"""High-level Stage 3 pipeline: coincidence endpoints → sinogram → FBP image."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.sinogram import Sinogram, build_sinogram


@dataclass
class ReconstructionConfig:
    """User-facing reconstruction parameters."""

    method: str = "fbp"
    filter_name: str = "ramp"  # ramp | hann
    n_theta: int = 180
    n_s: int = 128
    image_size: int = 128
    fov: float = 1.0
    s_max: float = 0.92  # detector-ring radius by default

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
            f"{n:,} coincidences · sinogram {cfg.n_theta}×{cfg.n_s}"
        ),
    )
