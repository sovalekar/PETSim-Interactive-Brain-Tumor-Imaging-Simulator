"""Stage 3: sinogram formation and filtered backprojection (FBP).

Reconstruction uses only measured coincidence endpoints (detector hits).
True annihilation coordinates are never used as reconstruction inputs.
"""

from petsim.reconstruction.fbp import filtered_backprojection
from petsim.reconstruction.lor import endpoints_to_sinogram_coords
from petsim.reconstruction.pipeline import (
    ReconstructionConfig,
    ReconstructionResult,
    reconstruct_from_endpoints,
)
from petsim.reconstruction.sinogram import Sinogram, build_sinogram, smooth_sinogram_data

__all__ = [
    "ReconstructionConfig",
    "ReconstructionResult",
    "Sinogram",
    "build_sinogram",
    "endpoints_to_sinogram_coords",
    "filtered_backprojection",
    "reconstruct_from_endpoints",
    "smooth_sinogram_data",
]
