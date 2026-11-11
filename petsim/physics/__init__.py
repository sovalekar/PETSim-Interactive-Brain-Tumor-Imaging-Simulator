"""Stage 2 PET physics: annihilation sampling, photon paths, coincidences.

This package is intentionally free of Qt / Matplotlib so the physics engine
can be unit-tested and later reused by a reconstruction module (Stage 3).
"""

from petsim.physics.acquisition import AcquisitionBuffer, AcquisitionStats, AcquisitionConfig
from petsim.physics.pet_engine import AnnihilationEvent, PETEngine

__all__ = [
    "AcquisitionBuffer",
    "AcquisitionConfig",
    "AcquisitionStats",
    "AnnihilationEvent",
    "PETEngine",
]
