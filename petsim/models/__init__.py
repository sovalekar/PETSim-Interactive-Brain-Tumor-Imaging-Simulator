"""Domain models (phantoms, scanner geometry). No physics simulation here."""

from petsim.models.phantom import PhantomParameters, SyntheticBrainPhantom
from petsim.models.scanner import DetectorRingGeometry

__all__ = [
    "DetectorRingGeometry",
    "PhantomParameters",
    "SyntheticBrainPhantom",
]
