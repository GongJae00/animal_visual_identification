"""Classification and embedding heads.

ArcFace: standard angular margin softmax.
MagArcFace: magnitude-aware adaptive margin (fixed margin, aux norm loss).
MagArcFaceHead: magnitude-gated variant with explicit norm penalty.
EvidentialHead: Dirichlet-based uncertainty estimation (requires EDL training).
"""

from __future__ import annotations

from cvi.heads.arcface import MagArcFace, ArcFaceHead
from cvi.heads.magface import MagArcFaceHead
from cvi.heads.evidential import EvidentialHead
from cvi.heads.model import PetFaceArcFace

__all__ = [
    "ArcFaceHead", "MagArcFace", "MagArcFaceHead",
    "EvidentialHead", "PetFaceArcFace",
]
