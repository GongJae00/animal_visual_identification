"""Contract-bound research visualization publication tools.

Importing :mod:`vis` does not import Matplotlib or inspect local assets.
"""

from vis.contracts import FigureData, SourceBinding
from vis.privacy import PublicationScope
from vis.registry import FIGURE_REGISTRY, FigureSpec

__all__ = [
    "FIGURE_REGISTRY",
    "FigureData",
    "FigureSpec",
    "PublicationScope",
    "SourceBinding",
]
