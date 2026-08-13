"""Contract-bound research visualization publication tools.

Importing :mod:`visualization` does not import Matplotlib or inspect local assets.
"""

from visualization.contracts import FigureData, SourceBinding
from visualization.privacy import PublicationScope
from visualization.registry import FIGURE_REGISTRY, FigureSpec

__all__ = [
    "FIGURE_REGISTRY",
    "FigureData",
    "FigureSpec",
    "PublicationScope",
    "SourceBinding",
]
