"""Contract-bound research visualization. Package import does not load Matplotlib."""

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
