"""Identification observer. Writes Visualization/vis/01_identification/.

Imports identification.export only, never identification.training.
"""

from visualization.identification.draw import STAGE, SUBSTAGES, VIS_DIR, render

__all__ = ["STAGE", "SUBSTAGES", "VIS_DIR", "render"]
