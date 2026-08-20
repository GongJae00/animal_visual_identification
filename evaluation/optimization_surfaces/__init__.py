"""Concatenated optimization surfaces. Public catalog rows only."""

from evaluation.optimization_surfaces.enrollment import SURFACES as _ENROLLMENT
from evaluation.optimization_surfaces.evaluation import SURFACES as _EVALUATION
from evaluation.optimization_surfaces.gallery import SURFACES as _GALLERY
from evaluation.optimization_surfaces.identification import SURFACES as _IDENTIFICATION
from evaluation.optimization_surfaces.parsing import SURFACES as _PARSING
from evaluation.optimization_surfaces.representation import SURFACES as _REPRESENTATION
from evaluation.optimization_surfaces.runtime import SURFACES as _RUNTIME
from evaluation.optimization_surfaces.search import SURFACES as _SEARCH

SURFACES = (
    *_PARSING,
    *_IDENTIFICATION,
    *_REPRESENTATION,
    *_ENROLLMENT,
    *_GALLERY,
    *_SEARCH,
    *_RUNTIME,
    *_EVALUATION,
)
