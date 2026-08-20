"""Optimization evaluation catalog: stage × axis × lever × status.

This module is a protocol ledger, not a results table. It does not record
measured values. Backbone choice is out of scope; listed surfaces stay if
the encoder changes. Not biometric validation and not a performance claim.

CLI: ``uv run python -m evaluation.commands.evaluate optimization-protocol --help``
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

from shared.foundation.protected_io import write_private_json_bundle

PROTOCOL_SCHEMA = "evaluation.optimization_protocol.v1"
INTERPRETATION = (
    "OPTIMIZATION_CATALOG_NOT_MEASURED_VALUES_NOT_BIOMETRIC_VALIDATION_NOT_A_PERFORMANCE_CLAIM"
)

STAGE_IDS: tuple[str, ...] = (
    "parsing.detection",
    "parsing.segmentation",
    "parsing.regions",
    "parsing.quality",
    "parsing.crops",
    "identification.appearance",
    "identification.face",
    "identification.nose",
    "representation.evidence",
    "representation.channels",
    "representation.quality",
    "enrollment.registry",
    "enrollment.write",
    "gallery.store",
    "search.scoring",
    "search.matching",
    "prototype.runtime",
    "operations.measurement",
    "evaluation.integrity",
)

_RUNTIME_STAGE_IDS = frozenset(
    {"prototype.runtime", "operations.measurement", "evaluation.integrity"}
)

_VIS_SUBSTAGES: dict[str, tuple[str, str]] = {
    "parsing.detection": ("parsing", "00_detection"),
    "parsing.segmentation": ("parsing", "01_segmentation"),
    "parsing.regions": ("parsing", "02_regions"),
    "parsing.quality": ("parsing", "03_quality"),
    "parsing.crops": ("parsing", "04_crops"),
    "identification.appearance": ("identification", "00_appearance"),
    "identification.face": ("identification", "01_face"),
    "identification.nose": ("identification", "02_nose"),
    "representation.evidence": ("representation", "00_evidence"),
    "representation.channels": ("representation", "01_channels"),
    "representation.quality": ("representation", "02_quality"),
    "enrollment.registry": ("enrollment", "00_registry"),
    "enrollment.write": ("enrollment", "01_write"),
    "gallery.store": ("gallery", "00_store"),
    "search.scoring": ("search", "00_scoring"),
    "search.matching": ("search", "01_matching"),
}

_VIS_STAGE_ORDER = (
    "parsing",
    "identification",
    "representation",
    "enrollment",
    "gallery",
    "search",
)


class OptimizationAxis(StrEnum):
    parallel_sequential = "parallel_sequential"
    memory = "memory"
    latency = "latency"
    compute = "compute"
    code = "code"
    execution = "execution"
    evaluation = "evaluation"
    cuda = "cuda"
    math = "math"
    io = "io"
    session = "session"
    batch = "batch"


class SurfaceStatus(StrEnum):
    wired = "wired"
    admitted = "admitted"
    forbidden = "forbidden"
    out_of_product_boundary = "out_of_product_boundary"


@dataclass(frozen=True, slots=True)
class OptimizationSurface:
    surface_id: str
    stage_id: str
    axis: OptimizationAxis
    lever: str
    survives_backbone_swap: bool
    status: SurfaceStatus
    owner: str
    measurement: str
    constraint: str

    def __post_init__(self) -> None:
        if self.stage_id not in STAGE_IDS:
            raise ValueError(f"unknown optimization stage: {self.stage_id}")
        if not self.surface_id or not self.lever or not self.owner:
            raise ValueError("optimization surface fields must be non-empty")
        if not self.measurement or not self.constraint:
            raise ValueError("optimization surface measurement and constraint must be non-empty")
        if type(self.survives_backbone_swap) is not bool:
            raise TypeError("survives_backbone_swap must be bool")

    def to_dict(self) -> dict[str, str | bool]:
        return {
            "surface_id": self.surface_id,
            "stage_id": self.stage_id,
            "axis": str(self.axis),
            "lever": self.lever,
            "survives_backbone_swap": self.survives_backbone_swap,
            "status": str(self.status),
            "owner": self.owner,
            "measurement": self.measurement,
            "constraint": self.constraint,
        }


def _load_surfaces() -> tuple[OptimizationSurface, ...]:
    from evaluation.optimization_surfaces import SURFACES

    return SURFACES


def protocol_document() -> dict[str, Any]:
    surfaces = _load_surfaces()
    ids = tuple(surface.surface_id for surface in surfaces)
    if len(ids) != len(set(ids)):
        raise RuntimeError("optimization surface_id values must be unique")
    return {
        "schema_version": PROTOCOL_SCHEMA,
        "interpretation": INTERPRETATION,
        "backbone_independent": True,
        "stage_ids": list(STAGE_IDS),
        "axes": [axis.value for axis in OptimizationAxis],
        "surfaces": [surface.to_dict() for surface in surfaces],
    }


def visualization_trace() -> dict[str, Any]:
    surfaces = _load_surfaces()
    substages: dict[str, dict[str, list[dict[str, str | bool]]]] = {}
    for stage_name in _VIS_STAGE_ORDER:
        bucket: dict[str, list[dict[str, str | bool]]] = {}
        for owner_stage, vis in _VIS_SUBSTAGES.values():
            if owner_stage == stage_name and vis not in bucket:
                bucket[vis] = []
        substages[stage_name] = bucket
    runtime: list[dict[str, str | bool]] = []
    for surface in surfaces:
        payload = surface.to_dict()
        if surface.stage_id in _RUNTIME_STAGE_IDS:
            runtime.append(payload)
            continue
        mapped = _VIS_SUBSTAGES.get(surface.stage_id)
        if mapped is None:
            raise RuntimeError(f"unmapped optimization stage: {surface.stage_id}")
        stage_name, vis = mapped
        substages[stage_name][vis].append(payload)
    return {
        "protocol": protocol_document(),
        "substages": substages,
        "runtime": runtime,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m evaluation.commands.evaluate optimization-protocol",
        description=__doc__,
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Visualization-trace JSON (protocol + per-substage surfaces).",
    )
    args = parser.parse_args(argv)
    trace = visualization_trace()
    write_private_json_bundle(((args.output, trace),))
    print(
        json.dumps(
            {
                "event": "optimization_protocol_written",
                "schema_version": PROTOCOL_SCHEMA,
                "output": str(args.output),
                "surfaces": len(trace["protocol"]["surfaces"]),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
