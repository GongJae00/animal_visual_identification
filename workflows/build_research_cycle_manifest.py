"""Build a deterministic retrospective research-only admission manifest.

Commands: cycle (default), source-admissions.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from data.source_lock import get_record
from foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from identity.splits.protected_public_split import (
    FrozenPublicSplitEvidenceGraph,
    PublicSplitSourceBundle,
)
from identity.research.research_cycle_admission import (
    IdentityTargetMode,
    ResearchLicenseLane,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
    ResearchSourceRole,
    build_research_cycle_manifest,
)
from identity.exposure.role_exposure import RoleExposureLedger, RoleExposureReceipt

_SOURCE_MANIFEST_LIMITS = {
    "maximum_bytes": 536_870_912,
    "maximum_nodes": 10_000_000,
    "maximum_keys": 5_000_000,
    "maximum_array_length": 1_000_000,
}
_IDENTITY_DATASETS = frozenset(
    {"dogfacenet224", "mpdd", "sibetan", "yt-bb-dog"}
)
_ALL_DATASETS = _IDENTITY_DATASETS | {"ap10k-dog", "dogflw"}


def _run_cycle(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cycle-name", required=True)
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--dependency-graph", required=True, type=Path)
    parser.add_argument("--source-admissions", required=True, type=Path)
    parser.add_argument("--role-exposure-ledger", required=True, type=Path)
    parser.add_argument("--role-exposure-receipt", required=True, type=Path)
    parser.add_argument(
        "--source-manifest",
        action="append",
        nargs=2,
        required=True,
        metavar=("DATASET", "PATH"),
        help="source manifest whose canonical content hash is declared; repeat for all six datasets",
    )
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite research-cycle output: {args.output}")

    source_document = read_strict_json_document(args.source_bundle)
    graph_document = read_strict_json_document(args.dependency_graph)
    admissions_document = read_strict_json_document(args.source_admissions)
    ledger_document = read_strict_json_document(args.role_exposure_ledger)
    receipt_document = read_strict_json_document(args.role_exposure_receipt)
    source = PublicSplitSourceBundle.from_dict(source_document.payload)
    graph = FrozenPublicSplitEvidenceGraph.from_dict(graph_document.payload)
    admissions = ResearchSourceAdmissions.from_dict(admissions_document.payload)
    ledger = RoleExposureLedger.from_dict(ledger_document.payload)
    receipt = RoleExposureReceipt.from_dict(receipt_document.payload)

    declared = {
        item.dataset_name: item.source_manifest_sha256 for item in admissions.sources
    }
    observed: dict[str, str] = {}
    for dataset, path_text in args.source_manifest:
        if dataset in observed:
            raise ValueError(f"duplicate --source-manifest dataset: {dataset}")
        document = read_strict_json_document(
            Path(path_text), **_SOURCE_MANIFEST_LIMITS
        )
        observed[dataset] = document.canonical_payload_sha256
    if observed != declared:
        raise ValueError("source manifest content hashes differ from source admissions")

    manifest = build_research_cycle_manifest(
        cycle_name=args.cycle_name,
        source=source,
        graph=graph,
        source_admissions=admissions,
        role_exposure_ledger=ledger,
        role_exposure_receipt=receipt,
    )
    write_private_json_bundle(((args.output, manifest.to_dict()),))
    print(
        json.dumps(
            {
                "status": "CREATED_RESEARCH_CYCLE_MANIFEST",
                "manifest_sha256": manifest.manifest_sha256,
                "identity_count": len(manifest.identity_assignments),
                "sample_count": len(manifest.sample_assignments),
                "output": str(args.output),
                "final_evaluation_permitted": False,
            },
            sort_keys=True,
        )
    )
    return 0


def _run_source_admissions(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(
        description="Bind six explicit source manifests into retrospective research admissions."
    )
    parser.add_argument(
        "--source-manifest",
        action="append",
        nargs=2,
        required=True,
        metavar=("DATASET", "PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite research source admissions")
    paths: dict[str, Path] = {}
    for dataset, path_text in args.source_manifest:
        if dataset in paths:
            raise ValueError(f"duplicate source manifest dataset: {dataset}")
        paths[dataset] = Path(path_text)
    if set(paths) != _ALL_DATASETS:
        raise ValueError("research source admissions require all six datasets")
    admissions = ResearchSourceAdmissions(
        tuple(
            ResearchSourceAdmission(
                dataset_name=dataset,
                source_manifest_sha256=read_strict_json_document(
                    paths[dataset], **_SOURCE_MANIFEST_LIMITS
                ).canonical_payload_sha256,
                license_id=get_record(dataset).license_id,
                license_lane=ResearchLicenseLane.RESEARCH_ONLY,
                source_role=(
                    ResearchSourceRole.IDENTITY_RESEARCH
                    if dataset in _IDENTITY_DATASETS
                    else ResearchSourceRole.AUXILIARY_ONLY
                ),
                identity_target_mode=(
                    IdentityTargetMode.CANONICAL_REGISTERED_UUIDV5
                    if dataset in _IDENTITY_DATASETS
                    else IdentityTargetMode.NONE
                ),
            )
            for dataset in sorted(paths)
        )
    )
    write_private_json_bundle(((args.output, admissions.to_dict()),))
    print(
        json.dumps(
            {
                "status": "CREATED_RESEARCH_SOURCE_ADMISSIONS",
                "admissions_sha256": admissions.admissions_sha256,
                "datasets": sorted(paths),
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "cycle"
    if argv and argv[0] in {"cycle", "source-admissions"}:
        command = argv[0]
        argv = argv[1:]
    return {
        "cycle": _run_cycle,
        "source-admissions": _run_source_admissions,
    }[command](argv)


if __name__ == "__main__":
    raise SystemExit(main())
