"""Build a deterministic retrospective research-only admission manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from cvi.protected_public_split import (
    FrozenPublicSplitEvidenceGraph,
    PublicSplitSourceBundle,
)
from cvi.research_cycle_admission import (
    ResearchSourceAdmissions,
    build_research_cycle_manifest,
)
from cvi.role_exposure import RoleExposureLedger, RoleExposureReceipt


def main() -> int:
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
    args = parser.parse_args()

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
        document = read_strict_json_document(Path(path_text))
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


if __name__ == "__main__":
    raise SystemExit(main())
