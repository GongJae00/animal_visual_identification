"""Bind six explicit source manifests into retrospective research admissions."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.source_lock import get_record
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from identity.research.research_cycle_admission import (
    IdentityTargetMode,
    ResearchLicenseLane,
    ResearchSourceAdmission,
    ResearchSourceAdmissions,
    ResearchSourceRole,
)

_IDENTITY_DATASETS = frozenset(
    {"dogfacenet224", "mpdd", "sibetan", "yt-bb-dog"}
)
_ALL_DATASETS = _IDENTITY_DATASETS | {"ap10k-dog", "dogflw"}
_SOURCE_MANIFEST_LIMITS = {
    "maximum_bytes": 536_870_912,
    "maximum_nodes": 10_000_000,
    "maximum_keys": 5_000_000,
    "maximum_array_length": 1_000_000,
}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--source-manifest",
        action="append",
        nargs=2,
        required=True,
        metavar=("DATASET", "PATH"),
    )
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
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


if __name__ == "__main__":
    raise SystemExit(main())
