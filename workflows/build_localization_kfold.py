"""Build AP-10K and DogFLW source-group-safe localization K-folds."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data.adapters import ADAPTERS
from data.source_lock import get_record
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from identity_governance.research_cycle_admission import ResearchSourceAdmissions
from localization.fold_protocol import (
    LocalizationKFoldPolicy,
    build_localization_kfold_manifest,
    localization_kfold_bundle,
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--source-admissions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--dev-offset", type=int, default=1)
    parser.add_argument(
        "--source-manifest",
        action="append",
        nargs=2,
        required=True,
        metavar=("DATASET", "PATH"),
    )
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite localization K-fold output")
    admissions = ResearchSourceAdmissions.from_dict(
        read_strict_json_document(args.source_admissions).payload
    )
    source_hashes = {
        item.dataset_name: item.source_manifest_sha256
        for item in admissions.sources
        if item.dataset_name in {"ap10k-dog", "dogflw"}
    }
    observed: dict[str, str] = {}
    source_manifests: dict[str, dict[str, object]] = {}
    for dataset, path_text in args.source_manifest:
        if dataset in observed:
            raise ValueError(f"duplicate --source-manifest dataset: {dataset}")
        document = read_strict_json_document(Path(path_text))
        observed[dataset] = document.canonical_payload_sha256
        source_manifests[dataset] = document.payload
    if observed != source_hashes:
        raise ValueError("localization source manifest hashes differ from admissions")
    samples = tuple(
        sample
        for dataset in ("ap10k-dog", "dogflw")
        for sample in ADAPTERS[dataset](Path(get_record(dataset).data_root))
    )
    manifest = build_localization_kfold_manifest(
        samples,
        protocol_name=args.protocol_name,
        policy=LocalizationKFoldPolicy(
            fold_count=args.fold_count, dev_offset=args.dev_offset
        ),
        source_manifests=source_manifests,
    )
    write_private_json_bundle(((args.output, localization_kfold_bundle(manifest)),))
    print(
        json.dumps(
            {
                "status": "CREATED_LOCALIZATION_KFOLD",
                "manifest_sha256": manifest.manifest_sha256,
                "sample_count": len(manifest.assignments),
                "fold_count": manifest.policy.fold_count,
                "identity_target_mode": "NONE",
                "final_evaluation_permitted": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
