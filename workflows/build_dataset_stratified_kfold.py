"""Build a retrospective dataset-stratified identity K-fold manifest."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from identity.research.dataset_stratified_kfold import (
    DatasetStratifiedKFoldPolicy,
    build_dataset_stratified_identity_kfold,
    dataset_stratified_kfold_bundle,
)
from identity.research.research_cycle_admission import ResearchCycleManifest


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-name", required=True)
    parser.add_argument("--research-cycle", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--fold-count", type=int, default=5)
    parser.add_argument("--dev-offset", type=int, default=1)
    parser.add_argument("--gallery-fraction-numerator", type=int, default=1)
    parser.add_argument("--gallery-fraction-denominator", type=int, default=2)
    parser.add_argument("--minimum-identities-per-fold", type=int, default=1)
    parser.add_argument("--minimum-retrieval-identities-per-fold", type=int, default=1)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError("refusing to overwrite K-fold output")
    document = read_strict_json_document(
        args.research_cycle,
        maximum_bytes=2_147_483_648,
        maximum_nodes=25_000_000,
        maximum_keys=10_000_000,
        maximum_array_length=1_000_000,
    )
    cycle = ResearchCycleManifest.from_dict(document.payload)
    policy = DatasetStratifiedKFoldPolicy(
        fold_count=args.fold_count,
        dev_offset=args.dev_offset,
        gallery_fraction_numerator=args.gallery_fraction_numerator,
        gallery_fraction_denominator=args.gallery_fraction_denominator,
        minimum_identities_per_fold=args.minimum_identities_per_fold,
        minimum_retrieval_identities_per_fold=(
            args.minimum_retrieval_identities_per_fold
        ),
    )
    manifest = build_dataset_stratified_identity_kfold(
        protocol_name=args.protocol_name,
        research_cycle=cycle,
        policy=policy,
    )
    write_private_json_bundle(((args.output, dataset_stratified_kfold_bundle(manifest)),))
    print(
        json.dumps(
            {
                "status": "CREATED_DATASET_STRATIFIED_IDENTITY_KFOLD",
                "manifest_sha256": manifest.manifest_sha256,
                "fold_count": policy.fold_count,
                "identity_count": len(manifest.identity_assignments),
                "sample_count": len(manifest.sample_assignments),
                "final_evaluation_permitted": False,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
