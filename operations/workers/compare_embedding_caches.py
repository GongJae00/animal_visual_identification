"""Create a protected label-blind embedding numerical-admission receipt."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.controls.control_scoring import EmbeddingCacheManifest
from prototype.export.embedding_producer import EmbeddingProducerConfig
from evaluation.integrity.numerical_admission import NumericalDriftPolicy, compare_embedding_caches
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference-cache-directory", required=True, type=Path)
    parser.add_argument("--candidate-cache-directory", required=True, type=Path)
    parser.add_argument("--reference-cache-manifest", required=True, type=Path)
    parser.add_argument("--candidate-cache-manifest", required=True, type=Path)
    parser.add_argument("--reference-producer-config", required=True, type=Path)
    parser.add_argument("--candidate-producer-config", required=True, type=Path)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)

    receipt = compare_embedding_caches(
        reference_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.reference_cache_manifest)
        ),
        candidate_manifest=EmbeddingCacheManifest.from_dict(
            read_strict_json_object(args.candidate_cache_manifest)
        ),
        reference_config=EmbeddingProducerConfig.from_dict(
            read_strict_json_object(args.reference_producer_config)
        ),
        candidate_config=EmbeddingProducerConfig.from_dict(
            read_strict_json_object(args.candidate_producer_config)
        ),
        reference_root=args.reference_cache_directory,
        candidate_root=args.candidate_cache_directory,
        policy=NumericalDriftPolicy.from_dict(
            read_strict_json_object(args.policy)
        ),
    )
    output = {
        "schema_version": "cvi.numerical_admission_bundle.v1",
        "receipt_sha256": receipt.receipt_sha256,
        "receipt": receipt.to_dict(),
    }
    write_private_json_bundle(((args.receipt, output),))
    print(
        json.dumps(
            {
                "status": "CREATED",
                "decision": receipt.decision.value,
                "receipt_sha256": receipt.receipt_sha256,
                "vectors": receipt.summary.vectors,
                "values": receipt.summary.values,
                "bytes_read": receipt.summary.bytes_read,
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
