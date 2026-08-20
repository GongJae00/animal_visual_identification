"""Materialize exact N3 TRAIN/DEV embeddings for the bounded N4 adapter."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--native-bundle", type=Path, required=True)
    parser.add_argument("--native-bundle-sha256", required=True)
    parser.add_argument("--native-root", type=Path, required=True)
    parser.add_argument("--n3-lineage", type=Path, required=True)
    parser.add_argument("--n3-lineage-sha256", required=True)
    parser.add_argument("--n3-runtime-manifest", type=Path, required=True)
    parser.add_argument("--n3-runtime-manifest-sha256", required=True)
    parser.add_argument("--n3-onnx", type=Path, required=True)
    parser.add_argument("--n3-onnx-sha256", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--use-cuda", action=argparse.BooleanOptionalAction, default=False
    )
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    from archive.nose_metric.experiments.n4_metric_adapter import materialize_embedding_cache

    bundle = materialize_embedding_cache(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        n3_lineage_path=args.n3_lineage,
        n3_lineage_sha256=args.n3_lineage_sha256,
        n3_runtime_manifest_path=args.n3_runtime_manifest,
        n3_runtime_manifest_sha256=args.n3_runtime_manifest_sha256,
        n3_onnx_path=args.n3_onnx,
        n3_onnx_sha256=args.n3_onnx_sha256,
        output_dir=args.output_dir,
        use_cuda=args.use_cuda,
    )
    print(
        json.dumps(
            {
                "status": bundle["cache"]["status"],
                "cache_sha256": bundle["cache_sha256"],
                "row_count": len(bundle["cache"]["rows"]),
                "output_dir": str(args.output_dir),
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
