"""Prepare a strict external cache of student-mask Nose embedding supports."""

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
    parser.add_argument("--student-lineage", type=Path, required=True)
    parser.add_argument("--student-lineage-sha256", required=True)
    parser.add_argument("--student-root", type=Path, required=True)
    parser.add_argument("--mask-manifest", type=Path, required=True)
    parser.add_argument("--mask-manifest-sha256", required=True)
    parser.add_argument("--mask-onnx", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> None:
    args = parse_args(argv)
    # Keep ONNX Runtime and training-lineage dependencies out of the help path.
    from identification.export.nose.data.embedding_views import prepare_embedding_views
    from shared.foundation.provenance import content_sha256

    manifest = prepare_embedding_views(
        native_bundle_path=args.native_bundle,
        native_bundle_sha256=args.native_bundle_sha256,
        native_root=args.native_root,
        student_lineage_path=args.student_lineage,
        student_lineage_sha256=args.student_lineage_sha256,
        student_root=args.student_root,
        mask_manifest_path=args.mask_manifest,
        mask_manifest_sha256=args.mask_manifest_sha256,
        mask_onnx_path=args.mask_onnx,
        output_dir=args.output_dir,
        use_cuda=args.device == "cuda",
    )
    print(
        json.dumps(
            {
                "status": "CREATED",
                "output_dir": str(args.output_dir),
                "manifest": str(args.output_dir / "nose-embedding-views.json"),
                "manifest_sha256": manifest["manifest_sha256"],
                "manifest_payload_sha256": content_sha256(manifest),
                "record_count": manifest["record_count"],
            },
            allow_nan=False,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
