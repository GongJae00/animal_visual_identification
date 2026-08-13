"""Audit native YT Nose signal quality and mask integrity without identity labels."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.nose_signal_quality import analyze_native_nose_signal
from parsing.nose_region.native_yt import validate_manifest_bundle
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--artifacts-root", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    document = read_strict_json_document(
        args.manifest,
        maximum_bytes=536_870_912,
        maximum_nodes=10_000_000,
        maximum_keys=5_000_000,
        maximum_array_length=1_000_000,
    )
    bundle = document.payload
    if bundle.get("manifest_sha256") != args.manifest_sha256:
        raise ValueError("native YT Nose manifest differs from external pin")
    manifest = validate_manifest_bundle(bundle, root=args.artifacts_root)
    report = analyze_native_nose_signal(manifest, artifacts_root=args.artifacts_root)
    report["source_binding"] = {
        "path": os.fspath(args.manifest),
        "file_sha256": document.raw_sha256,
        "manifest_sha256": args.manifest_sha256,
    }
    report["code_sha256s"] = {
        relative: _sha(Path(__file__).resolve().parents[1] / relative)
        for relative in (
            "experiments/nose_signal_quality.py",
            "parsing/nose_region/manifest.py",
            "parsing/nose_region/native_yt.py",
            "workflows/audit_yt_nose_signal_quality.py",
        )
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    output = {
        "schema_version": "cvi.nose_signal_quality_report_bundle.v1",
        "report_sha256": content_sha256(report),
        "report": report,
    }
    write_private_json_bundle(((args.output, output),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(args.output),
                "report_sha256": output["report_sha256"],
                "population": report["population"],
                "policy_attrition": report["policy_attrition"],
            },
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
