"""Create paired DEV/EVAL bootstrap comparisons from a bound YT masked report."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

from experiments.masked_comparison import compare_methods_to_appearance
from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.provenance import content_sha256


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--policy-report", type=Path, required=True)
    parser.add_argument("--policy-report-sha256", required=True)
    parser.add_argument("--resamples", type=int, default=10_000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(args.output)
    document = read_strict_json_document(args.policy_report)
    bundle = document.payload
    if (
        not isinstance(bundle, dict)
        or bundle.get("schema_version") not in {
            "cvi.yt_masked_multievidence_policy_bundle.v1",
            "cvi.yt_masked_multievidence_policy_bundle.v2",
        }
        or bundle.get("report_sha256") != args.policy_report_sha256
        or content_sha256(bundle.get("report")) != args.policy_report_sha256
    ):
        raise ValueError("YT masked policy report differs from external pin")
    report = bundle["report"]
    if report.get("evaluation") is None:
        raise ValueError("YT masked policy has no EVAL result")
    summary = {
        "schema_version": "cvi.yt_masked_multievidence_paired_comparison.v1",
        "status": "PASS_YT_MASKED_PAIRED_COMPARISON",
        "interpretation": "YT_TRACK_PROXY_PAIRED_RESEARCH_COMPARISON_NOT_LIFELONG_IDENTITY_VALIDATION",
        "source_binding": {
            "path": os.fspath(args.policy_report),
            "file_sha256": document.raw_sha256,
            "report_sha256": args.policy_report_sha256,
        },
        "bootstrap": {"resamples": args.resamples, "seed": args.seed, "cluster_unit": "registered_dog_id"},
        "dev": compare_methods_to_appearance(report["dev"], resamples=args.resamples, seed=args.seed),
        "evaluation": compare_methods_to_appearance(report["evaluation"], resamples=args.resamples, seed=args.seed + 100),
        "code_sha256s": {
            relative: _sha(Path(__file__).resolve().parents[1] / relative)
            for relative in (
                "experiments/masked_comparison.py",
                "evaluation/retrieval.py",
                "workflows/summarize_yt_masked_multievidence.py",
            )
        },
    }
    summary = json.loads(json.dumps(summary, allow_nan=False))
    output = {"schema_version": "cvi.yt_masked_multievidence_paired_comparison_bundle.v1", "report_sha256": content_sha256(summary), "report": summary}
    write_private_json_bundle(((args.output, output),))
    print(json.dumps({"status": summary["status"], "output": os.fspath(args.output), "report_sha256": output["report_sha256"], "evaluation": summary["evaluation"]}, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
