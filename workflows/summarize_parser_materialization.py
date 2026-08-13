"""Summarize parser routes, terminal reasons, and v3 selection lineage."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

from foundation.protected_io import read_strict_json_document
from foundation.provenance import content_sha256
from embedding.methods.full_segment.preparation.materialization import read_route_plan_bundle
from data.full_segment.route_plan import validate_full128_route_plan_bundle

REPORT_SCHEMA = "cvi.parser_materialization_summary.v1"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", type=Path, required=True)
    parser.add_argument("--materialization-root", type=Path, required=True)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()
    report = summarize(
        route_plan=args.route_plan,
        materialization_root=args.materialization_root,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(report, sort_keys=True))
    return 0


def summarize(
    *, route_plan: Path, materialization_root: Path, allow_partial: bool
) -> dict[str, Any]:
    bundle = validate_full128_route_plan_bundle(
        read_route_plan_bundle(route_plan.absolute()), verify_files=False
    )
    root = materialization_root.resolve(strict=True)
    rows = {row["sample_token"]: row for row in bundle["plan"]["records"]}
    route_counts: Counter[str] = Counter()
    reason_counts: Counter[str] = Counter()
    selection_rule_counts: Counter[str] = Counter()
    reason_dataset_counts: Counter[tuple[str, str]] = Counter()
    missing = []
    receipt_hashes = []
    for token, row in rows.items():
        path = root / "samples" / token / "execution-receipt.json"
        if not path.is_file() or path.is_symlink():
            missing.append(token)
            continue
        document = read_strict_json_document(path, maximum_bytes=4_194_304)
        receipt = document.payload
        payload = {key: value for key, value in receipt.items() if key != "receipt_sha256"}
        if (
            receipt.get("receipt_sha256") != content_sha256(payload)
            or receipt.get("sample_token") != token
            or receipt.get("plan_record_sha256") != row["record_sha256"]
            or receipt.get("plan_sha256") != bundle["plan_sha256"]
        ):
            raise ValueError("parser materialization receipt binding differs")
        receipt_hashes.append(document.raw_sha256)
        route_counts[receipt["actual_route"]] += 1
        reason = receipt["terminal_reason"]
        if reason is not None:
            reason_counts[reason] += 1
            reason_dataset_counts[(reason, row["dataset_name"])] += 1
        lineage = receipt["parser_lineage"]
        if row["schema_version"] == "cvi.full128_route_plan_record.v3":
            if not isinstance(lineage, dict) or not isinstance(
                lineage.get("selection"), dict
            ):
                raise ValueError("parser v3 receipt lacks selection lineage")
            selection_rule_counts[lineage["selection"]["rule"]] += 1
    if missing and not allow_partial:
        raise ValueError("parser materialization is incomplete")
    return {
        "schema_version": REPORT_SCHEMA,
        "route_plan_sha256": bundle["plan_sha256"],
        "route_plan_record_count": len(rows),
        "observed_receipt_count": len(receipt_hashes),
        "missing_receipt_count": len(missing),
        "missing_sample_tokens_sha256": content_sha256(sorted(missing)),
        "receipt_file_sha256s_sha256": content_sha256(sorted(receipt_hashes)),
        "actual_route_counts": dict(sorted(route_counts.items())),
        "terminal_reason_counts": dict(sorted(reason_counts.items())),
        "terminal_reason_dataset_counts": {
            f"{reason}|{dataset}": count
            for (reason, dataset), count in sorted(reason_dataset_counts.items())
        },
        "selection_rule_counts": dict(sorted(selection_rule_counts.items())),
    }


if __name__ == "__main__":
    raise SystemExit(main())
