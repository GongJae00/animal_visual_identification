"""Evaluate a content-addressed first-order IdentityEngine capacity plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.capacity import CapacityPlan


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--duration-seconds", required=True, type=float)
    args = parser.parse_args()

    payload = json.loads(
        args.plan.resolve(strict=True).read_text(encoding="utf-8")
    )
    if not isinstance(payload, dict):
        raise TypeError("capacity plan root must be an object")
    plan = CapacityPlan.from_dict(payload)
    output = {
        "schema_version": "cvi.capacity_evaluation.v1",
        "plan_sha256": plan.config_sha256,
        "duration_seconds": args.duration_seconds,
        "expected_stage_calls": plan.expected_stage_calls(
            args.duration_seconds
        ),
        "peak_state_stage_calls": plan.peak_state_stage_calls(
            args.duration_seconds
        ),
        "resource_loads": [
            load.to_dict() for load in plan.resource_loads()
        ],
    }
    print(
        json.dumps(output, ensure_ascii=False, sort_keys=True, indent=2)
    )


if __name__ == "__main__":
    main()
