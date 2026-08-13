"""Summarize chronological G1 observations from JSON Lines."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evidence_fusion.coverage import (
    CoverageAccumulator,
    CoverageObservation,
    CoveragePolicy,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--observations", required=True, type=Path)
    parser.add_argument("--timeline-start-ns", required=True, type=int)
    parser.add_argument("--timeline-end-ns", required=True, type=int)
    args = parser.parse_args()

    policy_payload = json.loads(
        args.policy.resolve(strict=True).read_text(encoding="utf-8")
    )
    accumulator = CoverageAccumulator(
        CoveragePolicy.from_dict(policy_payload),
        timeline_start_ns=args.timeline_start_ns,
    )
    with args.observations.resolve(strict=True).open(encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                accumulator.observe(CoverageObservation.from_dict(payload))
            except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
                raise ValueError(
                    f"invalid observation at line {line_number}: {error}"
                ) from error
    print(
        json.dumps(
            accumulator.finalize(timeline_end_ns=args.timeline_end_ns),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
