"""Build the canonical six-dataset retrospective research task plan."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from evaluation.splits.research.research_cycle_admission import ResearchSourceAdmissions
from evaluation.splits.research.research_task_plan import build_primary_research_task_plan


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-admissions", required=True, type=Path)
    parser.add_argument("--plan-name", default="six-dataset-robust-reid-v1")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    if args.output.exists() or args.output.is_symlink():
        raise FileExistsError(f"refusing to overwrite research task plan: {args.output}")

    admissions = ResearchSourceAdmissions.from_dict(
        read_strict_json_object(args.source_admissions)
    )
    plan = build_primary_research_task_plan(admissions, plan_name=args.plan_name)
    write_private_json_bundle(((args.output, plan.to_dict()),))
    print(
        json.dumps(
            {
                "status": "CREATED_RESEARCH_TASK_PLAN",
                "plan_sha256": plan.plan_sha256,
                "dataset_count": len({item.dataset_name for item in plan.assignments}),
                "assignment_count": len(plan.assignments),
                "final_evaluation_permitted": plan.final_evaluation_permitted,
                "output": str(args.output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
