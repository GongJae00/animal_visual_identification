"""Build one protected metadata-only Full128 route plan from admitted datasets."""

from __future__ import annotations

import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import write_private_json_bundle
from data.full_segment.route_plan import build_full128_route_plan


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parser-runtime-manifest-sha256", required=True)
    parser.add_argument("--parser-policy-sha256", required=True)
    parser.add_argument("--maximum-samples-per-dataset", type=int)
    parser.add_argument("--dogface-classes-train", type=Path)
    parser.add_argument("--dogface-classes-test", type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    repository = Path(__file__).resolve().parents[1]
    requested_output = args.output.absolute()
    output = requested_output.parent.resolve(strict=True) / requested_output.name
    if output == repository or output.is_relative_to(repository):
        raise ValueError("Full128 route-plan output must remain outside the repository")
    if requested_output.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full128 route plan: {output}")

    data_dir_value = os.environ.get("CANINE_IDENTITY_DATA_DIR")
    if data_dir_value is None:
        raise ValueError(
            "CANINE_IDENTITY_DATA_DIR must be set before workflow process import"
        )
    data_dir = Path(data_dir_value)
    if not data_dir.is_absolute():
        raise ValueError("CANINE_IDENTITY_DATA_DIR must be absolute")

    bundle = build_full128_route_plan(
        parser_runtime_manifest_sha256=args.parser_runtime_manifest_sha256,
        parser_policy_sha256=args.parser_policy_sha256,
        maximum_samples_per_dataset=args.maximum_samples_per_dataset,
        dogface_classes_train_path=args.dogface_classes_train,
        dogface_classes_test_path=args.dogface_classes_test,
    )
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FULL128_ROUTE_PLAN",
                "plan_sha256": bundle["plan_sha256"],
                "record_count": len(bundle["plan"]["records"]),
                "selection": bundle["plan"]["selection"]["mode"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
