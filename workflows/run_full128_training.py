"""Train and cache selected Full128 B0/B1/B2 variants from an assembly."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import read_strict_json_document
from embedding.methods.full_segment.training.artifacts import (
    default_full128_run_config,
    validate_full128_run_config,
)
from embedding.methods.full_segment.preparation.data import load_full128_assembly
from embedding.learning.full_segment.full128 import run_full128_training


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--print-default-config",
        action="store_true",
        help="print the current factual protocol defaults and exit",
    )
    parser.add_argument("--assembly", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--variants", nargs="+", choices=("B0", "B1", "B2"))
    parser.add_argument("--b2-checkpoint", type=Path)
    parser.add_argument("--b2-intake-bundle", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    if args.print_default_config:
        if any(
            value is not None
            for value in (
                args.assembly,
                args.config,
                args.output_dir,
                args.variants,
                args.b2_checkpoint,
                args.b2_intake_bundle,
            )
        ):
            raise ValueError("--print-default-config cannot be combined with run inputs")
        print(json.dumps(default_full128_run_config(), indent=2, sort_keys=True))
        return 0
    if any(
        value is None
        for value in (args.assembly, args.config, args.output_dir, args.variants)
    ):
        raise ValueError("--assembly, --config, --output-dir, and --variants are required")

    repository = Path(__file__).resolve().parents[1]
    config_path = args.config.absolute()
    config = validate_full128_run_config(
        read_strict_json_document(config_path, maximum_bytes=1_048_576).payload
    )
    assembly = _external_input(args.assembly, repository, "Full128 assembly")
    inventory, _ = load_full128_assembly(
        assembly,
        validation_workers=max(1, config["workers"]),
    )
    result = run_full128_training(
        inventory=inventory,
        run_config=config,
        output_dir=args.output_dir,
        variants=args.variants,
        b2_checkpoint_path=(
            None
            if args.b2_checkpoint is None
            else _external_input(args.b2_checkpoint, repository, "B2 checkpoint")
        ),
        b2_intake_bundle_path=(
            None
            if args.b2_intake_bundle is None
            else _external_input(
                args.b2_intake_bundle, repository, "B2 intake bundle"
            )
        ),
    )
    print(
        json.dumps(
            {
                "status": (
                    "FULL128_EXACT_FAMILY_COMPLETE"
                    if result["family_complete"]
                    else "FULL128_SELECTED_VARIANTS_COMPLETE_FAMILY_INCOMPLETE"
                ),
                **result,
            },
            sort_keys=True,
        )
    )
    return 0


def _external_input(path: Path, repository: Path, label: str) -> Path:
    requested = path.absolute()
    if requested.is_symlink():
        raise ValueError(f"{label} must not be a symlink")
    resolved = requested.resolve(strict=True)
    if not resolved.is_file():
        raise ValueError(f"{label} must be a regular file")
    if resolved == repository or resolved.is_relative_to(repository):
        raise ValueError(f"{label} must remain outside the repository")
    return resolved


if __name__ == "__main__":
    raise SystemExit(main())
