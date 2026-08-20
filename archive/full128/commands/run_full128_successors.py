"""Train, resume, smoke-test, or inspect the Full128 successor family."""

from __future__ import annotations

import argparse
import json
from collections.abc import Sequence
from pathlib import Path

from archive.full128.evaluation.full128_successors import (
    build_successor_embedding_cache_descriptor,
    validate_fixed_evaluation_panel,
)
from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256
from archive.full128.methods.models.successor_models import (
    build_b2_fv,
    dinov2_contract_bindings,
    load_receipt_bound_dinov2_patch_backbone,
)
from archive.full128.learning.full128_successor_production import (
    B5_PARENTS,
    PRODUCTION_CANDIDATES,
    default_production_config,
    prepare_production_runtime,
    run_successor_production,
    validate_production_config,
)
from archive.full128.learning.full128_successors import (
    build_successor_family_manifest,
    default_successor_training_config,
    smoke_successor_execution,
)

_LARGE_JSON = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    modes = parser.add_mutually_exclusive_group(required=True)
    modes.add_argument("--print-default-config", action="store_true")
    modes.add_argument("--print-smoke-config", action="store_true")
    modes.add_argument("--print-family-manifest", action="store_true")
    modes.add_argument("--smoke-output", type=Path)
    modes.add_argument("--check-prerequisites", action="store_true")
    modes.add_argument("--production-output", type=Path)
    modes.add_argument("--print-production-config", action="store_true")
    parser.add_argument("--successor-inventory", type=Path)
    population = parser.add_mutually_exclusive_group()
    population.add_argument("--evaluation-panel", type=Path)
    population.add_argument("--required-sample-population", type=Path)
    parser.add_argument("--config", type=Path)
    parser.add_argument("--candidates", nargs="+", choices=PRODUCTION_CANDIDATES)
    parser.add_argument("--b5-parent-id", choices=B5_PARENTS)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--real-smoke", action="store_true")
    parser.add_argument("--real-smoke-fit-limit", type=int)
    parser.add_argument("--b2-checkpoint", type=Path)
    parser.add_argument("--b2-intake-bundle", type=Path)
    parser.add_argument("--dinov2-model-directory", type=Path)
    parser.add_argument("--dinov2-weight-intake-bundle", type=Path)
    parser.add_argument("--dinov2-preprocessor-intake-bundle", type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    source_args = (
        args.b2_checkpoint,
        args.b2_intake_bundle,
        args.dinov2_model_directory,
        args.dinov2_weight_intake_bundle,
        args.dinov2_preprocessor_intake_bundle,
    )
    if args.print_default_config:
        _reject_sources(source_args)
        print(json.dumps(default_successor_training_config(), indent=2, sort_keys=True))
        return 0
    if args.print_smoke_config:
        _reject_sources(source_args)
        print(
            json.dumps(
                default_successor_training_config(smoke=True), indent=2, sort_keys=True
            )
        )
        return 0
    if args.print_family_manifest:
        _reject_sources(source_args)
        print(json.dumps(build_successor_family_manifest(), indent=2, sort_keys=True))
        return 0
    if args.print_production_config:
        _reject_sources(source_args)
        print(json.dumps(default_production_config(), indent=2, sort_keys=True))
        return 0
    if args.smoke_output is not None:
        _reject_sources(source_args)
        receipt = smoke_successor_execution(args.smoke_output)
        print(
            json.dumps(
                {
                    "status": "FULL128_SUCCESSOR_SYNTHETIC_SMOKE_COMPLETE",
                    "output_dir": str(args.smoke_output.absolute()),
                    "smoke_receipt_sha256": receipt["smoke_receipt_sha256"],
                },
                sort_keys=True,
            )
        )
        return 0

    production_config = None
    if args.production_output is not None:
        if any(
            value is None
            for value in (
                args.successor_inventory,
                args.config,
                args.candidates,
                args.b5_parent_id,
            )
        ) or (
            args.evaluation_panel is None and args.required_sample_population is None
        ):
            raise ValueError(
                "production requires inventory, evaluation population, config, "
                "candidates, and B5 parent"
            )
        production_config = validate_production_config(
            read_strict_json_document(args.config, maximum_bytes=1_048_576).payload
        )
        if args.seed is not None:
            if args.seed < 0:
                raise ValueError("production seed override must be non-negative")
            production_config = {**production_config, "seed": args.seed}
        if args.real_smoke:
            if args.real_smoke_fit_limit is None:
                raise ValueError("--real-smoke requires --real-smoke-fit-limit")
            production_config = {
                **production_config,
                "supervised_steps": 1,
                "ssl_steps": 1,
            }
        elif args.real_smoke_fit_limit is not None:
            raise ValueError("--real-smoke-fit-limit requires --real-smoke")
        prepare_production_runtime(production_config)
    if any(value is None for value in source_args):
        raise ValueError(
            "prerequisite and production modes require B2 and all DINOv2 inputs"
        )
    b2 = build_b2_fv(
        args.b2_checkpoint,
        intake_bundle_path=args.b2_intake_bundle,
    )
    backbone, contract = load_receipt_bound_dinov2_patch_backbone(
        model_directory=args.dinov2_model_directory,
        weight_intake_bundle=args.dinov2_weight_intake_bundle,
        preprocessor_intake_bundle=args.dinov2_preprocessor_intake_bundle,
    )
    if args.production_output is not None:
        inventory = read_strict_json_document(
            args.successor_inventory, **_LARGE_JSON
        ).payload
        assert production_config is not None
        required_tokens, panel_sha256 = _evaluation_population(args)
        result = run_successor_production(
            successor_inventory_bundle=inventory,
            required_evaluation_tokens=required_tokens,
            evaluation_panel_sha256=panel_sha256,
            output_dir=args.production_output,
            candidates=args.candidates,
            b5_parent_id=args.b5_parent_id,
            config=production_config,
            b2_checkpoint_path=args.b2_checkpoint,
            b2_intake_bundle_path=args.b2_intake_bundle,
            dinov2_backbone=backbone,
            dinov2_contract=contract,
            descriptor_builder=build_successor_embedding_cache_descriptor,
            real_smoke_fit_limit=args.real_smoke_fit_limit,
        )
        print(
            json.dumps(
                {"status": "FULL128_SUCCESSOR_PRODUCTION_COMPLETE", **result},
                sort_keys=True,
            )
        )
        return 0
    del backbone
    family = build_successor_family_manifest()
    print(
        json.dumps(
            {
                "status": "FULL128_SUCCESSOR_PREREQUISITES_VALID",
                "family_manifest_sha256": content_sha256(family),
                "b2": {
                    "weight_sha256": b2.initialization_sha256,
                    "source_contract_sha256": (
                        b2.initialization_source_contract_sha256
                    ),
                    "intake_receipt_sha256": (b2.initialization_intake_receipt_sha256),
                    "usage_lane": b2.initialization_usage_lane,
                    "learned_checkpoint_reuse": False,
                },
                "dinov2": dinov2_contract_bindings(contract),
            },
            sort_keys=True,
        )
    )
    return 0


def _evaluation_population(args: argparse.Namespace) -> tuple[list[str], str]:
    if args.evaluation_panel is not None:
        panel = validate_fixed_evaluation_panel(
            read_strict_json_document(args.evaluation_panel, **_LARGE_JSON).payload
        )
        return panel["required_sample_tokens"], panel["panel_sha256"]
    value = read_strict_json_document(
        args.required_sample_population, maximum_bytes=268_435_456
    ).payload
    if not isinstance(value, dict) or set(value) != {
        "schema_version",
        "evaluation_panel_sha256",
        "required_sample_tokens",
        "required_sample_tokens_sha256",
    }:
        raise ValueError("required successor sample population fields differ")
    tokens = value["required_sample_tokens"]
    if (
        value["schema_version"] != "cvi.full128_successor_required_population.v1"
        or not isinstance(tokens, list)
        or value["required_sample_tokens_sha256"] != content_sha256(tokens)
    ):
        raise ValueError("required successor sample population binding differs")
    return tokens, value["evaluation_panel_sha256"]


def _reject_sources(values: Sequence[Path | None]) -> None:
    if any(value is not None for value in values):
        raise ValueError("source inputs are valid only with --check-prerequisites")


if __name__ == "__main__":
    raise SystemExit(main())
