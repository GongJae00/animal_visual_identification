"""Materialize or assemble a content-bound Full128 route-plan bundle."""

from __future__ import annotations

from legacy.version.root import repository_root as find_repo_root
import argparse
import json
import sys
from collections.abc import Sequence
from pathlib import Path

from foundation.protected_io import write_private_json_bundle
from embedding.methods.full_segment.preparation.materialization import (
    assemble_full128_materialization,
    build_bound_parser_runtime,
    materialize_full128_route_plan,
    read_route_plan_bundle,
)
from data.full_segment.route_plan import (
    validate_full128_route_plan_bundle,
)


def _add_plan_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument(
        "--fast-verify-on-read",
        action="store_true",
        help=(
            "skip upfront source/evidence verification; materialization and assembly "
            "bind artifacts during deterministic per-job reads"
        ),
    )


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)

    validate = commands.add_parser("validate", help="validate the route-plan bundle")
    _add_plan_arguments(validate)

    materialize = commands.add_parser(
        "materialize", help="materialize one deterministic resumable shard"
    )
    _add_plan_arguments(materialize)
    materialize.add_argument("--output-root", required=True, type=Path)
    materialize.add_argument("--parsing-runtime-manifest", required=True, type=Path)
    materialize.add_argument("--foreground-model-dir", required=True, type=Path)
    materialize.add_argument("--foreground-model-manifest", required=True, type=Path)
    materialize.add_argument("--instance-model-dir", required=True, type=Path)
    materialize.add_argument("--instance-model-manifest", required=True, type=Path)
    materialize.add_argument("--device", required=True, choices=("cpu", "cuda"))
    materialize.add_argument("--shard-count", type=int, default=1)
    materialize.add_argument("--shard-index", type=int, default=0)
    materialize.add_argument("--maximum-jobs", type=int)

    assemble = commands.add_parser(
        "assemble", help="require complete coverage and build split/inventory bundles"
    )
    _add_plan_arguments(assemble)
    assemble.add_argument("--output-root", required=True, type=Path)
    assemble.add_argument("--output", required=True, type=Path)
    assemble.add_argument("--allocation-name", default="full128-route-plan-v1")
    assemble.add_argument(
        "--progress-every-jobs",
        type=_positive_int,
        help="write deterministic assembly progress to stderr at this job interval",
    )
    return parser.parse_args(argv)


def _positive_int(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be positive")
    return parsed


def _external_new_output(path: Path) -> Path:
    repository = find_repo_root(__file__)
    requested = path.absolute()
    parent = requested.parent.resolve(strict=True)
    output = parent / requested.name
    if output == repository or output.is_relative_to(repository):
        raise ValueError("Full128 assembly output must remain outside the repository")
    if requested.is_symlink() or output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite Full128 assembly: {output}")
    return output


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    bundle = read_route_plan_bundle(args.route_plan.absolute())
    if args.command != "assemble":
        validate_full128_route_plan_bundle(
            bundle,
            verify_files=not args.fast_verify_on_read,
        )
    if args.command == "validate":
        print(
            json.dumps(
                {
                    "status": "VALID_FULL128_ROUTE_PLAN",
                    "plan_sha256": bundle["plan_sha256"],
                    "record_count": len(bundle["plan"]["records"]),
                    "verified_plan_files_upfront": not args.fast_verify_on_read,
                },
                sort_keys=True,
            )
        )
        return 0
    if args.command == "materialize":
        runtime = build_bound_parser_runtime(
            route_plan_bundle=bundle,
            parsing_runtime_manifest=args.parsing_runtime_manifest.absolute(),
            foreground_model_dir=args.foreground_model_dir.absolute(),
            foreground_model_manifest=args.foreground_model_manifest.absolute(),
            instance_model_dir=args.instance_model_dir.absolute(),
            instance_model_manifest=args.instance_model_manifest.absolute(),
            device=args.device,
        )
        summary = materialize_full128_route_plan(
            bundle,
            output_root=args.output_root.absolute(),
            parser_runtime=runtime,
            verify_plan_files_upfront=False,
            shard_count=args.shard_count,
            shard_index=args.shard_index,
            maximum_jobs=args.maximum_jobs,
        )
        print(
            json.dumps(
                {"status": "FULL128_MATERIALIZATION_COMPLETE", **summary},
                sort_keys=True,
            )
        )
        return 0
    output = _external_new_output(args.output)
    progress = None
    if args.progress_every_jobs is not None:
        interval = args.progress_every_jobs

        def report_progress(completed: int, total: int) -> None:
            if completed % interval == 0 or completed == total:
                print(
                    json.dumps(
                        {
                            "status": "FULL128_ASSEMBLY_PROGRESS",
                            "completed_jobs": completed,
                            "total_jobs": total,
                        },
                        sort_keys=True,
                    ),
                    file=sys.stderr,
                )

        progress = report_progress
    assembly = assemble_full128_materialization(
        bundle,
        output_root=args.output_root.absolute(),
        allocation_name=args.allocation_name,
        verify_plan_files_upfront=False,
        progress=progress,
    )
    write_private_json_bundle(((output, assembly),))
    print(
        json.dumps(
            {
                "status": "CREATED_FULL128_MATERIALIZATION_ASSEMBLY",
                "assembly_sha256": assembly["assembly_sha256"],
                "sample_count": assembly["sample_count"],
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
