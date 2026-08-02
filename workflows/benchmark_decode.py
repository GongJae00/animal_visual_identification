"""Benchmark a bounded source interval with software or CUDA FFmpeg decode."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.decode import DecodeBackend, DecodeConfig, benchmark_decode


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--source-sha256", required=True)
    parser.add_argument(
        "--backend",
        choices=[backend.value for backend in DecodeBackend],
        default=DecodeBackend.CPU.value,
    )
    parser.add_argument("--duration-seconds", required=True, type=float)
    parser.add_argument("--threads", type=int)
    parser.add_argument("--gpu-device-index", type=int)
    parser.add_argument("--gpu-telemetry-interval-seconds", type=float)
    contamination = parser.add_mutually_exclusive_group()
    contamination.add_argument(
        "--attest-no-unrelated-gpu-work",
        dest="unrelated_gpu_work_excluded",
        action="store_true",
        help=(
            "Operator attestation only; one declaration is required with "
            "GPU telemetry and is not "
            "independently verified by the tool."
        ),
    )
    contamination.add_argument(
        "--declare-unrelated-gpu-work",
        dest="unrelated_gpu_work_excluded",
        action="store_false",
        help="Mark the device-wide telemetry receipt as contaminated.",
    )
    parser.set_defaults(unrelated_gpu_work_excluded=None)
    parser.add_argument("--warmup", type=int, default=0)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float)
    args = parser.parse_args()

    summary = benchmark_decode(
        args.source.resolve(strict=True),
        source_id=args.source_id,
        source_sha256=args.source_sha256,
        config=DecodeConfig(
            backend=DecodeBackend(args.backend),
            duration_seconds=args.duration_seconds,
            threads=args.threads,
            gpu_device_index=args.gpu_device_index,
            gpu_telemetry_interval_seconds=(
                args.gpu_telemetry_interval_seconds
            ),
        ),
        warmup_runs=args.warmup,
        repeat_runs=args.repeats,
        timeout_seconds=args.timeout_seconds,
        unrelated_gpu_work_excluded_by_operator=(
            args.unrelated_gpu_work_excluded
        ),
    )
    print(
        json.dumps(
            summary.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
