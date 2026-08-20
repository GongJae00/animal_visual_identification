"""Operations measurement CLI.

Subcommands: onnx (default), probe, decode, capacity, compare.

Run: ``uv run python -m operations.commands.measure --help``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _run_onnx(argv: list[str]) -> None:
    from operations.measurement.onnx_inference_benchmark import (
        OnnxBenchmarkBackend,
        OnnxInferenceBenchmarkPolicy,
        benchmark_onnx_inference,
    )
    from shared.foundation.protected_io import (
        read_strict_json_object,
        write_private_json_bundle,
    )

    parser = argparse.ArgumentParser(
        description="Run a bounded fresh-process ONNX inference measurement."
    )
    parser.add_argument(
        "--backend",
        required=True,
        choices=[item.value for item in OnnxBenchmarkBackend],
    )
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--backend-config", required=True, type=Path)
    parser.add_argument("--preprocessing", required=True, type=Path)
    parser.add_argument(
        "--artifact",
        required=True,
        action="append",
        type=Path,
        help="Repeat in the exact frozen batch order.",
    )
    parser.add_argument("--dependency-lock", required=True, type=Path)
    parser.add_argument(
        "--runtime-library-policy",
        required=True,
        type=Path,
    )
    parser.add_argument("--code-revision", required=True)
    parser.add_argument("--policy", required=True, type=Path)
    parser.add_argument("--receipt", required=True, type=Path)
    args = parser.parse_args(argv)

    policy = OnnxInferenceBenchmarkPolicy.from_dict(
        read_strict_json_object(args.policy)
    )
    summary = benchmark_onnx_inference(
        backend=OnnxBenchmarkBackend(args.backend),
        model_path=args.model,
        backend_config_path=args.backend_config,
        preprocessing_path=args.preprocessing,
        artifact_paths=tuple(args.artifact),
        dependency_lock_path=args.dependency_lock,
        runtime_library_policy_path=args.runtime_library_policy,
        code_revision=args.code_revision,
        policy=policy,
    )
    receipt = {
        "schema_version": "cvi.onnx_inference_benchmark_receipt.v3",
        "summary_sha256": summary.summary_sha256,
        "summary": summary.to_dict(),
    }
    write_private_json_bundle(((args.receipt, receipt),))


def _run_probe(argv: list[str]) -> None:
    from data.acquisition import (
        ModalityInterval,
        ModalityState,
        RawVideoRecord,
        probe_video_file,
        sha256_file,
    )

    parser = argparse.ArgumentParser(
        description="Create a G0 raw-video record without copying or decoding."
    )
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--cage-id", required=True)
    parser.add_argument("--camera-setting-version", required=True)
    parser.add_argument("--recording-start-ns", required=True, type=int)
    parser.add_argument(
        "--modality",
        choices=[state.value for state in ModalityState],
        default=ModalityState.UNKNOWN.value,
    )
    args = parser.parse_args(argv)

    source = args.source.resolve(strict=True)
    probe = probe_video_file(source)
    duration_ns = round(probe.duration_seconds * 1_000_000_000)
    recording_end_ns = args.recording_start_ns + duration_ns
    record = RawVideoRecord(
        source_id=args.source_id,
        source_uri=str(source),
        source_sha256=sha256_file(source),
        byte_size=source.stat().st_size,
        camera_id=args.camera_id,
        cage_id=args.cage_id,
        camera_setting_version=args.camera_setting_version,
        recording_start_ns=args.recording_start_ns,
        recording_end_ns=recording_end_ns,
        probe=probe,
        modality_intervals=(
            ModalityInterval(
                args.recording_start_ns,
                recording_end_ns,
                ModalityState(args.modality),
            ),
        ),
    )
    print(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))


def _run_decode(argv: list[str]) -> None:
    from operations.measurement.telemetry import monitor_operation
    from operations.video.decode import DecodeBackend, DecodeConfig, benchmark_decode

    parser = argparse.ArgumentParser(
        description="Benchmark a bounded source interval with FFmpeg decode."
    )
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
            "GPU telemetry and is not independently verified by the tool."
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
    args = parser.parse_args(argv)

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
        telemetry_monitor=monitor_operation,
    )
    print(
        json.dumps(
            summary.to_dict(),
            ensure_ascii=False,
            sort_keys=True,
            indent=2,
        )
    )


def _run_capacity(argv: list[str]) -> int:
    from evaluation.splits.duplicate_graph_capacity import (
        analyze_duplicate_graph_capacity,
    )
    from evaluation.splits.protected_public_split import (
        FrozenPublicSplitEvidenceGraph,
        ProtectedPublicSplitPolicy,
        PublicSplitSourceBundle,
    )
    from shared.foundation.protected_io import (
        read_strict_json_object,
        write_private_json_bundle,
    )

    parser = argparse.ArgumentParser(
        description=(
            "Analyze frozen duplicate-component quota capacity without "
            "allocating a split."
        )
    )
    parser.add_argument("--source-bundle", required=True, type=Path)
    parser.add_argument("--evidence-graph", required=True, type=Path)
    parser.add_argument("--split-policy", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)
    report = analyze_duplicate_graph_capacity(
        source=PublicSplitSourceBundle.from_dict(
            read_strict_json_object(args.source_bundle)
        ),
        graph=FrozenPublicSplitEvidenceGraph.from_dict(
            read_strict_json_object(args.evidence_graph)
        ),
        policy=ProtectedPublicSplitPolicy.from_dict(
            read_strict_json_object(args.split_policy)
        ),
    )
    write_private_json_bundle(((args.output, report),))
    print(json.dumps({
        "status": report["status"],
        "largest_allocation_block_identity_count": report[
            "largest_allocation_block_identity_count"
        ],
        "quarantined_identity_count": report["quarantined_identity_count"],
        "failed_quota_lanes": report["failed_quota_lanes"],
        "report_sha256": report["report_sha256"],
        "output": str(args.output),
    }, sort_keys=True))
    return 2 if report["failed_quota_lanes"] else 0


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "onnx"
    if argv and argv[0] in {"onnx", "probe", "decode", "capacity", "compare"}:
        command = argv[0]
        argv = argv[1:]
    if command == "capacity":
        raise SystemExit(_run_capacity(argv))
    if command == "compare":
        from operations.measurement.compare_onnx_measurements import main as run_compare

        previous = sys.argv
        sys.argv = [previous[0], *argv]
        try:
            run_compare()
        finally:
            sys.argv = previous
        return
    {
        "onnx": _run_onnx,
        "probe": _run_probe,
        "decode": _run_decode,
    }[command](argv)


if __name__ == "__main__":
    main()
