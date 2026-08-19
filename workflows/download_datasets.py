"""Show dataset acquisition status and manual source tips.

No dataset has an admitted automatic download. Every named selector is
disabled and fails with manual-acquisition guidance before network or model
framework imports. The default ``all`` selection is an intentional no-op.

Usage:
    uv run python workflows/download_datasets.py
    uv run python workflows/download_datasets.py --dataset dogfacenet
    uv run python workflows/download_datasets.py --dataset yt-bb-dog
    uv run python workflows/download_datasets.py --list
    uv run python workflows/download_datasets.py create-manifest ...
    uv run python workflows/download_datasets.py check MANIFEST
    uv run python workflows/download_datasets.py camera-spec ...

Data root resolution:
    1. ``--data-root``
    2. ``CANINE_IDENTITY_DATA_DIR`` when set before startup
    3. ``~/canine_identity_data``
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from contracts.model_paths import DATA_DIR, SUPPORTED_DATASETS
from data.acquisition import (
    AcquisitionManifest,
    CameraSpecification,
    IRMechanism,
    RawVideoRecord,
)


class ManualAcquisitionRequired(RuntimeError):
    """Raised when a disabled selector requires manual acquisition."""


_DATASETS: dict[str, dict[str, str]] = {
    "dogfacenet": {
        "mode": "disabled/manual",
        "description": "DogFaceNet from Hugging Face",
        "source": "https://huggingface.co/datasets/dimidagd/DogFaceNet_224resize",
        "reason": (
            "the source tip is not pinned to a revision, content hash, and "
            "license receipt"
        ),
    },
    "ap10k-dog": {
        "mode": "disabled/manual",
        "description": "AP-10K domestic dog subset",
        "source": "https://github.com/AlexTheBad/AP-10K",
        "reason": "no automatic acquisition contract is admitted",
    },
    "dogflw": {
        "mode": "disabled/manual",
        "description": "Dog Facial Landmarks in the Wild",
        "source": "https://www.kaggle.com/datasets/georgemartvel/dogflw",
        "reason": "no automatic acquisition contract is admitted",
    },
    "yt-bb-dog": {
        "mode": "disabled/manual",
        "description": "YT-BB-Dog",
        "source": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "reason": "no automatic acquisition contract is admitted",
    },
    "sibetan": {
        "mode": "disabled/manual",
        "description": "SiBeTan",
        "source": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "reason": "no automatic acquisition contract is admitted",
    },
    "mpdd": {
        "mode": "disabled/manual",
        "description": "Multi-pose dog dataset (MPDD)",
        "source": "https://data.mendeley.com/datasets/v5j6m8dzhv/1",
        "reason": "no automatic acquisition contract is admitted",
    },
}


def _resolve_data_root(cli_root: str | None) -> Path:
    return Path(cli_root).expanduser() if cli_root else DATA_DIR


def _image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for entry in path.rglob("*")
    )


def _dataset_destination(name: str, data_root: Path) -> Path:
    try:
        directory = SUPPORTED_DATASETS[name]["dir"]
    except KeyError as exc:
        raise ValueError(f"Dataset metadata is missing for {name!r}") from exc
    return data_root / "datasets" / directory


def _list_datasets(data_root: Path) -> None:
    for name, dataset in _DATASETS.items():
        destination = _dataset_destination(name, data_root)
        count = _image_count(destination)
        local_status = f"{count} local images" if count else "not present locally"
        operation = (
            f"{dataset['mode']} acquisition only ({dataset['reason']}): "
            f"{dataset['source']}"
        )
        print(f"{name}: {operation}; {local_status}")


def download_dataset(name: str, data_root: Path) -> None:
    try:
        dataset = _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}") from exc

    raise ManualAcquisitionRequired(
        f"{dataset['description']} automatic acquisition is disabled because "
        f"{dataset['reason']}. Review the source and applicable terms at "
        f"{dataset['source']}, then place authorized material under "
        f"{_dataset_destination(name, data_root)}. No network request or directory "
        "creation was attempted."
    )


def _run_download(argv: list[str]) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=[*_DATASETS, "all"])
    parser.add_argument(
        "--data-root",
        default=None,
        help="Data root directory (default: CANINE_IDENTITY_DATA_DIR or ~/canine_identity_data)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show supported operations and local status",
    )
    args = parser.parse_args(argv)

    data_root = _resolve_data_root(args.data_root)
    if args.list:
        _list_datasets(data_root)
        return

    if args.dataset == "all":
        print(
            "Intentional no-op: no dataset has an admitted automatic download; "
            "no network request or filesystem change was attempted."
        )
        return

    try:
        download_dataset(args.dataset, data_root)
    except ManualAcquisitionRequired as exc:
        parser.exit(2, f"error: {exc}\n")


def _run_create_manifest(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-spec", required=True, type=Path, action="append")
    parser.add_argument("--raw-video-record", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    cameras = []
    for spec_path in args.camera_spec:
        payload = json.loads(spec_path.resolve(strict=True).read_text(encoding="utf-8"))
        cameras.append(CameraSpecification.from_dict(payload))

    videos = []
    for record_path in args.raw_video_record:
        payload = json.loads(record_path.resolve(strict=True).read_text(encoding="utf-8"))
        videos.append(RawVideoRecord.from_dict(payload))

    manifest = AcquisitionManifest(
        cameras=tuple(cameras),
        videos=tuple(videos),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def _run_check(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("manifest", type=Path)
    args = parser.parse_args(argv)

    payload = json.loads(
        args.manifest.resolve(strict=True).read_text(encoding="utf-8")
    )
    manifest = AcquisitionManifest.from_dict(payload)
    blockers = manifest.gate_blockers()
    result = {
        "schema_version": manifest.schema_version,
        "manifest_sha256": manifest.manifest_sha256,
        "camera_versions": len(manifest.cameras),
        "source_videos": len(manifest.videos),
        "gate_status": "PASS" if not blockers else "BLOCKED",
        "blockers": list(blockers),
    }
    print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))
    if blockers:
        raise SystemExit(2)


def _run_camera_spec(argv: list[str]) -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--camera-setting-version", required=True)
    parser.add_argument("--sensor-model", required=False)
    parser.add_argument(
        "--ir-mechanism",
        choices=[m.value for m in IRMechanism],
        default=IRMechanism.UNKNOWN.value,
    )
    parser.add_argument("--ir-spectral-band", required=False)
    parser.add_argument("--width", required=False, type=int)
    parser.add_argument("--height", required=False, type=int)
    parser.add_argument("--stored-fps", required=False, type=float)
    parser.add_argument("--shutter", required=False)
    parser.add_argument("--gain-mode", required=False)
    parser.add_argument("--exposure-mode", required=False)
    parser.add_argument("--white-balance-mode", required=False)
    parser.add_argument("--wdr-enabled", required=False, type=lambda x: x.lower() == "true")
    parser.add_argument("--ir-cut-behavior", required=False)
    parser.add_argument("--codec", required=False)
    parser.add_argument("--target-bitrate-mbps", required=False, type=float)
    parser.add_argument("--gop-length", required=False, type=int)
    parser.add_argument("--focus-mode", required=False)
    parser.add_argument("--focal-length-mm", required=False, type=float)
    parser.add_argument("--horizontal-fov-deg", required=False, type=float)
    parser.add_argument("--installation-height-m", required=False, type=float)
    parser.add_argument("--cage-center-distance-m", required=False, type=float)
    parser.add_argument("--pan-deg", required=False, type=float)
    parser.add_argument("--tilt-deg", required=False, type=float)
    parser.add_argument("--timestamp-accuracy-ms", required=False, type=float)
    parser.add_argument("--measured-frame-drop-rate", required=False, type=float)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args(argv)

    spec = CameraSpecification(
        camera_id=args.camera_id,
        camera_setting_version=args.camera_setting_version,
        sensor_model=args.sensor_model,
        ir_mechanism=IRMechanism(args.ir_mechanism),
        ir_spectral_band=args.ir_spectral_band,
        width=args.width,
        height=args.height,
        stored_fps=args.stored_fps,
        shutter=args.shutter,
        gain_mode=args.gain_mode,
        exposure_mode=args.exposure_mode,
        white_balance_mode=args.white_balance_mode,
        wdr_enabled=args.wdr_enabled,
        ir_cut_behavior=args.ir_cut_behavior,
        codec=args.codec,
        target_bitrate_mbps=args.target_bitrate_mbps,
        gop_length=args.gop_length,
        focus_mode=args.focus_mode,
        focal_length_mm=args.focal_length_mm,
        horizontal_fov_deg=args.horizontal_fov_deg,
        installation_height_m=args.installation_height_m,
        cage_center_distance_m=args.cage_center_distance_m,
        pan_deg=args.pan_deg,
        tilt_deg=args.tilt_deg,
        timestamp_accuracy_ms=args.timestamp_accuracy_ms,
        measured_frame_drop_rate=args.measured_frame_drop_rate,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(spec.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


def main(argv: list[str] | None = None) -> None:
    argv = list(sys.argv[1:] if argv is None else argv)
    command = "download"
    if argv and argv[0] in {"download", "create-manifest", "check", "camera-spec"}:
        command = argv[0]
        argv = argv[1:]
    {
        "download": _run_download,
        "create-manifest": _run_create_manifest,
        "check": _run_check,
        "camera-spec": _run_camera_spec,
    }[command](argv)


if __name__ == "__main__":
    main()
