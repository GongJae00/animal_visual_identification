"""Create a G0 camera specification JSON without assuming any source video."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.acquisition import CameraSpecification, IRMechanism


def main() -> None:
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
    args = parser.parse_args()

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


if __name__ == "__main__":
    main()