"""Convert detector/tracker output to EvidenceObservation JSONL for G1 coverage."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def parse_detection(det: dict[str, Any], frame_timestamp_ns: int, camera_id: str | None = None,
                    session_id: str | None = None, track_id: str | None = None) -> dict[str, Any]:
    """Parse a single detection dict to EvidenceObservation fields."""
    obs = {
        "timestamp_ns": frame_timestamp_ns,
        "modality": det.get("modality", "RGB"),
        "dog_count": 1 if det.get("bbox") or det.get("box") else 0,
    }

    # Bounding box -> dog_crop_height_px
    bbox = det.get("bbox") or det.get("box")
    if bbox:
        if isinstance(bbox, dict):
            obs["dog_crop_height_px"] = int(bbox.get("h", bbox.get("height", 0)))
        elif isinstance(bbox, (list, tuple)) and len(bbox) >= 4:
            obs["dog_crop_height_px"] = int(bbox[3])

    # Quality metrics
    for field in ("head_long_edge_px", "face_min_edge_px", "visible_fraction",
                  "occlusion_fraction", "motion_blur_score", "defocus_blur_score",
                  "cage_bar_occlusion_fraction", "localization_confidence",
                  "ir_saturation_fraction"):
        if field in det:
            val = det[field]
            obs[field] = float(val) if val is not None else None

    if "exposure_ok" in det:
        obs["exposure_ok"] = bool(det["exposure_ok"])

    # Track namespace - only include if all three are available
    has_track = track_id or "track_id" in det
    if has_track and camera_id and session_id:
        obs["camera_id"] = camera_id
        obs["session_id"] = session_id
        obs["track_id"] = track_id or str(det["track_id"])

    return obs


def convert_jsonl_input(input_path: Path, output_path: Path,
                        camera_id: str | None = None,
                        session_id: str | None = None,
                        default_modality: str = "RGB") -> None:
    """Convert JSONL where each line is either:
    - Frame format: {timestamp_ns, detections: [...]}
    - Already-formatted EvidenceObservation: {timestamp_ns, dog_count, ...}
    """
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            line = line.strip()
            if not line:
                continue
            frame = json.loads(line)
            
            # Check if already in EvidenceObservation format (has dog_count)
            if "dog_count" in frame:
                # Already formatted - pass through, optionally add camera/session ONLY if track_id present
                obs = dict(frame)
                has_track = "track_id" in frame and frame["track_id"] is not None
                if camera_id and session_id and has_track:
                    if "camera_id" not in obs:
                        obs["camera_id"] = camera_id
                    if "session_id" not in obs:
                        obs["session_id"] = session_id
                # If no track_id, remove camera_id/session_id to avoid namespace violation
                elif not has_track:
                    obs.pop("camera_id", None)
                    obs.pop("session_id", None)
                    obs.pop("track_id", None)
                fout.write(json.dumps(obs) + "\n")
            else:
                # Frame format with detections
                ts = frame.get("timestamp_ns") or frame.get("timestamp") or frame.get("ts")
                if ts is None:
                    raise ValueError("each frame must have timestamp_ns")
                ts = int(ts)
                detections = frame.get("detections", frame.get("dets", []))
                if not detections:
                    obs = {
                        "timestamp_ns": ts,
                        "modality": frame.get("modality", default_modality),
                        "dog_count": 0,
                    }
                    # Only add namespace if track_id available for no-dog frames
                    if camera_id and session_id and "track_id" in frame:
                        obs["camera_id"] = camera_id
                        obs["session_id"] = session_id
                        obs["track_id"] = str(frame["track_id"])
                    fout.write(json.dumps(obs) + "\n")
                else:
                    for i, det in enumerate(detections):
                        # Only add namespace if this detection has track_id
                        has_track = "track_id" in det
                        obs = parse_detection(det, ts, camera_id, session_id,
                                              det.get("track_id") if has_track else None)
                        fout.write(json.dumps(obs) + "\n")


def convert_mot_txt(input_path: Path, output_path: Path,
                    fps: float, camera_id: str | None = None,
                    session_id: str | None = None) -> None:
    """Convert MOT format: frame, track_id, x, y, w, h, conf, -1, -1, -1."""
    period_ns = int(1_000_000_000 / fps)
    with input_path.open() as fin, output_path.open("w") as fout:
        for line in fin:
            parts = line.strip().split(",")
            if len(parts) < 7:
                continue
            frame_idx = int(parts[0])
            track_id = parts[1]
            x, y, w, h = map(float, parts[2:6])
            conf = float(parts[6])
            ts = frame_idx * period_ns
            obs = {
                "timestamp_ns": ts,
                "modality": "RGB",
                "dog_count": 1,
                "dog_crop_height_px": int(h),
                "localization_confidence": conf,
            }
            if camera_id and session_id:
                obs["camera_id"] = camera_id
                obs["session_id"] = session_id
                obs["track_id"] = track_id
            fout.write(json.dumps(obs) + "\n")


def convert_coco_json(input_path: Path, output_path: Path,
                      camera_id: str | None = None,
                      session_id: str | None = None) -> None:
    """Convert COCO-style JSON with images and annotations."""
    data = json.loads(input_path.read_text())
    images = {img["id"]: img for img in data.get("images", [])}
    with output_path.open("w") as fout:
        for ann in data.get("annotations", []):
            img = images.get(ann["image_id"])
            if not img:
                continue
            ts = img.get("timestamp_ns") or img.get("timestamp") or 0
            ts = int(ts)
            bbox = ann.get("bbox", [])
            obs = {
                "timestamp_ns": ts,
                "modality": img.get("modality", "RGB"),
                "dog_count": 1,
                "dog_crop_height_px": int(bbox[3]) if len(bbox) >= 4 else None,
                "localization_confidence": ann.get("score"),
            }
            if camera_id and session_id:
                obs["camera_id"] = camera_id
                obs["session_id"] = session_id
                if ann.get("track_id") is not None:
                    obs["track_id"] = str(ann["track_id"])
            # Remove None values
            obs = {k: v for k, v in obs.items() if v is not None}
            fout.write(json.dumps(obs) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Convert detector/tracker output to G1 EvidenceObservation JSONL"
    )
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--format", choices=("jsonl", "mot", "coco"), default="jsonl",
                        help="Input format")
    parser.add_argument("--camera-id", default=None)
    parser.add_argument("--session-id", default=None)
    parser.add_argument("--fps", type=float, default=30.0, help="For MOT format")
    parser.add_argument("--default-modality", default="RGB")
    args = parser.parse_args()

    if args.format == "jsonl":
        convert_jsonl_input(args.input, args.output, args.camera_id, args.session_id, args.default_modality)
    elif args.format == "mot":
        convert_mot_txt(args.input, args.output, args.fps, args.camera_id, args.session_id)
    elif args.format == "coco":
        convert_coco_json(args.input, args.output, args.camera_id, args.session_id)


if __name__ == "__main__":
    main()