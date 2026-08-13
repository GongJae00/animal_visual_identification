"""Materialize explicit Face and native Nose outcomes for every SiBeTan crop."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
from pathlib import Path

from PIL import Image

from data.adapters import adapt_sibetan
from experiments.sibetan_evidence import (
    build_evidence_bundle_v2,
    validate_evidence_bundle_v2,
)
from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256
from parsing.nose_region.manifest import encode_png_crop
from parsing.nose_region.native_yt import (
    compute_quality,
    load_localizer_checkpoint,
    nose_geometry,
    predict_localizer,
)
from parsing.prediction_cache import read_prediction_cache
from parsing.roi_manifest import read_roi_manifest

POLICY = {
    "target_association": "EXACTLY_ONE_POSE_DOG_INSTANCE",
    "head_geometry": "ROI_MANIFEST_FACE_CROP_RECT_XYXY",
    "localizer_input": "RAW_SOURCE_RGB_HEAD_RECT",
    "localizer_resize": "BILINEAR_STRETCH_224X224",
    "nose_margin": 0.08,
    "minimum_localizer_confidence": 0.5,
    "frontality_admission": "NONE_CONTINUOUS_QUALITY",
    "native_short_side_admission": "NONE_CONTINUOUS_QUALITY",
    "crop_encoding": "PNG_RGB_LOSSLESS_FROM_DECODED_SOURCE",
}


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _face_unavailable(reasons: list[str]) -> dict[str, object]:
    return {
        "state": "UNAVAILABLE",
        "reasons": sorted(set(reasons)),
        "proposal_box_xyxy": None,
        "source_box_xyxy": None,
        "upstream_quality": None,
        "quality": {},
        "crop_path": None,
        "crop_sha256": None,
        "crop_width": None,
        "crop_height": None,
    }


def _nose_unavailable(reasons: list[str], **diagnostics) -> dict[str, object]:
    return {
        "state": "UNAVAILABLE",
        "reasons": sorted(set(reasons)),
        "head_box_xyxy": diagnostics.get("head_box_xyxy"),
        "head_relative_box_xyxy": diagnostics.get("head_relative_box_xyxy"),
        "source_box_xyxy": diagnostics.get("source_box_xyxy"),
        "keypoints": diagnostics.get("keypoints"),
        "localizer_confidence": diagnostics.get("localizer_confidence"),
        "frontality": diagnostics.get("frontality"),
        "native_short_side": diagnostics.get("native_short_side"),
        "quality": diagnostics.get("quality", {}),
        "crop_path": None,
        "crop_sha256": None,
        "crop_width": None,
        "crop_height": None,
    }


def _face_available(
    path: str, digest: str, size: tuple[int, int], *, proposal, source_box, upstream_quality
) -> dict[str, object]:
    return {
        "state": "AVAILABLE",
        "reasons": [],
        "proposal_box_xyxy": proposal,
        "source_box_xyxy": list(source_box),
        "upstream_quality": upstream_quality,
        "quality": {"native_short_side": min(size)},
        "crop_path": path,
        "crop_sha256": digest,
        "crop_width": size[0],
        "crop_height": size[1],
    }


def _nose_available(path: str, digest: str, size: tuple[int, int], **diagnostics) -> dict[str, object]:
    return {
        "state": "AVAILABLE", "reasons": [],
        "head_box_xyxy": diagnostics["head_box_xyxy"],
        "head_relative_box_xyxy": diagnostics["head_relative_box_xyxy"],
        "source_box_xyxy": diagnostics["source_box_xyxy"],
        "keypoints": diagnostics["keypoints"],
        "localizer_confidence": diagnostics["localizer_confidence"],
        "frontality": diagnostics["frontality"],
        "native_short_side": diagnostics["native_short_side"],
        "quality": diagnostics["quality"],
        "crop_path": path, "crop_sha256": digest,
        "crop_width": size[0], "crop_height": size[1],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--prediction-cache", type=Path, required=True)
    parser.add_argument("--face-manifest", type=Path, required=True)
    parser.add_argument("--localizer-checkpoint", type=Path, required=True)
    parser.add_argument("--localizer-checkpoint-sha256", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.output_dir.exists() or args.output_dir.is_symlink():
        raise FileExistsError(f"refusing to overwrite SiBeTan evidence: {args.output_dir}")
    if _sha(args.localizer_checkpoint) != args.localizer_checkpoint_sha256:
        raise ValueError("Nose localizer checkpoint SHA-256 differs")

    samples = adapt_sibetan(args.data_root)
    cache = read_prediction_cache(args.prediction_cache)
    if cache["dataset_name"] != "sibetan":
        raise ValueError("prediction cache dataset differs")
    pose_by_sample = {row["sample_id"]: row for row in cache["records"]}
    if set(pose_by_sample) != {sample.sample_id for sample in samples}:
        raise ValueError("prediction cache does not exactly cover SiBeTan")
    face_manifest = read_roi_manifest(args.face_manifest)
    face_by_sample: dict[str, list[dict]] = {}
    for row in face_manifest["records"]:
        if row["registered_identity_id"] is not None and row["face_crop_path"] is not None:
            face_by_sample.setdefault(row["sample_id"], []).append(row)

    checkpoint_bytes = args.localizer_checkpoint.read_bytes()
    model, device, localizer_bindings = load_localizer_checkpoint(
        checkpoint_bytes, args.device
    )
    parent = args.output_dir.parent.resolve(strict=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{args.output_dir.name}.staging-", dir=parent))
    try:
        (staging / "face_crops").mkdir()
        (staging / "nose_crops").mkdir()
        rows = []
        for sample in sorted(samples, key=lambda item: item.sample_id):
            pose = pose_by_sample[sample.sample_id]
            if (
                pose["image_path"] != sample.image_path
                or pose["image_sha256"] != sample.image_sha256
                or pose["width"] != sample.width
                or pose["height"] != sample.height
            ):
                raise ValueError("pose cache and SiBeTan source differ")
            detections = len(pose["dog_boxes"])
            association_reasons = (
                ["NO_TARGET_DOG_INSTANCE"]
                if detections == 0
                else (["MULTIPLE_DOG_INSTANCES"] if detections != 1 else [])
            )
            face_candidates = face_by_sample.get(sample.sample_id, [])
            image_path = args.data_root / sample.image_path
            if _sha(image_path) != sample.image_sha256:
                raise ValueError("SiBeTan source image SHA-256 differs")
            with Image.open(image_path) as opened:
                image = opened.convert("RGB")
                image.load()
            face_rect = None
            if association_reasons:
                face = _face_unavailable(association_reasons)
            elif len(face_candidates) != 1:
                face = _face_unavailable(["FACE_GEOMETRY_UNAVAILABLE"])
            else:
                face_row = face_candidates[0]
                face_rect = tuple(face_row["face_crop_rect_xyxy"])
                face_payload, face_size = encode_png_crop(image, face_rect)
                face_relative = f"face_crops/{sample.sample_id}.png"
                (staging / face_relative).write_bytes(face_payload)
                face = _face_available(
                    face_relative, hashlib.sha256(face_payload).hexdigest(), face_size,
                    proposal=face_row["face_roi_xyxy"], source_box=face_rect,
                    upstream_quality=face_row["face_quality"],
                )
            if face_rect is None:
                nose = _nose_unavailable(face["reasons"])
            else:
                head = image.crop(face_rect)
                prediction = predict_localizer(model, device, head)
                box, confidence, frontality = nose_geometry(
                    prediction, head.width, head.height, margin=POLICY["nose_margin"]
                )
                source_box = [
                    face_rect[0] + box[0], face_rect[1] + box[1],
                    face_rect[0] + box[2], face_rect[1] + box[3],
                ]
                short_side = min(source_box[2] - source_box[0], source_box[3] - source_box[1])
                keypoints = {
                    "normalized_head": prediction,
                    "source_xyc": [
                        [face_rect[0] + point[0] * head.width,
                         face_rect[1] + point[1] * head.height, point[2]]
                        for point in prediction
                    ],
                }
                crop = image.crop(tuple(source_box))
                quality = compute_quality(
                    crop, None, native_short_side=short_side,
                    detector_confidence=confidence, frontality=frontality,
                )
                diagnostics = {
                    "head_box_xyxy": list(face_rect),
                    "head_relative_box_xyxy": list(box),
                    "source_box_xyxy": source_box,
                    "keypoints": keypoints,
                    "localizer_confidence": confidence,
                    "frontality": frontality,
                    "native_short_side": short_side,
                    "quality": quality,
                }
                if confidence < POLICY["minimum_localizer_confidence"]:
                    nose = _nose_unavailable(["LOW_LOCALIZATION_CONFIDENCE"], **diagnostics)
                else:
                    payload, size = encode_png_crop(image, source_box)
                    relative = f"nose_crops/{sample.sample_id}.png"
                    (staging / relative).write_bytes(payload)
                    nose = _nose_available(
                        relative, hashlib.sha256(payload).hexdigest(), size, **diagnostics
                    )
            rows.append(
                {
                    "sample_id": sample.sample_id,
                    "registered_identity_id": sample.registered_identity_id,
                    "source_group_id": sample.source_group_id,
                    "image_path": sample.image_path,
                    "image_sha256": sample.image_sha256,
                    "source_width": sample.width,
                    "source_height": sample.height,
                    "face": face,
                    "nose": nose,
                }
            )
        bundle = build_evidence_bundle_v2(
            records=rows,
            input_bindings={
                "prediction_cache_path": os.fspath(args.prediction_cache),
                "prediction_cache_sha256": _sha(args.prediction_cache),
                "prediction_cache_content_sha256": content_sha256(cache),
                "face_manifest_path": os.fspath(args.face_manifest),
                "face_manifest_sha256": _sha(args.face_manifest),
                "face_manifest_content_sha256": content_sha256(face_manifest),
                "localizer_checkpoint_path": os.fspath(args.localizer_checkpoint),
                "localizer_checkpoint_sha256": args.localizer_checkpoint_sha256,
                "localizer_bindings_sha256": localizer_bindings["content_sha256"],
                "code_sha256s": {
                    relative: _sha(Path(__file__).resolve().parents[1] / relative)
                    for relative in (
                        "experiments/sibetan_evidence.py",
                        "parsing/nose_region/native_yt.py",
                        "workflows/materialize_sibetan_multievidence.py",
                    )
                },
            },
            policy=POLICY,
        )
        write_private_json_bundle(((staging / "evidence.json", bundle),))
        validate_evidence_bundle_v2(bundle, root=staging)
        os.rename(staging, args.output_dir)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    print(
        json.dumps(
            {
                "status": "CREATED_SIBETAN_MULTIEVIDENCE",
                "output": os.fspath(args.output_dir / "evidence.json"),
                "manifest_sha256": bundle["manifest_sha256"],
                "state_counts": bundle["manifest"]["state_counts"],
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
