"""Exact, optional SAM2.1 teacher-mask production for native YT nose records.

The selection and manifest code in this module has no dependency on SAM2 or
PyTorch.  A SAM2 checkout is imported only by :func:`load_local_sam2` after its
caller-supplied files and revision have been validated.
"""

from __future__ import annotations

import hashlib
import io
import math
import re
import subprocess
import sys
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

import cv2
import numpy as np
from PIL import Image

from parsing.export.regions.localizer import KEYPOINT_ORDER, NOSE_POINT_INDICES
from parsing.export.regions.native_yt import (
    TEACHER_SCHEMA,
    decode_source_image,
    validate_manifest_bundle,
)
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import read_retained_regular_file


SOURCE_IMAGE_MANIFEST_SCHEMA = "archive.nose.yt_native_nose_teacher_source_images.v1"
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_GIT_COMMIT = re.compile(r"[0-9a-f]{40}\Z")
_SOURCE_FIELDS = {
    "sample_token",
    "sequence_token",
    "track_token",
    "frame_index",
    "source_image_path",
    "source_sha256",
    "source_width",
    "source_height",
    "nose_box_xyxy",
    "keypoints",
}
_KEYPOINT_FIELDS = {
    "name",
    "normalized_x",
    "normalized_y",
    "source_x",
    "source_y",
    "confidence",
}
_OUTPUT_RECORD_FIELDS = {
    "sample_token",
    "sequence_token",
    "track_token",
    "frame_index",
    "source_sha256",
    "source_width",
    "source_height",
    "coordinate_space",
    "nose_box_xyxy",
    "positive_keypoints_xy",
    "status",
    "rejection_reasons",
    "mask_path",
    "mask_sha256",
    "selection",
}


@dataclass(frozen=True, slots=True)
class TeacherSource:
    sample_token: str
    sequence_token: str
    track_token: str
    frame_index: int
    source_sha256: str
    source_width: int
    source_height: int
    nose_box_xyxy: tuple[int, int, int, int]
    positive_keypoints_xy: tuple[tuple[float, float], ...]
    source_bytes: bytes


@dataclass(frozen=True, slots=True)
class MaskSelectionPolicy:
    model_score_weight: float = 0.50
    anatomical_overlap_weight: float = 0.35
    compactness_weight: float = 0.15
    minimum_model_score: float = 0.50
    minimum_anatomical_overlap: float = 0.45
    minimum_compactness: float = 0.05
    minimum_area_to_box_ratio: float = 0.05
    maximum_area_to_box_ratio: float = 1.50
    ambiguity_margin: float = 0.05

    def __post_init__(self) -> None:
        values = self.to_dict()
        for name, value in values.items():
            if not isinstance(value, float) or not math.isfinite(value):
                raise ValueError(f"selection policy {name} must be finite float")
        weights = (
            self.model_score_weight,
            self.anatomical_overlap_weight,
            self.compactness_weight,
        )
        if any(value < 0.0 for value in weights) or not math.isclose(
            sum(weights), 1.0, abs_tol=1e-12
        ):
            raise ValueError("selection policy weights must be nonnegative and sum to one")
        for name in (
            "minimum_model_score",
            "minimum_anatomical_overlap",
            "minimum_compactness",
            "ambiguity_margin",
        ):
            if not 0.0 <= getattr(self, name) <= 1.0:
                raise ValueError(f"selection policy {name} must be in [0,1]")
        if not 0.0 < self.minimum_area_to_box_ratio < self.maximum_area_to_box_ratio:
            raise ValueError("selection policy area-to-box limits differ")

    def to_dict(self) -> dict[str, float]:
        return {
            name: getattr(self, name)
            for name in self.__dataclass_fields__
        }


def validate_source_image_manifest(
    payload: object,
    *,
    root: Path,
    source_receipt_file_sha256: str,
) -> tuple[TeacherSource, ...]:
    """Validate and retain exact source images plus source-space nose prompts."""

    _require_sha256(source_receipt_file_sha256, "source receipt file SHA-256")
    expected_fields = {"schema_version", "source_receipt_file_sha256", "records"}
    if (
        not isinstance(payload, dict)
        or set(payload) != expected_fields
        or payload["schema_version"] != SOURCE_IMAGE_MANIFEST_SCHEMA
        or payload["source_receipt_file_sha256"] != source_receipt_file_sha256
        or not isinstance(payload["records"], list)
        or not payload["records"]
    ):
        raise ValueError("teacher source image manifest schema or receipt differs")
    resolved_root = root.resolve(strict=True)
    if not resolved_root.is_dir():
        raise ValueError("teacher source image root must be a directory")
    records = payload["records"]
    if records != sorted(records, key=lambda row: row.get("sample_token", "")):
        raise ValueError("teacher source records must be sorted by sample token")
    seen: set[str] = set()
    track_frames: set[tuple[str, int]] = set()
    result: list[TeacherSource] = []
    for row in records:
        source = _validate_source_row(row, resolved_root)
        if source.sample_token in seen:
            raise ValueError("teacher source manifest repeats a sample token")
        frame_key = (source.track_token, source.frame_index)
        if frame_key in track_frames:
            raise ValueError("teacher source manifest repeats a track frame")
        seen.add(source.sample_token)
        track_frames.add(frame_key)
        result.append(source)
    return tuple(result)


def sources_from_native_manifest(
    bundle: object,
    *,
    source_bytes_by_token: Mapping[str, bytes],
) -> tuple[TeacherSource, ...]:
    """Adapt a validated native YT manifest and exact caller-loaded source bytes."""

    manifest = validate_manifest_bundle(bundle)
    localized = [row for row in manifest["records"] if row["record_state"] != "NO_ROI"]
    expected_tokens = {row["sample_token"] for row in localized}
    if set(source_bytes_by_token) != expected_tokens:
        raise ValueError("native teacher source byte coverage differs")
    result: list[TeacherSource] = []
    for row in localized:
        source_bytes = source_bytes_by_token[row["sample_token"]]
        if not isinstance(source_bytes, bytes):
            raise TypeError("native teacher source bytes must be bytes")
        if hashlib.sha256(source_bytes).hexdigest() != row["source_sha256"]:
            raise ValueError("native teacher source image SHA-256 differs")
        image = decode_source_image(source_bytes)
        if image.size != (row["source_width"], row["source_height"]):
            raise ValueError("native teacher source dimensions differ")
        positive = tuple(
            (float(point["source_x"]), float(point["source_y"]))
            for index, point in enumerate(row["keypoints"])
            if index in NOSE_POINT_INDICES and point["confidence"] > 0.0
        )
        if not positive:
            raise ValueError("native teacher source has no positive nose keypoints")
        result.append(
            TeacherSource(
                sample_token=row["sample_token"],
                sequence_token=row["sequence_token"],
                track_token=row["track_token"],
                frame_index=row["frame_index"],
                source_sha256=row["source_sha256"],
                source_width=row["source_width"],
                source_height=row["source_height"],
                nose_box_xyxy=tuple(row["nose_box_xyxy"]),
                positive_keypoints_xy=positive,
                source_bytes=source_bytes,
            )
        )
    if not result:
        raise ValueError("native YT manifest contains no localized teacher sources")
    return tuple(result)


def produce_teacher_manifest(
    sources: Sequence[TeacherSource],
    predictor: Any,
    *,
    source_binding: Mapping[str, Any],
    producer: Mapping[str, Any],
    policy: MaskSelectionPolicy = MaskSelectionPolicy(),
    propagate_tracks: bool = False,
) -> tuple[dict[str, Any], dict[str, bytes]]:
    """Run exact prompts, reject ambiguity, and return a bound manifest/artifacts."""

    if not sources:
        raise ValueError("teacher production requires at least one source")
    if list(sources) != sorted(sources, key=lambda item: item.sample_token):
        raise ValueError("teacher sources must be uniquely sorted by sample token")
    if len({item.sample_token for item in sources}) != len(sources):
        raise ValueError("teacher sources must be uniquely sorted by sample token")
    source_payload = dict(source_binding)
    producer_payload = dict(producer)
    if not source_payload or not producer_payload:
        raise ValueError("teacher source and producer bindings must be non-empty")

    images: dict[str, np.ndarray] = {}
    candidate_sets: dict[str, list[tuple[np.ndarray, float, str, str]]] = {}
    for source in sources:
        image = decode_source_image(source.source_bytes)
        if image.size != (source.source_width, source.source_height):
            raise ValueError("teacher source dimensions differ from manifest")
        if hashlib.sha256(source.source_bytes).hexdigest() != source.source_sha256:
            raise ValueError("teacher source image SHA-256 differs from manifest")
        rgb = np.asarray(image, dtype=np.uint8)
        images[source.sample_token] = rgb
        candidate_sets[source.sample_token] = _predict_image_candidates(
            predictor, rgb, source
        )

    direct = {
        source.sample_token: _select_mask(
            source, candidate_sets[source.sample_token], policy
        )
        for source in sources
    }
    propagation_method = getattr(predictor, "propagate_track", None)
    video_available = callable(propagation_method) and bool(
        getattr(predictor, "video_api_available", True)
    )
    propagation = {
        "requested": propagate_tracks,
        "api_available": video_available,
        "frame_runs_attempted": 0,
        "frame_runs_propagated": 0,
        "score_semantics": "VIDEO_LOGIT_FOREGROUND_MEAN",
    }
    if propagate_tracks and propagation["api_available"]:
        by_track: dict[str, list[TeacherSource]] = {}
        for source in sources:
            by_track.setdefault(source.track_token, []).append(source)
        for ordered_track in by_track.values():
            ordered_track.sort(key=lambda item: item.frame_index)
            runs: list[list[TeacherSource]] = []
            for source in ordered_track:
                size = (source.source_width, source.source_height)
                if not runs or size != (
                    runs[-1][-1].source_width,
                    runs[-1][-1].source_height,
                ):
                    runs.append([])
                runs[-1].append(source)
            for track in (run for run in runs if len(run) >= 2):
                accepted = [
                    (index, direct[item.sample_token])
                    for index, item in enumerate(track)
                    if direct[item.sample_token]["mask"] is not None
                ]
                if not accepted:
                    continue
                propagation["frame_runs_attempted"] += 1
                seed_index, seed_result = max(
                    accepted, key=lambda item: item[1]["selected_score"]
                )
                seed = track[seed_index]
                propagated = predictor.propagate_track(
                    images=[images[item.sample_token] for item in track],
                    seed_index=seed_index,
                    box=np.asarray(seed.nose_box_xyxy, dtype=np.float32),
                    point_coords=np.asarray(seed.positive_keypoints_xy, dtype=np.float32),
                    point_labels=np.ones(len(seed.positive_keypoints_xy), dtype=np.int32),
                    seed_mask=seed_result["mask"].copy(),
                )
                parsed = _parse_propagated_candidates(propagated, track)
                for index, candidates in parsed.items():
                    candidate_sets[track[index].sample_token].extend(candidates)
                propagation["frame_runs_propagated"] += 1

    records: list[dict[str, Any]] = []
    artifacts: dict[str, bytes] = {}
    for source in sources:
        selected = _select_mask(source, candidate_sets[source.sample_token], policy)
        mask = selected.pop("mask")
        selected.pop("selected_score")
        relative = None
        digest = None
        if mask is not None:
            relative = f"masks/{source.sample_token}.png"
            payload = _mask_png(mask)
            artifacts[relative] = payload
            digest = hashlib.sha256(payload).hexdigest()
        records.append(
            {
                "sample_token": source.sample_token,
                "sequence_token": source.sequence_token,
                "track_token": source.track_token,
                "frame_index": source.frame_index,
                "source_sha256": source.source_sha256,
                "source_width": source.source_width,
                "source_height": source.source_height,
                "coordinate_space": "SOURCE_IMAGE_PIXELS",
                "nose_box_xyxy": list(source.nose_box_xyxy),
                "positive_keypoints_xy": [list(point) for point in source.positive_keypoints_xy],
                "status": "ACCEPTED" if mask is not None else "REJECTED",
                "rejection_reasons": selected.pop("rejection_reasons"),
                "mask_path": relative,
                "mask_sha256": digest,
                "selection": selected,
            }
        )
    body = {
        "schema_version": TEACHER_SCHEMA,
        "source_binding": source_payload,
        "source_binding_sha256": content_sha256(source_payload),
        "producer": producer_payload,
        "producer_sha256": content_sha256(producer_payload),
        "selection_policy": policy.to_dict(),
        "propagation": propagation,
        "records": records,
        "record_counts": {
            "ACCEPTED": sum(row["status"] == "ACCEPTED" for row in records),
            "REJECTED": sum(row["status"] == "REJECTED" for row in records),
        },
    }
    manifest = {**body, "manifest_sha256": content_sha256(body)}
    validate_teacher_manifest(manifest, artifacts=artifacts)
    return manifest, artifacts


def validate_teacher_manifest(
    payload: object,
    *,
    root: Path | None = None,
    artifacts: Mapping[str, bytes] | None = None,
) -> dict[str, Any]:
    """Validate a provenance-bound teacher manifest and optional L PNGs."""

    expected = {
        "schema_version",
        "source_binding",
        "source_binding_sha256",
        "producer",
        "producer_sha256",
        "selection_policy",
        "propagation",
        "records",
        "record_counts",
        "manifest_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("teacher mask manifest schema differs")
    body = {key: value for key, value in payload.items() if key != "manifest_sha256"}
    if payload["schema_version"] != TEACHER_SCHEMA or content_sha256(body) != payload["manifest_sha256"]:
        raise ValueError("teacher mask manifest content digest differs")
    for field in ("source_binding", "producer"):
        if not isinstance(payload[field], dict) or not payload[field]:
            raise ValueError(f"teacher mask {field} differs")
        if content_sha256(payload[field]) != payload[f"{field}_sha256"]:
            raise ValueError(f"teacher mask {field} digest differs")
    source_binding = payload["source_binding"]
    required_source_bindings = {
        "source_manifest_schema",
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "source_receipt_filename",
        "source_receipt_file_sha256",
    }
    if not required_source_bindings <= set(source_binding):
        raise ValueError("teacher mask source binding schema differs")
    for name in (
        "source_manifest_file_sha256",
        "source_manifest_payload_sha256",
        "source_receipt_file_sha256",
    ):
        _require_sha256(source_binding[name], f"teacher mask {name}")
    if not isinstance(source_binding["source_manifest_schema"], str) or not source_binding["source_manifest_schema"] or not isinstance(source_binding["source_receipt_filename"], str) or not source_binding["source_receipt_filename"]:
        raise ValueError("teacher mask source binding values differ")
    producer = payload["producer"]
    required_producer = {
        "model_name",
        "sam2_checkout_commit",
        "sam2_python_sources_sha256",
        "sam2_config_relative_path",
        "sam2_config_sha256",
        "sam2_checkpoint_filename",
        "sam2_checkpoint_sha256",
        "license_id",
        "license_snapshot_sha256",
        "device",
        "prompt_contract",
        "output_encoding",
        "tool_provenance",
        "tool_provenance_sha256",
    }
    if set(producer) != required_producer:
        raise ValueError("teacher mask producer schema differs")
    if producer["model_name"] != "sam2.1" or producer["license_id"] != "Apache-2.0":
        raise ValueError("teacher mask model or license differs")
    if not isinstance(producer["sam2_checkout_commit"], str) or _GIT_COMMIT.fullmatch(producer["sam2_checkout_commit"]) is None:
        raise ValueError("teacher mask SAM2 checkout commit differs")
    for name in (
        "sam2_python_sources_sha256",
        "sam2_config_sha256",
        "sam2_checkpoint_sha256",
        "license_snapshot_sha256",
        "tool_provenance_sha256",
    ):
        _require_sha256(producer[name], f"teacher mask producer {name}")
    if not isinstance(producer["tool_provenance"], dict) or content_sha256(producer["tool_provenance"]) != producer["tool_provenance_sha256"]:
        raise ValueError("teacher mask tool provenance digest differs")
    if producer["device"] not in {"cpu", "cuda"} or producer["prompt_contract"] != "NOSE_BOX_AND_POSITIVE_NOSE_KEYPOINTS" or producer["output_encoding"] != "SOURCE_RESOLUTION_BINARY_L_PNG":
        raise ValueError("teacher mask runtime contract differs")
    for name in ("sam2_config_relative_path", "sam2_checkpoint_filename"):
        if not isinstance(producer[name], str) or not producer[name]:
            raise ValueError("teacher mask model artifact name differs")
    policy = MaskSelectionPolicy(**payload["selection_policy"])
    propagation = payload["propagation"]
    propagation_fields = {
        "requested", "api_available", "frame_runs_attempted",
        "frame_runs_propagated", "score_semantics"
    }
    if not isinstance(propagation, dict) or set(propagation) != propagation_fields:
        raise ValueError("teacher mask propagation schema differs")
    if not all(isinstance(propagation[name], bool) for name in ("requested", "api_available")):
        raise ValueError("teacher mask propagation flags differ")
    for name in ("frame_runs_attempted", "frame_runs_propagated"):
        value = propagation[name]
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise ValueError("teacher mask propagation counts differ")
    if propagation["frame_runs_propagated"] > propagation["frame_runs_attempted"]:
        raise ValueError("teacher mask propagation counts differ")
    if propagation["score_semantics"] != "VIDEO_LOGIT_FOREGROUND_MEAN":
        raise ValueError("teacher mask propagation score semantics differ")
    records = payload["records"]
    if not isinstance(records, list) or not records or records != sorted(records, key=lambda row: row.get("sample_token", "")):
        raise ValueError("teacher mask records must be non-empty and sorted")
    resolved_root = root.resolve(strict=True) if root is not None else None
    seen: set[str] = set()
    counts = {"ACCEPTED": 0, "REJECTED": 0}
    for row in records:
        _validate_output_record(row, policy)
        token = row["sample_token"]
        if token in seen:
            raise ValueError("teacher mask manifest repeats a sample token")
        seen.add(token)
        counts[row["status"]] += 1
        if row["status"] == "ACCEPTED":
            relative = _safe_relative_path(row["mask_path"], "teacher mask path")
            expected_path = PurePosixPath("masks", f"{token}.png")
            if relative != expected_path:
                raise ValueError("teacher mask path differs")
            mask_bytes = None
            if artifacts is not None:
                mask_bytes = artifacts.get(relative.as_posix())
            elif resolved_root is not None:
                target = resolved_root.joinpath(*relative.parts)
                if target.is_symlink():
                    raise ValueError("teacher mask path is unsafe")
                resolved = target.resolve(strict=True)
                if not resolved.is_relative_to(resolved_root) or not resolved.is_file():
                    raise ValueError("teacher mask path is unsafe")
                mask_bytes = read_retained_regular_file(
                    resolved,
                    maximum_bytes=67_108_864,
                    capture_payload=True,
                    subject="teacher mask",
                ).payload
            if (artifacts is not None or resolved_root is not None) and mask_bytes is None:
                raise ValueError("teacher mask artifact is absent")
            if mask_bytes is not None:
                _validate_mask_bytes(mask_bytes, row)
    if payload["record_counts"] != counts:
        raise ValueError("teacher mask record counts differ")
    return payload


def validate_sam2_artifacts(
    *,
    checkout: Path,
    expected_checkout_commit: str,
    config_path: Path,
    expected_config_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    license_snapshot_path: Path,
    expected_license_snapshot_sha256: str,
) -> dict[str, Any]:
    """Validate caller-supplied local SAM2.1 code and artifact bindings."""

    if _GIT_COMMIT.fullmatch(expected_checkout_commit) is None:
        raise ValueError("SAM2 checkout commit must be lowercase 40-character Git SHA")
    for value, name in (
        (expected_config_sha256, "SAM2 config SHA-256"),
        (expected_checkpoint_sha256, "SAM2 checkpoint SHA-256"),
        (expected_license_snapshot_sha256, "SAM2 license snapshot SHA-256"),
    ):
        _require_sha256(value, name)
    if not checkout.is_absolute() or checkout.is_symlink():
        raise ValueError("SAM2 checkout must be an absolute non-symlink directory")
    for path, name in (
        (config_path, "SAM2 config"),
        (checkpoint_path, "SAM2 checkpoint"),
        (license_snapshot_path, "SAM2 license snapshot"),
    ):
        if not path.is_absolute():
            raise ValueError(f"{name} must be an absolute path")
    resolved_checkout = checkout.resolve(strict=True)
    if not resolved_checkout.is_dir():
        raise ValueError("SAM2 checkout must be a directory")
    resolved_config = config_path.resolve(strict=True)
    if config_path.is_symlink() or not resolved_config.is_relative_to(resolved_checkout):
        raise ValueError("SAM2 config must be a real file inside the checkout")
    commit = subprocess.run(
        ["git", "-C", str(resolved_checkout), "rev-parse", "HEAD"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.strip()
    if commit != expected_checkout_commit:
        raise ValueError("SAM2 checkout commit differs")
    config = read_retained_regular_file(
        resolved_config,
        expected_sha256=expected_config_sha256,
        maximum_bytes=4_194_304,
        subject="SAM2 config",
    )
    checkpoint = read_retained_regular_file(
        checkpoint_path,
        expected_sha256=expected_checkpoint_sha256,
        subject="SAM2 checkpoint",
    )
    license_result = read_retained_regular_file(
        license_snapshot_path,
        expected_sha256=expected_license_snapshot_sha256,
        maximum_bytes=1_048_576,
        capture_payload=True,
        subject="SAM2 license snapshot",
    )
    license_bytes = license_result.payload or b""
    if b"Apache License" not in license_bytes or b"Version 2.0" not in license_bytes:
        raise ValueError("SAM2 license snapshot is not an Apache-2.0 license snapshot")
    source_rows: list[dict[str, Any]] = []
    for path in sorted(resolved_checkout.rglob("*.py")):
        if path.is_symlink() or not path.is_file():
            raise ValueError("SAM2 checkout Python source must be regular non-symlink files")
        item = read_retained_regular_file(path, subject="SAM2 checkout Python source")
        source_rows.append(
            {
                "relative_path": path.relative_to(resolved_checkout).as_posix(),
                "sha256": item.sha256,
                "byte_size": item.byte_count,
            }
        )
    if not source_rows:
        raise ValueError("SAM2 checkout contains no Python sources")
    return {
        "model_name": "sam2.1",
        "sam2_checkout_commit": commit,
        "sam2_python_sources_sha256": content_sha256(source_rows),
        "sam2_config_relative_path": resolved_config.relative_to(resolved_checkout).as_posix(),
        "sam2_config_sha256": config.sha256,
        "sam2_checkpoint_filename": checkpoint_path.name,
        "sam2_checkpoint_sha256": checkpoint.sha256,
        "license_id": "Apache-2.0",
        "license_snapshot_sha256": license_result.sha256,
    }


class _LocalSam2Predictor:
    def __init__(self, image_predictor: Any, video_predictor: Any | None) -> None:
        self._image = image_predictor
        self._video = video_predictor

    def set_image(self, image: np.ndarray) -> None:
        self._image.set_image(image)

    def predict(self, **kwargs: Any) -> Any:
        return self._image.predict(**kwargs)

    @property
    def video_api_available(self) -> bool:
        return self._video is not None

    def propagate_track(self, **kwargs: Any) -> dict[int, tuple[np.ndarray, np.ndarray]]:
        if self._video is None:
            raise RuntimeError("SAM2 video predictor is unavailable")
        images = kwargs["images"]
        with tempfile.TemporaryDirectory(prefix="sam2-video-") as temporary:
            root = Path(temporary)
            for index, image in enumerate(images):
                payload = io.BytesIO()
                Image.fromarray(image, mode="RGB").save(payload, format="PNG", compress_level=9)
                (root / f"{index:08d}.jpg").write_bytes(payload.getvalue())
            state = self._video.init_state(video_path=str(root))
            self._video.reset_state(state)
            self._video.add_new_points_or_box(
                inference_state=state,
                frame_idx=kwargs["seed_index"],
                obj_id=1,
                points=kwargs["point_coords"],
                labels=kwargs["point_labels"],
                box=kwargs["box"],
            )
            result: dict[int, tuple[np.ndarray, np.ndarray]] = {}
            for frame_index, _, logits in self._video.propagate_in_video(state):
                values = _to_numpy(logits)
                if values.ndim == 4:
                    values = values[:, 0]
                masks = values > 0.0
                scores = np.asarray(
                    [
                        float(
                            (1.0 / (1.0 + np.exp(-np.clip(plane[mask], -60.0, 60.0)))).mean()
                        )
                        if np.any(mask)
                        else 0.0
                        for plane, mask in zip(values, masks, strict=True)
                    ],
                    dtype=np.float64,
                )
                result[int(frame_index)] = (masks, scores)
            return result


def load_local_sam2(
    *,
    checkout: Path,
    expected_checkout_commit: str,
    config_path: Path,
    expected_config_sha256: str,
    checkpoint_path: Path,
    expected_checkpoint_sha256: str,
    license_snapshot_path: Path,
    expected_license_snapshot_sha256: str,
    device: str,
    enable_video: bool,
) -> tuple[Any, dict[str, Any]]:
    """Validate artifacts, then lazily import and construct local SAM2.1."""

    provenance = validate_sam2_artifacts(
        checkout=checkout,
        expected_checkout_commit=expected_checkout_commit,
        config_path=config_path,
        expected_config_sha256=expected_config_sha256,
        checkpoint_path=checkpoint_path,
        expected_checkpoint_sha256=expected_checkpoint_sha256,
        license_snapshot_path=license_snapshot_path,
        expected_license_snapshot_sha256=expected_license_snapshot_sha256,
    )
    resolved_checkout = checkout.resolve(strict=True)
    sys.path.insert(0, str(resolved_checkout))
    try:
        from sam2.build_sam import build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        config_name = _sam2_runtime_config_name(resolved_checkout, config_path)
        model = build_sam2(config_name, str(checkpoint_path), device=device)
        image_predictor = SAM2ImagePredictor(model)
        video_predictor = None
        if enable_video:
            from sam2.build_sam import build_sam2_video_predictor

            video_predictor = build_sam2_video_predictor(
                config_name, str(checkpoint_path), device=device
            )
    except (ImportError, ModuleNotFoundError) as exc:
        raise RuntimeError(
            "validated local SAM2 checkout could not be imported; install its declared dependencies"
        ) from exc
    finally:
        if sys.path and sys.path[0] == str(resolved_checkout):
            sys.path.pop(0)
    return _LocalSam2Predictor(image_predictor, video_predictor), provenance


def _sam2_runtime_config_name(checkout: Path, config_path: Path) -> str:
    """Return the package-relative Hydra name expected by official SAM2."""

    package_root = (checkout / "sam2").resolve(strict=True)
    resolved_config = config_path.resolve(strict=True)
    try:
        relative = resolved_config.relative_to(package_root)
    except ValueError as exc:
        raise ValueError("SAM2 runtime config must be inside the sam2 package") from exc
    if not relative.parts or relative.parts[0] != "configs":
        raise ValueError("SAM2 runtime config must be inside sam2/configs")
    return relative.as_posix()


def _validate_source_row(row: object, root: Path) -> TeacherSource:
    if not isinstance(row, dict) or set(row) != _SOURCE_FIELDS:
        raise ValueError("teacher source record schema differs")
    for name in ("sample_token", "sequence_token", "track_token", "source_sha256"):
        _require_sha256(row[name], f"teacher source {name}")
    for name in ("frame_index", "source_width", "source_height"):
        value = row[name]
        minimum = 0 if name == "frame_index" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"teacher source {name} differs")
    box = row["nose_box_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in box):
        raise ValueError("teacher source nose box differs")
    if not (0 <= box[0] < box[2] <= row["source_width"] and 0 <= box[1] < box[3] <= row["source_height"]):
        raise ValueError("teacher source nose box differs")
    keypoints = row["keypoints"]
    if not isinstance(keypoints, list) or len(keypoints) != len(KEYPOINT_ORDER):
        raise ValueError("teacher source keypoints differ")
    positive: list[tuple[float, float]] = []
    for index, (name, point) in enumerate(zip(KEYPOINT_ORDER, keypoints, strict=True)):
        if not isinstance(point, dict) or set(point) != _KEYPOINT_FIELDS or point["name"] != name:
            raise ValueError("teacher source keypoint schema differs")
        for field in _KEYPOINT_FIELDS - {"name"}:
            value = point[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                raise ValueError("teacher source keypoint value differs")
        if not (0.0 <= point["normalized_x"] <= 1.0 and 0.0 <= point["normalized_y"] <= 1.0 and 0.0 <= point["confidence"] <= 1.0):
            raise ValueError("teacher source normalized keypoint differs")
        if not math.isclose(point["source_x"], point["normalized_x"] * row["source_width"], abs_tol=1e-6) or not math.isclose(point["source_y"], point["normalized_y"] * row["source_height"], abs_tol=1e-6):
            raise ValueError("teacher source keypoint coordinate spaces differ")
        if index in NOSE_POINT_INDICES and point["confidence"] > 0.0:
            coordinate = (float(point["source_x"]), float(point["source_y"]))
            if not (box[0] <= coordinate[0] <= box[2] and box[1] <= coordinate[1] <= box[3]):
                raise ValueError("teacher source positive keypoint lies outside nose box")
            positive.append(coordinate)
    if not positive:
        raise ValueError("teacher source has no positive nose keypoints")
    relative = _safe_relative_path(row["source_image_path"], "teacher source image path")
    target = root.joinpath(*relative.parts)
    if target.is_symlink():
        raise ValueError("teacher source image path is unsafe")
    resolved = target.resolve(strict=True)
    if not resolved.is_relative_to(root) or not resolved.is_file():
        raise ValueError("teacher source image path is unsafe")
    retained = read_retained_regular_file(
        resolved,
        expected_sha256=row["source_sha256"],
        maximum_bytes=67_108_864,
        capture_payload=True,
        subject="teacher source image",
    )
    source_bytes = retained.payload or b""
    image = decode_source_image(source_bytes)
    if image.size != (row["source_width"], row["source_height"]):
        raise ValueError("teacher source image dimensions differ")
    return TeacherSource(
        sample_token=row["sample_token"],
        sequence_token=row["sequence_token"],
        track_token=row["track_token"],
        frame_index=row["frame_index"],
        source_sha256=row["source_sha256"],
        source_width=row["source_width"],
        source_height=row["source_height"],
        nose_box_xyxy=tuple(box),
        positive_keypoints_xy=tuple(positive),
        source_bytes=source_bytes,
    )


def _predict_image_candidates(
    predictor: Any, image: np.ndarray, source: TeacherSource
) -> list[tuple[np.ndarray, float, str, str]]:
    if not callable(getattr(predictor, "set_image", None)) or not callable(getattr(predictor, "predict", None)):
        raise TypeError("SAM2 image predictor must provide set_image and predict")
    predictor.set_image(image)
    output = predictor.predict(
        point_coords=np.asarray(source.positive_keypoints_xy, dtype=np.float32),
        point_labels=np.ones(len(source.positive_keypoints_xy), dtype=np.int32),
        box=np.asarray(source.nose_box_xyxy, dtype=np.float32),
        multimask_output=True,
    )
    if not isinstance(output, tuple) or len(output) < 2:
        raise ValueError("SAM2 predictor output schema differs")
    return _parse_candidates(output[0], output[1], source, "IMAGE_PREDICTOR")


def _parse_propagated_candidates(
    output: object, track: Sequence[TeacherSource]
) -> dict[int, list[tuple[np.ndarray, float, str, str]]]:
    if not isinstance(output, Mapping):
        raise ValueError("SAM2 video propagation output must map frame indexes")
    result: dict[int, list[tuple[np.ndarray, float, str, str]]] = {}
    for index, value in output.items():
        if isinstance(index, bool) or not isinstance(index, int) or not 0 <= index < len(track):
            raise ValueError("SAM2 video propagation frame index differs")
        if not isinstance(value, tuple) or len(value) != 2:
            raise ValueError("SAM2 video propagation candidate schema differs")
        result[index] = _parse_candidates(value[0], value[1], track[index], "VIDEO_PROPAGATION")
    return result


def _parse_candidates(
    masks: object, scores: object, source: TeacherSource, origin: str
) -> list[tuple[np.ndarray, float, str, str]]:
    mask_values = _to_numpy(masks)
    score_values = _to_numpy(scores).astype(np.float64, copy=False)
    if mask_values.ndim != 3 or mask_values.shape[1:] != (source.source_height, source.source_width):
        raise ValueError("SAM2 masks must be source-resolution [N,H,W]")
    if score_values.shape != (mask_values.shape[0],) or mask_values.shape[0] == 0:
        raise ValueError("SAM2 mask scores must be [N]")
    if not np.isfinite(mask_values).all() or not np.isfinite(score_values).all() or np.any((score_values < 0.0) | (score_values > 1.0)):
        raise ValueError("SAM2 masks and scores must be finite with scores in [0,1]")
    result: list[tuple[np.ndarray, float, str, str]] = []
    fingerprints: set[str] = set()
    for mask, score in zip(mask_values, score_values, strict=True):
        binary = np.asarray(mask > 0, dtype=bool)
        fingerprint = hashlib.sha256(binary.tobytes()).hexdigest()
        if fingerprint in fingerprints:
            continue
        fingerprints.add(fingerprint)
        semantics = (
            "PREDICTED_IOU"
            if origin == "IMAGE_PREDICTOR"
            else "VIDEO_LOGIT_FOREGROUND_MEAN"
        )
        result.append((binary, float(score), origin, semantics))
    return result


def _select_mask(
    source: TeacherSource,
    candidates: Sequence[tuple[np.ndarray, float, str, str]],
    policy: MaskSelectionPolicy,
) -> dict[str, Any]:
    unique: dict[str, tuple[int, np.ndarray, float, str, str]] = {}
    for index, (mask, model_score, origin, score_semantics) in enumerate(candidates):
        fingerprint = hashlib.sha256(mask.tobytes()).hexdigest()
        previous = unique.get(fingerprint)
        if previous is None or model_score > previous[2]:
            unique[fingerprint] = (
                index,
                mask,
                model_score,
                origin,
                score_semantics,
            )
    normalized = sorted(unique.values(), key=lambda item: item[0])
    summaries: list[dict[str, Any]] = []
    masks_by_index: dict[int, np.ndarray] = {}
    for index, mask, model_score, origin, score_semantics in normalized:
        masks_by_index[index] = mask
        anatomy, compactness, area_ratio = _mask_geometry(source, mask)
        combined = (
            policy.model_score_weight * model_score
            + policy.anatomical_overlap_weight * anatomy
            + policy.compactness_weight * compactness
        )
        reasons: list[str] = []
        if not np.any(mask):
            reasons.append("EMPTY_MASK")
        if model_score < policy.minimum_model_score:
            reasons.append("LOW_MODEL_SCORE")
        if anatomy < policy.minimum_anatomical_overlap:
            reasons.append("LOW_ANATOMICAL_OVERLAP")
        if compactness < policy.minimum_compactness:
            reasons.append("LOW_COMPACTNESS")
        if area_ratio < policy.minimum_area_to_box_ratio:
            reasons.append("MASK_TOO_SMALL")
        if area_ratio > policy.maximum_area_to_box_ratio:
            reasons.append("MASK_TOO_LARGE")
        summaries.append(
            {
                "candidate_index": index,
                "origin": origin,
                "score_semantics": score_semantics,
                "model_score": model_score,
                "anatomical_overlap": anatomy,
                "compactness": compactness,
                "area_pixels": int(mask.sum()),
                "area_to_box_ratio": area_ratio,
                "combined_score": combined,
                "eligible": not reasons,
                "rejection_reasons": reasons,
            }
        )
    eligible = [item for item in summaries if item["eligible"]]
    eligible.sort(key=lambda item: (-item["combined_score"], item["candidate_index"]))
    rejection_reasons: list[str] = []
    chosen = eligible[0] if eligible else None
    if chosen is None:
        rejection_reasons.append("NO_ELIGIBLE_MASK")
    elif len(eligible) > 1 and chosen["combined_score"] - eligible[1]["combined_score"] < policy.ambiguity_margin:
        rejection_reasons.append("AMBIGUOUS_MASKS")
        chosen = None
    return {
        "mask": None if chosen is None else masks_by_index[chosen["candidate_index"]],
        "selected_score": -1.0 if chosen is None else chosen["combined_score"],
        "rejection_reasons": rejection_reasons,
        "selected_candidate_index": None if chosen is None else chosen["candidate_index"],
        "candidates": summaries,
    }


def _mask_geometry(source: TeacherSource, mask: np.ndarray) -> tuple[float, float, float]:
    area = int(mask.sum())
    left, top, right, bottom = source.nose_box_xyxy
    box_area = (right - left) * (bottom - top)
    if area == 0:
        return 0.0, 0.0, 0.0
    inside = int(mask[top:bottom, left:right].sum()) / area
    point_hits = np.mean(
        [
            mask[
                min(source.source_height - 1, max(0, int(round(y)))),
                min(source.source_width - 1, max(0, int(round(x)))),
            ]
            for x, y in source.positive_keypoints_xy
        ]
    )
    prior = np.zeros(mask.shape, dtype=np.uint8)
    points = np.rint(np.asarray(source.positive_keypoints_xy)).astype(np.int32)
    if len(points) >= 3:
        cv2.fillConvexPoly(prior, cv2.convexHull(points), 1)
    else:
        for x, y in points:
            cv2.circle(prior, (int(x), int(y)), 1, 1, -1)
    radius = max(2, int(math.ceil(min(right - left, bottom - top) * 0.12)))
    prior = cv2.dilate(prior, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (radius * 2 + 1, radius * 2 + 1)))
    prior_area = int(prior.sum())
    prior_recall = float(np.logical_and(mask, prior > 0).sum() / prior_area) if prior_area else 0.0
    anatomy = float(0.40 * point_hits + 0.35 * prior_recall + 0.25 * inside)
    values = np.where(mask, 255, 0).astype(np.uint8)
    contours, _ = cv2.findContours(values, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    perimeter = sum(float(cv2.arcLength(contour, True)) for contour in contours)
    shape_compactness = min(1.0, 4.0 * math.pi * area / (perimeter * perimeter)) if perimeter > 0.0 else 0.0
    component_count, _, stats, _ = cv2.connectedComponentsWithStats(values, connectivity=8)
    largest = int(stats[1:, cv2.CC_STAT_AREA].max()) if component_count > 1 else 0
    compactness = float(shape_compactness * largest / area)
    return anatomy, compactness, float(area / box_area)


def _validate_output_record(row: object, policy: MaskSelectionPolicy) -> None:
    if not isinstance(row, dict) or set(row) != _OUTPUT_RECORD_FIELDS:
        raise ValueError("teacher mask record schema differs")
    for name in ("sample_token", "sequence_token", "track_token", "source_sha256"):
        _require_sha256(row[name], f"teacher mask {name}")
    if row["coordinate_space"] != "SOURCE_IMAGE_PIXELS" or row["status"] not in {"ACCEPTED", "REJECTED"}:
        raise ValueError("teacher mask record coordinate space or status differs")
    for name in ("frame_index", "source_width", "source_height"):
        value = row[name]
        minimum = 0 if name == "frame_index" else 1
        if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
            raise ValueError(f"teacher mask {name} differs")
    box = row["nose_box_xyxy"]
    if not isinstance(box, list) or len(box) != 4 or any(isinstance(value, bool) or not isinstance(value, int) for value in box) or not (0 <= box[0] < box[2] <= row["source_width"] and 0 <= box[1] < box[3] <= row["source_height"]):
        raise ValueError("teacher mask nose box differs")
    points = row["positive_keypoints_xy"]
    if not isinstance(points, list) or not points:
        raise ValueError("teacher mask positive keypoints differ")
    for point in points:
        if not isinstance(point, list) or len(point) != 2 or any(isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) for value in point):
            raise ValueError("teacher mask positive keypoints differ")
    if not isinstance(row["selection"], dict) or set(row["selection"]) != {"selected_candidate_index", "candidates"}:
        raise ValueError("teacher mask selection schema differs")
    candidates = row["selection"]["candidates"]
    if not isinstance(candidates, list):
        raise ValueError("teacher mask candidates differ")
    candidate_fields = {
        "candidate_index", "origin", "score_semantics", "model_score", "anatomical_overlap",
        "compactness", "area_pixels", "area_to_box_ratio", "combined_score",
        "eligible", "rejection_reasons",
    }
    indexes: set[int] = set()
    for candidate in candidates:
        if not isinstance(candidate, dict) or set(candidate) != candidate_fields:
            raise ValueError("teacher mask candidate schema differs")
        index = candidate["candidate_index"]
        if isinstance(index, bool) or not isinstance(index, int) or index < 0 or index in indexes:
            raise ValueError("teacher mask candidate index differs")
        indexes.add(index)
        if candidate["origin"] not in {"IMAGE_PREDICTOR", "VIDEO_PROPAGATION"}:
            raise ValueError("teacher mask candidate origin differs")
        expected_semantics = (
            "PREDICTED_IOU"
            if candidate["origin"] == "IMAGE_PREDICTOR"
            else "VIDEO_LOGIT_FOREGROUND_MEAN"
        )
        if candidate["score_semantics"] != expected_semantics:
            raise ValueError("teacher mask candidate score semantics differ")
        for name in (
            "model_score", "anatomical_overlap", "compactness",
            "area_to_box_ratio", "combined_score",
        ):
            value = candidate[name]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0.0:
                raise ValueError("teacher mask candidate score differs")
        for name in ("model_score", "anatomical_overlap", "compactness"):
            if candidate[name] > 1.0:
                raise ValueError("teacher mask candidate score differs")
        if isinstance(candidate["area_pixels"], bool) or not isinstance(candidate["area_pixels"], int) or candidate["area_pixels"] < 0:
            raise ValueError("teacher mask candidate area differs")
        if candidate["area_pixels"] > row["source_width"] * row["source_height"]:
            raise ValueError("teacher mask candidate area differs")
        if not isinstance(candidate["eligible"], bool) or not isinstance(candidate["rejection_reasons"], list) or any(not isinstance(value, str) or not value for value in candidate["rejection_reasons"]):
            raise ValueError("teacher mask candidate rejection differs")
        expected_combined = (
            policy.model_score_weight * candidate["model_score"]
            + policy.anatomical_overlap_weight * candidate["anatomical_overlap"]
            + policy.compactness_weight * candidate["compactness"]
        )
        if not math.isclose(candidate["combined_score"], expected_combined, abs_tol=1e-12):
            raise ValueError("teacher mask candidate combined score differs")
        expected_reasons: list[str] = []
        if candidate["area_pixels"] == 0:
            expected_reasons.append("EMPTY_MASK")
        if candidate["model_score"] < policy.minimum_model_score:
            expected_reasons.append("LOW_MODEL_SCORE")
        if candidate["anatomical_overlap"] < policy.minimum_anatomical_overlap:
            expected_reasons.append("LOW_ANATOMICAL_OVERLAP")
        if candidate["compactness"] < policy.minimum_compactness:
            expected_reasons.append("LOW_COMPACTNESS")
        if candidate["area_to_box_ratio"] < policy.minimum_area_to_box_ratio:
            expected_reasons.append("MASK_TOO_SMALL")
        if candidate["area_to_box_ratio"] > policy.maximum_area_to_box_ratio:
            expected_reasons.append("MASK_TOO_LARGE")
        if candidate["rejection_reasons"] != expected_reasons or candidate["eligible"] != (not expected_reasons):
            raise ValueError("teacher mask candidate eligibility differs")
    if not isinstance(row["rejection_reasons"], list) or any(not isinstance(value, str) or not value for value in row["rejection_reasons"]):
        raise ValueError("teacher mask rejection reasons differ")
    eligible = sorted(
        (candidate for candidate in candidates if candidate["eligible"]),
        key=lambda candidate: (-candidate["combined_score"], candidate["candidate_index"]),
    )
    ambiguous = (
        len(eligible) > 1
        and eligible[0]["combined_score"] - eligible[1]["combined_score"]
        < policy.ambiguity_margin
    )
    if row["status"] == "ACCEPTED":
        if row["rejection_reasons"] or row["mask_path"] is None or row["selection"]["selected_candidate_index"] is None:
            raise ValueError("accepted teacher mask record differs")
        _require_sha256(row["mask_sha256"], "teacher mask SHA-256")
        if not eligible or ambiguous or row["selection"]["selected_candidate_index"] != eligible[0]["candidate_index"]:
            raise ValueError("accepted teacher mask selection differs")
    else:
        expected_record_reasons = ["AMBIGUOUS_MASKS"] if ambiguous else ["NO_ELIGIBLE_MASK"]
        if row["mask_path"] is not None or row["mask_sha256"] is not None or row["rejection_reasons"] != expected_record_reasons or row["selection"]["selected_candidate_index"] is not None:
            raise ValueError("rejected teacher mask record differs")


def _validate_mask_bytes(payload: bytes, row: Mapping[str, Any]) -> None:
    if hashlib.sha256(payload).hexdigest() != row["mask_sha256"]:
        raise ValueError("teacher mask SHA-256 differs")
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.format != "PNG" or image.mode != "L" or image.size != (row["source_width"], row["source_height"]):
                raise ValueError("teacher mask must be a source-resolution L PNG")
            values = np.asarray(image)
    except (OSError, SyntaxError) as exc:
        raise ValueError("teacher mask is not a valid PNG") from exc
    if not np.isin(values, (0, 255)).all() or not np.any(values == 255):
        raise ValueError("teacher mask pixels must be non-empty binary L values")


def _mask_png(mask: np.ndarray) -> bytes:
    stream = io.BytesIO()
    Image.fromarray(np.where(mask, 255, 0).astype(np.uint8), mode="L").save(
        stream, format="PNG", optimize=False, compress_level=9
    )
    return stream.getvalue()


def _to_numpy(value: object) -> np.ndarray:
    current = value
    for method in ("detach", "cpu"):
        operation = getattr(current, method, None)
        if callable(operation):
            current = operation()
    return np.asarray(current)


def _safe_relative_path(value: object, name: str) -> PurePosixPath:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError(f"{name} is unsafe")
    path = PurePosixPath(value)
    if path.is_absolute() or value != path.as_posix() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"{name} is unsafe")
    return path


def _require_sha256(value: object, name: str) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")
    return value


__all__ = [
    "MaskSelectionPolicy",
    "SOURCE_IMAGE_MANIFEST_SCHEMA",
    "TeacherSource",
    "load_local_sam2",
    "produce_teacher_manifest",
    "sources_from_native_manifest",
    "validate_sam2_artifacts",
    "validate_source_image_manifest",
    "validate_teacher_manifest",
]
