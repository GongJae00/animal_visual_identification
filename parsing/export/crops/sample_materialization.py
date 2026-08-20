"""Materialize one common Full-segment record from frozen parsing or native input."""

from __future__ import annotations

import hashlib
import io
import json
import os
import stat
from pathlib import Path
from typing import Any

from PIL import Image, UnidentifiedImageError

from parsing.export.segmentation.full_segment_cache import (
    build_body_mask_observation,
    build_body_observation,
    build_frozen_parsing_binding,
    build_full_segment_cache,
    thaw_animal_parsing_prediction,
)
from parsing.export.segmentation.full_segment_contracts import (
    AnimalAssociation,
    BodyMaskPolicy,
    ObservationRoute,
    SourceViewScope,
    TerminalObservability,
    build_native_observation,
    build_terminal_observation,
)
from parsing.export.crops.full_segment_crop import (
    materialize_body_mask_full_crop,
    materialize_full_crop,
    materialize_native_full_crop,
)

REQUEST_SCHEMA = "cvi.full_segment_materialization_request.v1"
_REQUEST_FIELDS = {
    "schema_version",
    "source_id",
    "source_image_path",
    "source_sha256",
    "source_view_scope",
    "route",
    "frozen_parsing_path",
    "association",
    "face_observability",
    "nose_observability",
    "target_size",
    "context_fraction",
    "background_rgb",
}
_BODY_MASK_REQUEST_FIELDS = _REQUEST_FIELDS | {
    "authoritative_mask_path",
    "authoritative_mask_sha256",
    "body_mask_policy",
}
_MAX_SOURCE_BYTES = 67_108_864
_MAX_JSON_BYTES = 536_870_912


def run(request: object, *, output_dir: Path) -> dict[str, Any]:
    parsed = _parse_request(request)
    source_bytes = _read_regular_file(
        Path(parsed["source_image_path"]),
        maximum_bytes=_MAX_SOURCE_BYTES,
        label="Full segment source image",
    )
    return _run_parsed(parsed, output_dir=output_dir, source_bytes=source_bytes)


def run_prevalidated(
    request: object,
    *,
    output_dir: Path,
    source_bytes: bytes,
    frozen_parsing: dict[str, Any] | None,
    frozen_json_sha256: str | None,
) -> dict[str, Any]:
    """Materialize from bytes already verified by the enclosing Full128 job."""

    parsed = _parse_request(request)
    if not isinstance(source_bytes, bytes) or not source_bytes:
        raise ValueError("prevalidated Full segment source bytes must be non-empty")
    route = ObservationRoute(parsed["route"])
    if route is ObservationRoute.BODY_PARSING:
        if frozen_parsing is None or frozen_json_sha256 is None:
            raise ValueError("prevalidated body parsing inputs are incomplete")
        _require_sha256(frozen_json_sha256, "prevalidated frozen parsing JSON")
        prevalidated = (frozen_parsing, frozen_json_sha256)
    else:
        if frozen_parsing is not None or frozen_json_sha256 is not None:
            raise ValueError("non-body prevalidated input cannot include parsing")
        prevalidated = None
    return _run_parsed(
        parsed,
        output_dir=output_dir,
        source_bytes=source_bytes,
        prevalidated_frozen=prevalidated,
    )


def _run_parsed(
    parsed: dict[str, Any],
    *,
    output_dir: Path,
    source_bytes: bytes,
    prevalidated_frozen: tuple[dict[str, Any], str] | None = None,
) -> dict[str, Any]:
    if output_dir.exists():
        raise FileExistsError(
            f"refusing to overwrite Full segment output: {output_dir}"
        )
    if not output_dir.parent.is_dir():
        raise ValueError("Full segment output parent must exist")
    source_sha256 = hashlib.sha256(source_bytes).hexdigest()
    if source_sha256 != parsed["source_sha256"]:
        raise ValueError("Full segment request source digest differs")
    width, height = _image_size(source_bytes)
    route = ObservationRoute(parsed["route"])
    scope = SourceViewScope(parsed["source_view_scope"])
    face = TerminalObservability(parsed["face_observability"])
    nose = TerminalObservability(parsed["nose_observability"])
    frozen: dict[str, Any] | None = None
    frozen_json_sha256: str | None = None
    crop = None

    if route is ObservationRoute.BODY_PARSING:
        if prevalidated_frozen is None:
            frozen, frozen_json_sha256 = _read_json_object_with_sha256(
                Path(parsed["frozen_parsing_path"]), label="frozen animal parsing"
            )
        else:
            frozen, frozen_json_sha256 = prevalidated_frozen
        association = AnimalAssociation.from_dict(parsed["association"])
        prediction = thaw_animal_parsing_prediction(frozen)
        if prediction.source_width != width or prediction.source_height != height:
            raise ValueError("source image dimensions differ from frozen parsing")
        observation = build_body_observation(
            source_id=parsed["source_id"],
            source_sha256=source_sha256,
            source_view_scope=scope,
            frozen_parsing=frozen,
            association=association,
            face_observability=face,
            nose_observability=nose,
        )
        selected = prediction.instances[association.instance_index]
        if selected.quality.state in {"USABLE", "REVIEW"}:
            crop = materialize_full_crop(
                source_bytes,
                expected_source_sha256=source_sha256,
                frozen_parsing=frozen,
                association=association,
                target_size=parsed["target_size"],
                context_fraction=parsed["context_fraction"],
                background_rgb=tuple(parsed["background_rgb"]),
            )
            observation = build_body_observation(
                source_id=parsed["source_id"],
                source_sha256=source_sha256,
                source_view_scope=scope,
                frozen_parsing=frozen,
                association=association,
                face_observability=face,
                nose_observability=nose,
                full_rgb_sha256=crop.record["full_rgb_sha256"],
            )
    elif route is ObservationRoute.BODY_MASK:
        authoritative_mask_bytes = _read_regular_file(
            Path(parsed["authoritative_mask_path"]),
            maximum_bytes=_MAX_SOURCE_BYTES,
            label="authoritative body mask",
        )
        policy = BodyMaskPolicy.from_dict(parsed["body_mask_policy"])
        crop = materialize_body_mask_full_crop(
            source_bytes,
            authoritative_mask_bytes,
            expected_source_sha256=source_sha256,
            expected_authoritative_mask_sha256=parsed["authoritative_mask_sha256"],
            policy=policy,
            target_size=parsed["target_size"],
            context_fraction=parsed["context_fraction"],
            background_rgb=tuple(parsed["background_rgb"]),
        )
        observation = build_body_mask_observation(
            source_id=parsed["source_id"],
            source_sha256=source_sha256,
            source_width=width,
            source_height=height,
            source_view_scope=scope,
            authoritative_mask_sha256=crop.record["authoritative_mask_sha256"],
            mask_policy_sha256=policy.policy_sha256,
            full_rgb_sha256=crop.record["full_rgb_sha256"],
            face_observability=face,
            nose_observability=nose,
        )
    elif route in {ObservationRoute.NATIVE_FACE, ObservationRoute.NATIVE_HEAD}:
        crop = materialize_native_full_crop(
            source_bytes,
            expected_source_sha256=source_sha256,
            route=route,
            target_size=parsed["target_size"],
            background_rgb=tuple(parsed["background_rgb"]),
        )
        observation = build_native_observation(
            source_id=parsed["source_id"],
            source_sha256=source_sha256,
            source_width=width,
            source_height=height,
            source_view_scope=scope,
            native_artifact_sha256=source_sha256,
            full_rgb_sha256=crop.record["full_rgb_sha256"],
            nose_observability=nose,
        )
    else:
        observation = build_terminal_observation(
            source_id=parsed["source_id"],
            source_sha256=source_sha256,
            source_width=width,
            source_height=height,
            source_view_scope=scope,
            face_observability=face,
            nose_observability=nose,
        )

    frozen_record: dict[str, Any] | None = None
    if route is ObservationRoute.BODY_PARSING:
        if (
            frozen is None
            or frozen_json_sha256 is None
            or observation.association is None
        ):
            raise AssertionError("body parsing materialization lost frozen provenance")
        frozen_record = build_frozen_parsing_binding(
            frozen,
            frozen_json_sha256=frozen_json_sha256,
            source_id=parsed["source_id"],
            source_sha256=source_sha256,
            association=observation.association,
        )
    cache_bundle = build_full_segment_cache(
        (
            {
                "source_id": parsed["source_id"],
                "observation": observation.to_dict(),
                "frozen_parsing": frozen_record,
                "crop": None if crop is None else crop.record,
            },
        )
    )
    output_dir.mkdir(mode=0o700)
    if crop is not None:
        _write_new(output_dir / "full.png", crop.full_rgb_png)
        _write_new(output_dir / "full-mask.png", crop.full_mask_png)
    _write_new(
        output_dir / "full-segment-observation.json",
        _json_bytes(observation.to_dict()),
    )
    _write_new(
        output_dir / "full-segment-cache.json",
        _json_bytes(cache_bundle),
    )
    return cache_bundle


def _parse_request(value: object) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise TypeError("Full segment materialization request must be an object")
    try:
        declared_route = ObservationRoute(value.get("route"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Full segment request route differs") from exc
    expected_fields = (
        _BODY_MASK_REQUEST_FIELDS
        if declared_route is ObservationRoute.BODY_MASK
        else _REQUEST_FIELDS
    )
    if set(value) != expected_fields:
        raise ValueError("Full segment materialization request schema differs")
    if value["schema_version"] != REQUEST_SCHEMA:
        raise ValueError("unsupported Full segment materialization request schema")
    if not isinstance(value["source_id"], str) or not value["source_id"]:
        raise ValueError("Full segment request source ID must be non-empty")
    source_path = value["source_image_path"]
    if (
        not isinstance(source_path, str)
        or not source_path
        or not Path(source_path).is_absolute()
    ):
        raise ValueError("Full segment source image path must be absolute")
    _require_sha256(value["source_sha256"], "Full segment request source")
    try:
        scope = SourceViewScope(value["source_view_scope"])
        route = ObservationRoute(value["route"])
        face = TerminalObservability(value["face_observability"])
        nose = TerminalObservability(value["nose_observability"])
    except (TypeError, ValueError) as exc:
        raise ValueError("Full segment request enum value differs") from exc
    parsing_path = value["frozen_parsing_path"]
    association = value["association"]
    if route is ObservationRoute.BODY_PARSING:
        if (
            not isinstance(parsing_path, str)
            or not parsing_path
            or not Path(parsing_path).is_absolute()
            or association is None
            or scope
            not in {SourceViewScope.BODY_AVAILABLE, SourceViewScope.BODY_TRUNCATED}
        ):
            raise ValueError(
                "body request requires parsing path, association, and body scope"
            )
        AnimalAssociation.from_dict(association)
        if TerminalObservability.NATIVE in {face, nose}:
            raise ValueError("body request cannot claim native observability")
    elif route is ObservationRoute.BODY_MASK:
        mask_path = value["authoritative_mask_path"]
        if (
            parsing_path is not None
            or association is not None
            or not isinstance(mask_path, str)
            or not mask_path
            or not Path(mask_path).is_absolute()
            or scope
            not in {SourceViewScope.BODY_AVAILABLE, SourceViewScope.BODY_TRUNCATED}
        ):
            raise ValueError(
                "body-mask request requires a mask path, no parsing, and body scope"
            )
        _require_sha256(
            value["authoritative_mask_sha256"],
            "Full segment request authoritative mask",
        )
        BodyMaskPolicy.from_dict(value["body_mask_policy"])
        if TerminalObservability.NATIVE in {face, nose}:
            raise ValueError("body-mask request cannot claim native observability")
    elif parsing_path is not None or association is not None:
        raise ValueError("non-body request cannot include parsing or association")
    if (
        route is ObservationRoute.NATIVE_FACE
        and scope is not SourceViewScope.FACE_NATIVE
    ):
        raise ValueError("native face request scope differs")
    if (
        route is ObservationRoute.NATIVE_HEAD
        and scope is not SourceViewScope.HEAD_NATIVE
    ):
        raise ValueError("native head request scope differs")
    if (
        route in {ObservationRoute.NATIVE_FACE, ObservationRoute.NATIVE_HEAD}
        and face is not TerminalObservability.NATIVE
    ):
        raise ValueError("native request requires native Face observability")
    if route is ObservationRoute.NONE and scope not in {
        SourceViewScope.AMBIGUOUS,
        SourceViewScope.UNAVAILABLE,
    }:
        raise ValueError("terminal request scope differs")
    target_size = value["target_size"]
    context_fraction = value["context_fraction"]
    background = value["background_rgb"]
    if (
        isinstance(target_size, bool)
        or not isinstance(target_size, int)
        or not 1 <= target_size <= 4096
        or isinstance(context_fraction, bool)
        or not isinstance(context_fraction, (int, float))
        or not 0.0 <= context_fraction <= 1.0
        or not isinstance(background, list)
        or len(background) != 3
        or any(
            isinstance(item, bool) or not isinstance(item, int) or not 0 <= item <= 255
            for item in background
        )
    ):
        raise ValueError("Full segment request crop policy differs")
    return value


def _read_regular_file(path: Path, *, maximum_bytes: int, label: str) -> bytes:
    if not path.is_absolute():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    try:
        resolved = path.resolve(strict=True)
        metadata = path.stat(follow_symlinks=False)
    except FileNotFoundError:
        raise FileNotFoundError(f"{label} does not exist: {path}") from None
    except (OSError, RuntimeError) as exc:
        raise ValueError(f"{label} path cannot be resolved safely") from exc
    if resolved != path or path.is_symlink():
        raise ValueError(f"{label} must be an absolute non-symlink file")
    if not stat.S_ISREG(metadata.st_mode) or not 0 < metadata.st_size <= maximum_bytes:
        raise ValueError(f"{label} size or file type differs")
    flags = os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0) | getattr(os, "O_CLOEXEC", 0)
    try:
        descriptor = os.open(path, flags)
    except OSError as exc:
        raise ValueError(f"{label} cannot be opened safely") from exc
    chunks: list[bytes] = []
    try:
        before = os.fstat(descriptor)
        while chunk := os.read(descriptor, 1_048_576):
            chunks.append(chunk)
        after = os.fstat(descriptor)
        try:
            current = path.stat(follow_symlinks=False)
        except FileNotFoundError:
            raise RuntimeError(f"{label} changed while being read") from None
    finally:
        os.close(descriptor)
    identity = lambda item: (
        item.st_dev,
        item.st_ino,
        item.st_mode,
        item.st_size,
        item.st_mtime_ns,
        item.st_ctime_ns,
    )
    payload = b"".join(chunks)
    if (
        not (
            identity(metadata)
            == identity(before)
            == identity(after)
            == identity(current)
        )
        or len(payload) != before.st_size
    ):
        raise RuntimeError(f"{label} changed while being read")
    return payload


def _read_json_object(path: Path, *, label: str) -> dict[str, Any]:
    value, _ = _read_json_object_with_sha256(path, label=label)
    return value


def _read_json_object_with_sha256(
    path: Path, *, label: str
) -> tuple[dict[str, Any], str]:
    payload = _read_regular_file(path, maximum_bytes=_MAX_JSON_BYTES, label=label)
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"{label} must be UTF-8 JSON") from exc
    if not isinstance(value, dict):
        raise TypeError(f"{label} must contain an object")
    return value, hashlib.sha256(payload).hexdigest()


def _image_size(payload: bytes) -> tuple[int, int]:
    try:
        with Image.open(io.BytesIO(payload)) as image:
            if image.width * image.height > 33_554_432:
                raise ValueError("Full segment source pixel count exceeds policy")
            image.load()
            return image.size
    except (UnidentifiedImageError, OSError) as exc:
        raise ValueError("Full segment source is not a supported image") from exc


def _json_bytes(value: dict[str, Any]) -> bytes:
    return json.dumps(
        value,
        ensure_ascii=False,
        allow_nan=False,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")


def _write_new(path: Path, payload: bytes) -> None:
    temporary = path.with_name(f".{path.name}.tmp")
    try:
        with temporary.open("xb") as stream:
            stream.write(payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _require_sha256(value: object, label: str) -> None:
    if (
        not isinstance(value, str)
        or len(value) != 64
        or any(character not in "0123456789abcdef" for character in value)
    ):
        raise ValueError(f"{label} must be lowercase SHA-256")
