from __future__ import annotations

from collections.abc import Sequence
from hashlib import sha256
from pathlib import Path

_REJECTED_SUPERANIMAL_ONNX_SHA256 = {
    "243ef8a034a20ceec32fcf2963ebd2174b4737ca978bd8b6bb4b8087033f2381",
}
_MAXIMUM_LEGACY_ONNX_BYTES = 2_147_483_648


def reject_unverified_superanimal_onnx(
    model_path: Path,
    *,
    model_sha256: str | None = None,
    input_shape: Sequence[object] | None = None,
    output_shape: Sequence[object] | None = None,
) -> None:
    if "superanimal" in model_path.name.lower():
        raise RuntimeError(
            "SuperAnimal ONNX artifacts are disabled; no verified export "
            "and decoder contract exists"
        )
    if model_sha256 in _REJECTED_SUPERANIMAL_ONNX_SHA256:
        raise RuntimeError("Rejected a known invalid SuperAnimal ONNX artifact")
    if (
        input_shape is not None
        and output_shape is not None
        and list(input_shape[-3:]) == [3, 384, 384]
        and len(output_shape) == 2
        and output_shape[-1] == 39
    ):
        raise RuntimeError(
            "Rejected the unverified SuperAnimal replacement ONNX contract "
            "[batch,3,384,384] -> [batch,39]"
        )


def validated_onnx_bytes(
    model_path: Path,
    *,
    maximum_bytes: int = _MAXIMUM_LEGACY_ONNX_BYTES,
) -> bytes:
    """Inspect a bounded ONNX artifact before exposing it to ONNX Runtime."""
    reject_unverified_superanimal_onnx(model_path)
    before = model_path.stat()
    if not model_path.is_file() or before.st_size <= 0:
        raise ValueError("ONNX model must be a non-empty regular file")
    if before.st_size > maximum_bytes:
        raise ValueError("ONNX model exceeds the byte cap")
    with model_path.open("rb") as stream:
        payload = stream.read(maximum_bytes + 1)
    after = model_path.stat()
    if len(payload) > maximum_bytes:
        raise ValueError("ONNX model exceeds the byte cap")
    if (
        before.st_size != after.st_size
        or before.st_mtime_ns != after.st_mtime_ns
        or before.st_ino != after.st_ino
        or len(payload) != before.st_size
    ):
        raise ValueError("ONNX model changed while being inspected")

    digest = sha256(payload).hexdigest()
    reject_unverified_superanimal_onnx(model_path, model_sha256=digest)
    import onnx

    try:
        model = onnx.load_model_from_string(payload)
        onnx.checker.check_model(model)
    except Exception as exc:
        raise ValueError("ONNX model failed pre-runtime validation") from exc
    if len(model.graph.input) == 1 and len(model.graph.output) == 1:
        reject_unverified_superanimal_onnx(
            model_path,
            model_sha256=digest,
            input_shape=_onnx_shape(model.graph.input[0]),
            output_shape=_onnx_shape(model.graph.output[0]),
        )
    return payload


def _onnx_shape(value_info: object) -> tuple[object, ...]:
    dimensions: list[object] = []
    for dimension in value_info.type.tensor_type.shape.dim:  # type: ignore[attr-defined]
        if dimension.dim_value > 0:
            dimensions.append(int(dimension.dim_value))
        elif dimension.dim_param:
            dimensions.append(dimension.dim_param)
        else:
            dimensions.append(None)
    return tuple(dimensions)
