"""Receipt-bound admission of the fixed Meta PDQ image regression.

This boundary authenticates the already admitted source/regression bundle and
the portable native worker, decodes the eight official bridge JPEGs, and
requires exact agreement with the fixed-commit outputs.  It admits only this
regression behavior; it does not admit corpus thresholds, recall, performance,
or duplicate decisions.
"""

from __future__ import annotations

import io
import json
import re
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any

from data.audit.pdq.contracts import PDQ_D4_ORIENTATIONS
from data.audit.pdq.native import (
    CanonicalRGBRequest,
    PdqNativeBuildReceipt,
    hash_rgb_batch,
    verify_native_pdq_build,
)
from data.audit.pdq.regression_source_intake import (
    PDQ_REGRESSION_ASSET_PATHS,
    validate_pdq_regression_source_contract,
)
from data.audit.pdq.source_intake import PdqSourceContract, PdqSourceIntakeReceipt
from shared.foundation.protected_io import read_strict_json_object, write_private_json_bundle
from shared.foundation.provenance import content_sha256
from shared.foundation.retained_file import read_retained_regular_file


CANONICAL_REGRESSION_BUNDLE_SHA256 = (
    "70d352012fe55afd29f1304f668f685b1a121786142430147e1e5a78c81e2523"
)
CANONICAL_REGRESSION_CONTRACT_SHA256 = (
    "b6363d8504af8d2e64886af08812c58494f271173f1d46436f1681a859ee79a4"
)
CANONICAL_REGRESSION_RECEIPT_SHA256 = (
    "8e69899340019ef690c03ef232a68d62908bf0e53eb43396007129e88276775a"
)
CANONICAL_REGRESSION_AGGREGATE_SHA256 = (
    "45a33513cda19d6f03dfe183b1351ff8dedae47c5ab8a06ef2d4536d62b28311"
)
CANONICAL_EXPECTED_OUTPUT_SHA256 = (
    "a911dd0f5b8795268bf2eae94e8a6089e7d52d2a52b8f132e273444e789863d7"
)
CANONICAL_NATIVE_BUILD_RECEIPT_SHA256 = (
    "8aaa3b4204516ff1df70c208610ba45fd04032a88566e12380ed3db833f6c752"
)
CANONICAL_NATIVE_BINARY_SHA256 = (
    "b4774cfba578f6dd235b3892e6f880fad5f5c078076eda89bd618cd773aa0ad6"
)
CANONICAL_NATIVE_WORKER_SOURCE_SHA256 = (
    "710fa7390fe26951867ac204a48626610b1a5dca23b1e14ed479f0cd92079830"
)

_EXPECTED_RELATIVE_PATH = "pdq/cpp/reg_test/expected/out"
_BRIDGE_RELATIVE_PATHS = PDQ_REGRESSION_ASSET_PATHS[1:]
_BRIDGE_HASHES = (
    "d8f8f0cee0f4a84f0637022a078f67f0b36e2ed596621e1d33e6339c4e9c9b22",
    "30a10efdf1c83f429013d48d0ffffc52e34e0e35ada952a9d29605215aa9e5af",
    "2dad5a64b1a142e7d362a09857da895ae63b8c7fc23794b766b319361fc93188",
    "a5f0a457248995e8c9065c275aaa5498b61ba4bdf8fcf80387c32f8b5bfc4f05",
    "d8f80f33e0f417b20e37f5cd028f980fb36ed02a9662c1e233e64c634e9c64dd",
    "2da9259bb1a1bd1a5362576552da32a5e63b7380c2774b4866b346c91b89ce77",
    "f0a1e10271ccc0bd90530b720fff038de34ef1e8ada9a956d6967ade5ea91a50",
    "2df05aa8a4896a17c14682da5aaaab07b61b5b42f8fc07fc87c3d0741bfcb0fa",
)
_D4_HASHES = (
    "d8f8f0cee0f4a84f0637022a078f67f0b36e2ed596621e1d33e6339c4e9c9b22",
    "f0a10efd51c83d429053d48d0fffbc52e34e0e17ada956a9d29685211ea9e58f",
    "adad5a64b5a102e55b62a88052dacd5ae63b847fc337b4b766b399361bc93188",
    "a5f0b457248995e8c1065c275aaa56d8b61ba4bdf8fcfc0383c32f8b0bfc4f05",
    "d8f80f31e0f457b00637f5d5038f980fb36ed12a9662c1e233e64c634e9c64dd",
    "8dada49991a1bd1a5362577742da32a5e63b7b80c2364b4866b346c91bc9ce77",
    "f0a1e10271ccc0bd90530b720fff038de34ef1e8ada9a956d6967ade5ea91a50",
    "a5f44ba8a4996a17cd06a1d85aaaa927b61b5b42f8fc03fc83c3d0740bfcb0fa",
)
_BRIDGE_ENCODED_SHA256 = (
    "b5b0799616df52d475a3968dc7e54f1d0724c912244ffa6175bc786375dd7298",
    "a4ec0b39cf38469e3c5e16869a5ea1f23995b42a4b5de15d223e905b87788317",
    "fc9073d9b08f7b4d90e0fbdb4693fa7c58a9746dfc4c339fce5b3320905a3b3b",
    "ed7769b97c8eab168eefe5cc5a49a1372d25cfecb354e457153a8e45f208daa5",
    "3069d9eb8d9de760229cf99ddcc5a7834f08e3924b37c080c2df8bc5836737fb",
    "c2225164ee53418ec001fa33bf9a4cf95a73514f493c87f676a1af9f9624d19b",
    "355dd5a4ccef6e205ed705c73bce8676981f9f943a06453a2f68d9d8bba3f7f1",
    "092bfe6026946f3b9322662d010fb16dabf3da7ece9f0cbaad99772fdf16a30e",
)
_EXPECTED_BLOCK = (
    "d8f8f0cee0f4a84f0637022a078f67f0b36e2ed596621e1d33e6339c4e9c9b22,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-1-original.jpg\n"
    "30a10efdf1c83f429013d48d0ffffc52e34e0e35ada952a9d29605215aa9e5af,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-2-rotate-90.jpg\n"
    "2dad5a64b1a142e7d362a09857da895ae63b8c7fc23794b766b319361fc93188,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-3-rotate-180.jpg\n"
    "a5f0a457248995e8c9065c275aaa5498b61ba4bdf8fcf80387c32f8b5bfc4f05,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-4-rotate-270.jpg\n"
    "d8f80f33e0f417b20e37f5cd028f980fb36ed02a9662c1e233e64c634e9c64dd,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-5-flipx.jpg\n"
    "2da9259bb1a1bd1a5362576552da32a5e63b7380c2774b4866b346c91b89ce77,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-6-flipy.jpg\n"
    "f0a1e10271ccc0bd90530b720fff038de34ef1e8ada9a956d6967ade5ea91a50,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-7-flip-plus-1.jpg\n"
    "2df05aa8a4896a17c14682da5aaaab07b61b5b42f8fc07fc87c3d0741bfcb0fa,100,"
    "./reg_test/../../data/reg-test-input/dih/bridge-8-flip-minus-1.jpg"
).encode("ascii")
_D4_ACROSS_LINE = (
    ",".join(_D4_HASHES)
    + ",100,./reg_test/../../data/reg-test-input/dih/bridge-1-original.jpg"
).encode("ascii")
_INTERPRETATION = (
    "FIXED_COMMIT_OFFICIAL_REGRESSION_ONLY_NOT_CORPUS_THRESHOLD_RECALL_"
    "PERFORMANCE_OR_DUPLICATE_DECISION_ADMISSION"
)
_HEX64 = re.compile(r"[0-9a-f]{64}\Z")


@dataclass(frozen=True, slots=True)
class PDQOfficialBridgeResult:
    relative_path: str
    encoded_sha256: str
    width: int
    height: int
    original_hash: str
    quality: int

    def __post_init__(self) -> None:
        if self.relative_path not in _BRIDGE_RELATIVE_PATHS:
            raise ValueError("official PDQ bridge path differs")
        _require_sha256(self.encoded_sha256, "bridge encoded SHA-256")
        _require_sha256(self.original_hash, "bridge PDQ hash")
        for value, name in ((self.width, "width"), (self.height, "height")):
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"official PDQ bridge {name} differs")
        if self.quality != 100:
            raise ValueError("official PDQ bridge quality differs")

    def to_dict(self) -> dict[str, str | int]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PDQOfficialBridgeResult":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("official PDQ bridge result fields differ")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class PDQOfficialRegressionReceipt:
    regression_bundle_sha256: str
    regression_source_contract_sha256: str
    regression_source_receipt_sha256: str
    regression_retained_aggregate_sha256: str
    expected_output_sha256: str
    native_build_receipt_sha256: str
    native_binary_sha256: str
    native_worker_source_sha256: str
    decoder_name: str
    decoder_version: str
    jpeg_decoder_name: str
    jpeg_decoder_version: str
    bridge_results: tuple[PDQOfficialBridgeResult, ...]
    d4_orientation_order: tuple[str, ...]
    d4_hashes: tuple[str, ...]
    d4_quality: int
    tool_provenance_sha256: str
    decision: str
    interpretation: str = _INTERPRETATION
    schema_version: str = "cvi.pdq_official_regression_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.pdq_official_regression_receipt.v1":
            raise ValueError("unsupported official PDQ regression receipt")
        canonical = (
            self.regression_bundle_sha256,
            self.regression_source_contract_sha256,
            self.regression_source_receipt_sha256,
            self.regression_retained_aggregate_sha256,
            self.expected_output_sha256,
            self.native_build_receipt_sha256,
            self.native_binary_sha256,
            self.native_worker_source_sha256,
        )
        expected = (
            CANONICAL_REGRESSION_BUNDLE_SHA256,
            CANONICAL_REGRESSION_CONTRACT_SHA256,
            CANONICAL_REGRESSION_RECEIPT_SHA256,
            CANONICAL_REGRESSION_AGGREGATE_SHA256,
            CANONICAL_EXPECTED_OUTPUT_SHA256,
            CANONICAL_NATIVE_BUILD_RECEIPT_SHA256,
            CANONICAL_NATIVE_BINARY_SHA256,
            CANONICAL_NATIVE_WORKER_SOURCE_SHA256,
        )
        if canonical != expected:
            raise ValueError("official PDQ canonical evidence binding differs")
        _require_sha256(self.tool_provenance_sha256, "tool provenance SHA-256")
        if (self.decoder_name, self.jpeg_decoder_name) != (
            "Pillow",
            "libjpeg-turbo",
        ):
            raise ValueError("official PDQ decoder identity differs")
        for value in (self.decoder_version, self.jpeg_decoder_version):
            if not isinstance(value, str) or not value or len(value) > 128:
                raise ValueError("official PDQ decoder version differs")
        if len(self.bridge_results) != 8:
            raise ValueError("official PDQ regression requires eight bridges")
        if tuple(item.relative_path for item in self.bridge_results) != (
            _BRIDGE_RELATIVE_PATHS
        ):
            raise ValueError("official PDQ bridge result order differs")
        observed = tuple(
            (item.encoded_sha256, item.original_hash, item.quality)
            for item in self.bridge_results
        )
        if observed != tuple(zip(_BRIDGE_ENCODED_SHA256, _BRIDGE_HASHES, (100,) * 8)):
            raise ValueError("official PDQ bridge results differ")
        if self.d4_orientation_order != PDQ_D4_ORIENTATIONS:
            raise ValueError("official PDQ D4 orientation order differs")
        if self.d4_hashes != _D4_HASHES or self.d4_quality != 100:
            raise ValueError("official PDQ D4 result differs")
        if self.decision != "PASS_EXACT_FIXED_COMMIT_OFFICIAL_REGRESSION":
            raise ValueError("official PDQ regression decision differs")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("official PDQ regression interpretation differs")

    @property
    def receipt_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "regression_bundle_sha256": self.regression_bundle_sha256,
            "regression_source_contract_sha256": self.regression_source_contract_sha256,
            "regression_source_receipt_sha256": self.regression_source_receipt_sha256,
            "regression_retained_aggregate_sha256": self.regression_retained_aggregate_sha256,
            "expected_output_sha256": self.expected_output_sha256,
            "native_build_receipt_sha256": self.native_build_receipt_sha256,
            "native_binary_sha256": self.native_binary_sha256,
            "native_worker_source_sha256": self.native_worker_source_sha256,
            "decoder_name": self.decoder_name,
            "decoder_version": self.decoder_version,
            "jpeg_decoder_name": self.jpeg_decoder_name,
            "jpeg_decoder_version": self.jpeg_decoder_version,
            "bridge_results": [item.to_dict() for item in self.bridge_results],
            "d4_orientation_order": list(self.d4_orientation_order),
            "d4_hashes": list(self.d4_hashes),
            "d4_quality": self.d4_quality,
            "tool_provenance_sha256": self.tool_provenance_sha256,
            "decision": self.decision,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PDQOfficialRegressionReceipt":
        if set(payload) != set(cls.__dataclass_fields__):
            raise ValueError("official PDQ regression receipt fields differ")
        values = dict(payload)
        rows = values["bridge_results"]
        if not isinstance(rows, list) or any(not isinstance(row, dict) for row in rows):
            raise TypeError("official PDQ bridge results differ")
        values["bridge_results"] = tuple(
            PDQOfficialBridgeResult.from_dict(row) for row in rows
        )
        for field in ("d4_orientation_order", "d4_hashes"):
            if not isinstance(values[field], list):
                raise TypeError(f"official PDQ {field} differs")
            values[field] = tuple(values[field])
        return cls(**values)


def run_official_pdq_regression(
    *,
    regression_bundle_directory: Path,
    native_worker_directory: Path,
    tool_provenance: dict[str, Any],
) -> PDQOfficialRegressionReceipt:
    """Authenticate and execute the exact fixed-commit bridge regression."""

    if not isinstance(tool_provenance, dict) or not tool_provenance:
        raise ValueError("official PDQ tool provenance must be nonempty")
    tool_sha256 = content_sha256(tool_provenance)
    bundle_path = regression_bundle_directory / "intake-bundle.json"
    bundle_read = read_retained_regular_file(
        bundle_path,
        expected_sha256=CANONICAL_REGRESSION_BUNDLE_SHA256,
        maximum_bytes=1_000_000,
        capture_payload=True,
        subject="official PDQ regression bundle",
    )
    if bundle_read.payload is None:  # pragma: no cover - helper contract
        raise RuntimeError("official PDQ regression bundle payload is absent")
    bundle = json.loads(bundle_read.payload)
    if not isinstance(bundle, dict):
        raise TypeError("official PDQ regression bundle root differs")
    source = PdqSourceContract.from_dict(bundle["source_contract"])
    source_receipt = PdqSourceIntakeReceipt.from_dict(bundle["receipt"])
    validate_pdq_regression_source_contract(source)
    if (
        source.contract_sha256 != CANONICAL_REGRESSION_CONTRACT_SHA256
        or source_receipt.receipt_sha256 != CANONICAL_REGRESSION_RECEIPT_SHA256
        or source_receipt.retained_source_aggregate_sha256
        != CANONICAL_REGRESSION_AGGREGATE_SHA256
        or bundle.get("source_contract_sha256") != source.contract_sha256
        or bundle.get("receipt_sha256") != source_receipt.receipt_sha256
    ):
        raise ValueError("official PDQ regression source bindings differ")
    source_root = regression_bundle_directory / "source"
    _verify_regression_tree(source_root, source_receipt)
    expected_member = next(
        item for item in source_receipt.retained_members
        if item.relative_path == _EXPECTED_RELATIVE_PATH
    )
    expected_read = read_retained_regular_file(
        source_root / PurePosixPath(_EXPECTED_RELATIVE_PATH),
        expected_bytes=expected_member.byte_size,
        expected_sha256=CANONICAL_EXPECTED_OUTPUT_SHA256,
        maximum_bytes=7_000_000,
        capture_payload=True,
        subject="official PDQ expected output",
    )
    if expected_read.payload is None:  # pragma: no cover
        raise RuntimeError("official PDQ expected output payload is absent")
    _verify_expected_output(expected_read.payload)

    build_payload = read_strict_json_object(
        native_worker_directory / "build-receipt.json"
    )
    build_receipt = PdqNativeBuildReceipt.from_dict(build_payload)
    if (
        build_receipt.receipt_sha256 != CANONICAL_NATIVE_BUILD_RECEIPT_SHA256
        or build_receipt.binary_sha256 != CANONICAL_NATIVE_BINARY_SHA256
        or build_receipt.worker_source_sha256 != CANONICAL_NATIVE_WORKER_SOURCE_SHA256
    ):
        raise ValueError("official PDQ native worker binding differs")
    verify_native_pdq_build(native_worker_directory, build_receipt)

    try:
        import PIL
        from PIL import Image, features
    except ImportError as error:  # pragma: no cover - environment dependent
        raise RuntimeError("official PDQ regression requires Pillow") from error
    jpeg_turbo = features.version_feature("libjpeg_turbo")
    if not features.check_feature("libjpeg_turbo") or not jpeg_turbo:
        raise RuntimeError("official PDQ regression requires libjpeg-turbo")

    requests: list[CanonicalRGBRequest] = []
    geometry: list[tuple[int, int]] = []
    for sequence, (relative_path, encoded_sha256) in enumerate(
        zip(_BRIDGE_RELATIVE_PATHS, _BRIDGE_ENCODED_SHA256, strict=True)
    ):
        member = next(
            item for item in source_receipt.retained_members
            if item.relative_path == relative_path
        )
        read = read_retained_regular_file(
            source_root / PurePosixPath(relative_path),
            expected_bytes=member.byte_size,
            expected_sha256=encoded_sha256,
            maximum_bytes=1_000_000,
            capture_payload=True,
            subject="official PDQ bridge JPEG",
        )
        if read.payload is None:  # pragma: no cover
            raise RuntimeError("official PDQ bridge payload is absent")
        with Image.open(io.BytesIO(read.payload)) as image:
            if image.format != "JPEG" or image.mode != "RGB" or getattr(
                image, "n_frames", 1
            ) != 1:
                raise ValueError("official PDQ bridge decoder metadata differs")
            image.load()
            width, height = image.size
            rgb = image.tobytes("raw", "RGB")
        geometry.append((width, height))
        requests.append(CanonicalRGBRequest(
            width=width,
            height=height,
            rgb=rgb,
            request_sequence=sequence,
            request_token=encoded_sha256,
        ))
    results = hash_rgb_batch(
        tuple(requests),
        binary_path=native_worker_directory / build_receipt.binary_filename,
        expected_binary_sha256=build_receipt.binary_sha256,
        timeout_seconds=30.0,
    )
    bridge_results = tuple(
        PDQOfficialBridgeResult(
            relative_path=relative_path,
            encoded_sha256=encoded_sha256,
            width=width,
            height=height,
            original_hash=result.d4_hashes[0],
            quality=result.quality,
        )
        for relative_path, encoded_sha256, (width, height), result in zip(
            _BRIDGE_RELATIVE_PATHS,
            _BRIDGE_ENCODED_SHA256,
            geometry,
            results,
            strict=True,
        )
    )
    first = results[0]
    return PDQOfficialRegressionReceipt(
        regression_bundle_sha256=bundle_read.sha256,
        regression_source_contract_sha256=source.contract_sha256,
        regression_source_receipt_sha256=source_receipt.receipt_sha256,
        regression_retained_aggregate_sha256=(
            source_receipt.retained_source_aggregate_sha256
        ),
        expected_output_sha256=expected_read.sha256,
        native_build_receipt_sha256=build_receipt.receipt_sha256,
        native_binary_sha256=build_receipt.binary_sha256,
        native_worker_source_sha256=build_receipt.worker_source_sha256,
        decoder_name="Pillow",
        decoder_version=PIL.__version__,
        jpeg_decoder_name="libjpeg-turbo",
        jpeg_decoder_version=jpeg_turbo,
        bridge_results=bridge_results,
        d4_orientation_order=PDQ_D4_ORIENTATIONS,
        d4_hashes=first.d4_hashes,
        d4_quality=first.quality,
        tool_provenance_sha256=tool_sha256,
        decision="PASS_EXACT_FIXED_COMMIT_OFFICIAL_REGRESSION",
    )


def publish_official_pdq_regression(
    *,
    receipt: PDQOfficialRegressionReceipt,
    tool_provenance: dict[str, Any],
    output_path: Path,
) -> None:
    if not isinstance(receipt, PDQOfficialRegressionReceipt):
        raise TypeError("official PDQ regression receipt type differs")
    if content_sha256(tool_provenance) != receipt.tool_provenance_sha256:
        raise ValueError("official PDQ tool provenance binding differs")
    write_private_json_bundle(((output_path, {
        "schema_version": "cvi.pdq_official_regression_bundle.v1",
        "receipt": receipt.to_dict(),
        "receipt_sha256": receipt.receipt_sha256,
        "tool_provenance": tool_provenance,
        "tool_provenance_sha256": receipt.tool_provenance_sha256,
    }),))


def _verify_regression_tree(
    source_root: Path,
    receipt: PdqSourceIntakeReceipt,
) -> None:
    expected = {item.relative_path: item for item in receipt.retained_members}
    observed: set[str] = set()
    for path in source_root.rglob("*"):
        if path.is_symlink():
            raise ValueError("official PDQ regression tree contains a symlink")
        if path.is_dir():
            continue
        if not path.is_file():
            raise ValueError("official PDQ regression tree contains a special file")
        relative = path.relative_to(source_root).as_posix()
        member = expected.get(relative)
        if member is None:
            raise ValueError("official PDQ regression tree contains an unexpected file")
        read_retained_regular_file(
            path,
            expected_bytes=member.byte_size,
            expected_sha256=member.content_sha256,
            capture_payload=False,
            subject="official PDQ regression member",
        )
        observed.add(relative)
    if observed != set(expected):
        raise ValueError("official PDQ regression tree membership differs")


def _verify_expected_output(payload: bytes) -> None:
    if payload.count(_EXPECTED_BLOCK) != 1:
        raise ValueError("official PDQ bridge expected block differs")
    if payload.count(_D4_ACROSS_LINE) != 1:
        raise ValueError("official PDQ D4 expected line differs")


def _require_sha256(value: str, name: str) -> None:
    if not isinstance(value, str) or not _HEX64.fullmatch(value):
        raise ValueError(f"{name} differs")
