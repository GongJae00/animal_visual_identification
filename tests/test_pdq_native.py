from __future__ import annotations

import hashlib
import copy
import json
import os
import shutil
import struct
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

from cvi.pdq_contracts import PDQ_D4_ORIENTATIONS, PDQFingerprint
from cvi.pdq_native import (
    CANONICAL_INTAKE_BUNDLE_SHA256,
    CANONICAL_RETAINED_AGGREGATE_SHA256,
    CANONICAL_SOURCE_RECEIPT_SHA256,
    MAXIMUM_DIMENSION,
    PDQIO_RESIZE_DIMENSION,
    CanonicalRGBRequest,
    PdqNativeBuildReceipt,
    REQUEST_HEADER,
    REQUEST_MAGIC,
    RESPONSE_MAGIC,
    RESPONSE_PREFIX,
    RESPONSE_BYTES,
    build_native_pdq_worker,
    hash_rgb,
    hash_rgb_batch,
    verify_native_pdq_build,
)
from cvi.source_provenance import build_offline_tool_provenance


SOURCE_BUNDLE = Path(os.environ.get("CVI_PDQ_SOURCE_BUNDLE") or os.devnull)
WORKER_SOURCE = Path(__file__).parents[1] / "native/pdq_worker/main.cpp"
COMPILER = Path("/usr/bin/c++")
NATIVE_AVAILABLE = SOURCE_BUNDLE.is_dir() and COMPILER.exists()


def _builder_provenance() -> dict[str, object]:
    return build_offline_tool_provenance(
        Path(__file__).parents[1] / "tools/build_native_pdq_worker.py"
    )


def _fixture_rgb(width: int, height: int) -> bytes:
    return bytes(
        (x * 13 + y * 41 + channel * 79 + (x * y) % 251) % 256
        for y in range(height)
        for x in range(width)
        for channel in range(3)
    )


def _token(index: int) -> str:
    return f"{index:064x}"


def _rotate_ccw(width: int, height: int, rgb: bytes) -> bytes:
    output = bytearray(len(rgb))
    for y in range(height):
        for x in range(width):
            new_x, new_y = y, width - 1 - x
            source = (y * width + x) * 3
            destination = (new_y * height + new_x) * 3
            output[destination:destination + 3] = rgb[source:source + 3]
    return bytes(output)


def _pdqio_force_512(width: int, height: int, rgb: bytes) -> bytes:
    output = bytearray(PDQIO_RESIZE_DIMENSION * PDQIO_RESIZE_DIMENSION * 3)
    for output_y in range(PDQIO_RESIZE_DIMENSION):
        source_y = output_y * height // PDQIO_RESIZE_DIMENSION
        for output_x in range(PDQIO_RESIZE_DIMENSION):
            source_x = output_x * width // PDQIO_RESIZE_DIMENSION
            source = (source_y * width + source_x) * 3
            destination = (
                output_y * PDQIO_RESIZE_DIMENSION + output_x
            ) * 3
            output[destination:destination + 3] = rgb[source:source + 3]
    return bytes(output)


@unittest.skipUnless(NATIVE_AVAILABLE, "canonical PDQ v3 source/compiler unavailable")
class NativePdqIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.temporary = TemporaryDirectory()
        cls.root = Path(cls.temporary.name) / "build"
        cls.receipt, cls.strategy = build_native_pdq_worker(
            source_bundle_directory=SOURCE_BUNDLE,
            worker_source=WORKER_SOURCE,
            output_directory=cls.root,
            compiler=COMPILER,
            builder_tool_provenance=_builder_provenance(),
        )
        cls.binary = cls.root / "pdq-native-worker"

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_build_receipt_is_canonical_roundtrippable_and_no_overwrite(self) -> None:
        receipt = self.receipt
        self.assertEqual(receipt.intake_bundle_sha256, CANONICAL_INTAKE_BUNDLE_SHA256)
        self.assertEqual(receipt.source_receipt_sha256, CANONICAL_SOURCE_RECEIPT_SHA256)
        self.assertEqual(
            receipt.retained_source_aggregate_sha256,
            CANONICAL_RETAINED_AGGREGATE_SHA256,
        )
        self.assertEqual(
            PdqNativeBuildReceipt.from_dict(receipt.to_dict()), receipt
        )
        self.assertEqual(receipt.compiler_flags.count("-O2"), 1)
        self.assertNotIn("-march=native", receipt.compiler_flags)
        self.assertNotIn("-ffast-math", receipt.compiler_flags)
        self.assertIn(self.strategy, {
            "RENAMEAT2_NOREPLACE", "RESERVED_EMPTY_DIRECTORY_RENAME",
        })
        verify_native_pdq_build(self.root, receipt)
        with self.assertRaises(FileExistsError):
            build_native_pdq_worker(
                source_bundle_directory=SOURCE_BUNDLE,
                worker_source=WORKER_SOURCE,
                output_directory=self.root,
                compiler=COMPILER,
                builder_tool_provenance=_builder_provenance(),
            )

    def test_build_receipt_rejects_self_consistent_provenance_tampering(self) -> None:
        base = self.receipt.to_dict()
        mutations = {
            "intake bundle": ("intake_bundle_sha256", "0" * 64),
            "source contract": ("source_contract_sha256", "0" * 64),
            "source receipt": ("source_receipt_sha256", "0" * 64),
            "source tool": ("source_tool_provenance_sha256", "0" * 64),
            "source aggregate": ("retained_source_aggregate_sha256", "0" * 64),
            "builder provenance": ("builder_tool_provenance_sha256", "0" * 64),
            "builder manifest": ("builder_code_source_manifest_sha256", "0" * 64),
            "builder runtime": ("builder_runtime_sha256", "0" * 64),
            "worker source": ("worker_source_sha256", "0" * 64),
            "compiler path": ("compiler_realpath", "/tmp/c++"),
            "compiler hash": ("compiler_sha256", "0" * 64),
            "compiler version": ("compiler_version_first_line", "forged"),
            "binary hash": ("binary_sha256", "0" * 64),
            "binary filename": ("binary_filename", "other"),
            "pdqio resize": ("pdqio_resize_dimension", 511),
            "oversize preprocessing": ("oversize_preprocessing", "other"),
            "interpretation": ("interpretation", "ADMITTED"),
        }
        for name, (key, value) in mutations.items():
            payload = copy.deepcopy(base)
            payload[key] = value
            with self.subTest(name=name), self.assertRaises(ValueError):
                PdqNativeBuildReceipt.from_dict(payload)
        provenance_mutations = {
            "provenance extra field": lambda value: value.__setitem__("extra", True),
            "provenance schema": lambda value: value.__setitem__("schema_version", "other"),
            "manifest digest": lambda value: value.__setitem__("code_source_manifest_sha256", "0" * 64),
            "runtime digest": lambda value: value.__setitem__("runtime_sha256", "0" * 64),
            "runtime extra": lambda value: value["runtime"].__setitem__("extra", "x"),
            "row hash": lambda value: value["code_source_files"][0].__setitem__("content_sha256", "0" * 64),
            "row bytes": lambda value: value["code_source_files"][0].__setitem__("byte_size", -1),
            "row order": lambda value: value["code_source_files"].reverse(),
            "extra source namespace": lambda value: value["code_source_files"].append(
                {
                    "relative_path": "other/source.py",
                    "content_sha256": "0" * 64,
                    "byte_size": 1,
                }
            ),
            "missing CLI": lambda value: value.__setitem__(
                "code_source_files",
                [row for row in value["code_source_files"] if row["relative_path"] != "tools/build_native_pdq_worker.py"],
            ),
        }
        for name, mutate in provenance_mutations.items():
            payload = copy.deepcopy(base)
            mutate(payload["builder_tool_provenance"])
            with self.subTest(name=name), self.assertRaises((TypeError, ValueError)):
                PdqNativeBuildReceipt.from_dict(payload)
        for name, mutate in {
            "retained hash": lambda rows: rows[0].__setitem__(2, "0" * 64),
            "retained size": lambda rows: rows[0].__setitem__(1, 1),
            "retained path": lambda rows: rows[0].__setitem__(0, "other"),
            "retained order": lambda rows: rows.reverse(),
        }.items():
            payload = copy.deepcopy(base)
            mutate(payload["retained_members"])
            with self.subTest(name=name), self.assertRaises(ValueError):
                PdqNativeBuildReceipt.from_dict(payload)

    def test_build_rejects_mid_build_repository_provenance_change(self) -> None:
        provenance = _builder_provenance()
        changed = copy.deepcopy(provenance)
        changed["runtime"]["platform_release"] += "-changed"
        changed["runtime_sha256"] = hashlib.sha256(b"not-the-runtime").hexdigest()
        with TemporaryDirectory() as temporary, mock.patch(
            "cvi.pdq_native._current_builder_tool_provenance",
            side_effect=[provenance, changed],
        ):
            output = Path(temporary) / "build"
            with self.assertRaisesRegex(RuntimeError, "changed during build"):
                build_native_pdq_worker(
                    source_bundle_directory=SOURCE_BUNDLE,
                    worker_source=WORKER_SOURCE,
                    output_directory=output,
                    compiler=COMPILER,
                    builder_tool_provenance=provenance,
                )
            self.assertFalse(output.exists())

    def test_deterministic_batch_and_d4_metamorphic_set(self) -> None:
        width, height = 37, 29
        original_rgb = _fixture_rgb(width, height)
        rotated_rgb = _rotate_ccw(width, height, original_rgb)
        requests = (
            CanonicalRGBRequest(width, height, original_rgb, 100, _token(100)),
            CanonicalRGBRequest(height, width, rotated_rgb, 101, _token(101)),
            CanonicalRGBRequest(width, height, original_rgb, 102, _token(102)),
        )
        results = hash_rgb_batch(
            requests,
            binary_path=self.binary,
            expected_binary_sha256=self.receipt.binary_sha256,
        )
        self.assertEqual(results[0].d4_hashes, results[2].d4_hashes)
        self.assertEqual(results[0].quality, results[2].quality)
        self.assertEqual(results[0].request_sequence, 100)
        self.assertEqual(results[0].request_token, _token(100))
        self.assertEqual(results[0].quality, results[1].quality)
        self.assertEqual(set(results[0].d4_hashes), set(results[1].d4_hashes))
        self.assertEqual(results[0].d4_hashes[1], results[1].d4_hashes[0])
        self.assertEqual(len(results[0].d4_hashes), len(PDQ_D4_ORIENTATIONS))
        fingerprint = PDQFingerprint("a" * 64, results[0].d4_hashes, results[0].quality)
        self.assertEqual(fingerprint.d4_hashes, results[0].d4_hashes)

    def test_pdqio_oversize_mapping_and_512_boundary(self) -> None:
        oversize_width, oversize_height = 513, 7
        oversize_rgb = _fixture_rgb(oversize_width, oversize_height)
        explicit_oversize = _pdqio_force_512(
            oversize_width, oversize_height, oversize_rgb
        )
        boundary_width, boundary_height = 512, 7
        boundary_rgb = _fixture_rgb(boundary_width, boundary_height)
        hypothetical_boundary_resize = _pdqio_force_512(
            boundary_width, boundary_height, boundary_rgb
        )
        results = hash_rgb_batch(
            (
                CanonicalRGBRequest(
                    oversize_width, oversize_height, oversize_rgb,
                    200, _token(200),
                ),
                CanonicalRGBRequest(
                    512, 512, explicit_oversize, 201, _token(201),
                ),
                CanonicalRGBRequest(
                    boundary_width, boundary_height, boundary_rgb,
                    202, _token(202),
                ),
                CanonicalRGBRequest(
                    512, 512, hypothetical_boundary_resize, 203, _token(203),
                ),
            ),
            binary_path=self.binary,
            expected_binary_sha256=self.receipt.binary_sha256,
        )
        self.assertEqual(
            (results[0].d4_hashes, results[0].quality),
            (results[1].d4_hashes, results[1].quality),
        )
        self.assertNotEqual(
            (results[2].d4_hashes, results[2].quality),
            (results[3].d4_hashes, results[3].quality),
        )

    def test_worker_rejects_truncation_overflow_and_trailing_input(self) -> None:
        valid_rgb = _fixture_rgb(7, 5)
        valid = REQUEST_HEADER.pack(
            REQUEST_MAGIC, 2, 7, 7, 5, len(valid_rgb), bytes.fromhex(_token(7))
        ) + valid_rgb
        malformed = (
            b"partial",
            REQUEST_HEADER.pack(REQUEST_MAGIC, 2, 7, MAXIMUM_DIMENSION + 1, 1, 0, bytes(32)),
            REQUEST_HEADER.pack(REQUEST_MAGIC, 2, 7, 7, 5, len(valid_rgb) + 1, bytes(32))
            + valid_rgb,
            valid + b"trailing",
        )
        for payload in malformed:
            with self.subTest(bytes=len(payload)):
                completed = subprocess.run(
                    (str(self.binary),), input=payload, capture_output=True,
                    timeout=5.0, shell=False,
                )
                self.assertNotEqual(completed.returncode, 0)
                self.assertLessEqual(len(completed.stderr), 256)
        trailing = subprocess.run(
            (str(self.binary),), input=valid + valid + b"x", capture_output=True,
            timeout=5.0, shell=False,
        )
        self.assertEqual(len(trailing.stdout), 2 * RESPONSE_BYTES)

    def test_client_binds_binary_and_rejects_worker_failures(self) -> None:
        request = CanonicalRGBRequest(7, 5, _fixture_rgb(7, 5), 77, _token(77))
        with self.assertRaisesRegex(ValueError, "differs from expected hash"):
            hash_rgb(
                request, binary_path=self.binary,
                expected_binary_sha256="0" * 64,
            )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            cases = {
                "stderr": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    "sys.stderr.write('bad')\n",
                    RuntimeError,
                ),
                "extra": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.buffer.write(b'x'*{RESPONSE_BYTES + 1})\n",
                    RuntimeError,
                ),
                "timeout": (
                    "#!/usr/bin/python3\nimport time\ntime.sleep(30)\n",
                    TimeoutError,
                ),
            }
            for name, (source, error_type) in cases.items():
                path = root / name
                path.write_text(source, encoding="utf-8")
                path.chmod(0o500)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                with self.subTest(name=name), self.assertRaises(error_type):
                    hash_rgb(
                        request, binary_path=path,
                        expected_binary_sha256=digest,
                        timeout_seconds=0.1 if name == "timeout" else 2.0,
                    )

    def test_batch_is_atomic_on_partial_crash_and_sequence_mismatch(self) -> None:
        requests = (
            CanonicalRGBRequest(7, 5, _fixture_rgb(7, 5), 900, _token(900)),
            CanonicalRGBRequest(9, 6, _fixture_rgb(9, 6), 901, _token(901)),
        )
        valid_first = (
            RESPONSE_PREFIX.pack(RESPONSE_MAGIC, 2, 0, 0, 900, bytes.fromhex(_token(900))) + bytes(256)
        )
        wrong_sequence = (
            RESPONSE_PREFIX.pack(RESPONSE_MAGIC, 2, 0, 0, 999, bytes.fromhex(_token(900))) + bytes(256)
        )
        wrong_token = (
            RESPONSE_PREFIX.pack(RESPONSE_MAGIC, 2, 0, 0, 900, bytes.fromhex(_token(999))) + bytes(256)
        )
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            scripts = {
                "middle-crash": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.buffer.write(bytes.fromhex('{valid_first.hex()}'))\n"
                    "raise SystemExit(3)\n"
                ),
                "partial-response": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.buffer.write(b'x'*{RESPONSE_BYTES - 1})\n"
                ),
                "wrong-sequence": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.buffer.write(bytes.fromhex('{wrong_sequence.hex()}'))\n"
                ),
                "wrong-token": (
                    "#!/usr/bin/python3\nimport sys\nsys.stdin.buffer.read()\n"
                    f"sys.stdout.buffer.write(bytes.fromhex('{wrong_token.hex()}'))\n"
                ),
            }
            for name, source in scripts.items():
                path = root / name
                path.write_text(source, encoding="utf-8")
                path.chmod(0o500)
                digest = hashlib.sha256(path.read_bytes()).hexdigest()
                current_requests = (
                    requests
                    if name in {"middle-crash", "partial-response"}
                    else requests[:1]
                )
                with self.subTest(name=name), self.assertRaises(RuntimeError):
                    hash_rgb_batch(
                        current_requests,
                        binary_path=path,
                        expected_binary_sha256=digest,
                        timeout_seconds=2.0,
                    )


class NativePdqContractTests(unittest.TestCase):
    def test_rgb_geometry_and_receipt_shape_fail_closed(self) -> None:
        with self.assertRaisesRegex(ValueError, "payload length"):
            CanonicalRGBRequest(2, 2, b"x", 1, _token(1))
        with self.assertRaisesRegex(ValueError, "outside the fixed bound"):
            CanonicalRGBRequest(MAXIMUM_DIMENSION + 1, 1, b"", 1, _token(1))
        with self.assertRaises(TypeError):
            CanonicalRGBRequest(1, 1, bytearray(3), 1, _token(1))  # type: ignore[arg-type]
        row = CanonicalRGBRequest(1, 1, bytes(3), 4, _token(4))
        with self.assertRaisesRegex(ValueError, "must be unique"):
            hash_rgb_batch(
                (row, row), binary_path=Path("/does/not/matter"),
                expected_binary_sha256="0" * 64,
            )
        with self.assertRaisesRegex(ValueError, "tokens must be unique"):
            hash_rgb_batch(
                (
                    row,
                    CanonicalRGBRequest(1, 1, bytes(3), 5, _token(4)),
                ),
                binary_path=Path("/does/not/matter"),
                expected_binary_sha256="0" * 64,
            )
        payload = self._minimal_receipt_payload()
        payload["unknown"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            PdqNativeBuildReceipt.from_dict(payload)

    @staticmethod
    def _minimal_receipt_payload() -> dict[str, object]:
        if NATIVE_AVAILABLE:
            with TemporaryDirectory() as temporary:
                receipt, _ = build_native_pdq_worker(
                    source_bundle_directory=SOURCE_BUNDLE,
                    worker_source=WORKER_SOURCE,
                    output_directory=Path(temporary) / "build",
                    compiler=COMPILER,
                    builder_tool_provenance=_builder_provenance(),
                )
                return receipt.to_dict()
        return {field: None for field in PdqNativeBuildReceipt.__dataclass_fields__}


if __name__ == "__main__":
    unittest.main()
