from __future__ import annotations

import json
import shutil
import stat
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from contracts.contracts import Modality
from data.acquisition import sha256_file
from data.crop_export import CropBox, CropExportPolicy, OracleCropSource
from evaluation.controls.pairing import (
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairScoringRequest,
    PairStratum,
)
from workflows.export_oracle_crops import (
    _read_object,
    _receipt_target,
    _write_private_receipt,
    main,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _construction() -> PairConstructionResult:
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=HASH_B,
        attributes_sha256=HASH_C,
        eligible_query_count=1,
        selected_query_count=1,
        dropped_query_count=0,
        scoring_requests=(
            PairScoringRequest("pair-1", "token-rgb", "token-ir"),
        ),
        artifact_bindings=(
            PairArtifactBinding("token-rgb", "sample-rgb"),
            PairArtifactBinding("token-ir", "sample-ir"),
        ),
        ground_truth=(
            PairGroundTruth(
                "pair-1",
                "dog-1",
                "dog-2",
                "query-session",
                "reference-session",
                PairStratum.RANDOM,
            ),
        ),
        quotas=(),
    )


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


class CropExportToolTests(unittest.TestCase):
    def test_private_receipt_refuses_overwrite(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            target = root / "receipt.json"
            _write_private_receipt(target, {"status": "ok"})
            self.assertEqual(
                stat.S_IMODE(target.stat().st_mode),
                0o600,
            )
            original = target.read_bytes()
            with self.assertRaisesRegex(FileExistsError, "overwrite"):
                _write_private_receipt(target, {"status": "changed"})
            self.assertEqual(target.read_bytes(), original)

    def test_json_reader_rejects_duplicate_keys_and_symlinks(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            duplicate = root / "duplicate.json"
            duplicate.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                _read_object(duplicate)

            valid = root / "valid.json"
            valid.write_text(json.dumps({"a": 1}), encoding="utf-8")
            linked = root / "linked.json"
            linked.symlink_to(valid)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _read_object(linked)

    def test_receipt_target_rejects_existing_symlink(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "existing.json"
            existing.write_text("{}", encoding="utf-8")
            target = root / "receipt.json"
            target.symlink_to(existing)
            with self.assertRaisesRegex(ValueError, "symlink"):
                _receipt_target(target)

    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg tools are required",
    )
    def test_main_exports_authenticated_bundle_and_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = _construction()
            paths = {
                name: root / f"{name}.json"
                for name in (
                    "scoring",
                    "bindings",
                    "truth",
                    "summary",
                    "sources",
                    "policy",
                )
            }
            _write_json(paths["scoring"], pairs.scoring_payload())
            _write_json(paths["bindings"], pairs.artifact_binding_payload())
            _write_json(paths["truth"], pairs.ground_truth_payload())
            _write_json(paths["summary"], pairs.summary_payload())
            source_paths = (
                root / "source-rgb.png",
                root / "source-ir.png",
            )
            for source in source_paths:
                subprocess.run(
                    (
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=size=32x24:rate=1",
                        "-frames:v",
                        "1",
                        str(source),
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            sources = (
                OracleCropSource(
                    "sample-rgb",
                    str(source_paths[0]),
                    sha256_file(source_paths[0]),
                    Modality.RGB,
                    CropBox(0, 0, 16, 16),
                ),
                OracleCropSource(
                    "sample-ir",
                    str(source_paths[1]),
                    sha256_file(source_paths[1]),
                    Modality.IR,
                    CropBox(0, 0, 16, 16),
                ),
            )
            _write_json(
                paths["sources"],
                {
                    "schema_version": "cvi.oracle_crop_sources.v1",
                    "sources": [source.to_dict() for source in sources],
                },
            )
            _write_json(paths["policy"], CropExportPolicy().to_dict())
            output = root / "output"
            output.mkdir()
            receipt = root / "receipt.json"
            argv = (
                "export_oracle_crops.py",
                "--scoring-requests",
                str(paths["scoring"]),
                "--artifact-bindings",
                str(paths["bindings"]),
                "--ground-truth",
                str(paths["truth"]),
                "--pair-summary",
                str(paths["summary"]),
                "--crop-sources",
                str(paths["sources"]),
                "--export-policy",
                str(paths["policy"]),
                "--output-directory",
                str(output),
                "--receipt-output",
                str(receipt),
            )
            with patch("sys.argv", argv), redirect_stdout(StringIO()) as stdout:
                main()
            status = json.loads(stdout.getvalue())
            self.assertEqual(status["status"], "CREATED")
            self.assertEqual(len(tuple(output.iterdir())), 2)
            self.assertEqual(stat.S_IMODE(receipt.stat().st_mode), 0o600)
            receipt_payload = _read_object(receipt)
            self.assertEqual(
                receipt_payload["pair_set_sha256"],
                pairs.result_sha256,
            )


if __name__ == "__main__":
    unittest.main()
