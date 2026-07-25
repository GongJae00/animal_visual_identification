from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.pretrained_supporting_asset_intake import (
    MAXIMUM_JSON_ARRAY_LENGTH,
    MAXIMUM_JSON_DEPTH,
    MAXIMUM_JSON_KEYS,
    MAXIMUM_JSON_STRING_CHARACTERS,
    PretrainedSupportingAssetIntakeReceipt,
    PretrainedSupportingAssetKind,
    PretrainedSupportingAssetSourceContract,
    audit_pretrained_supporting_asset,
    parse_bounded_strict_json_object,
    validate_pretrained_supporting_asset_receipt_binding,
)
from cvi.pretrained_weight_intake import (
    PretrainedWeightChecksumAuthority,
    PretrainedWeightFileFormat,
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    PretrainedWeightUsageLane,
    audit_pretrained_weight_file,
)
from cvi.provenance import content_sha256


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class PretrainedSupportingAssetIntakeTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        payload: bytes = b'{"crop_size":{"height":224,"width":224}}\n',
        asset_lane: PretrainedWeightUsageLane = (
            PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ),
        weight_lane: PretrainedWeightUsageLane = (
            PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ),
    ) -> tuple[
        Path,
        Path,
        PretrainedSupportingAssetSourceContract,
        PretrainedWeightSourceContract,
        PretrainedWeightIntakeReceipt,
    ]:
        root.mkdir(parents=True, exist_ok=True)
        asset = root / "preprocessor_config.json.partial"
        weight = root / "model.safetensors"
        license_snapshot = root / "LICENSE.snapshot"
        training_snapshot = root / "TRAINING.snapshot"
        asset.write_bytes(payload)
        weight.write_bytes(b"synthetic non-model bytes")
        license_snapshot.write_text("fixture license", encoding="utf-8")
        training_snapshot.write_text("fixture training lineage", encoding="utf-8")
        weight_source = PretrainedWeightSourceContract(
            source_model_id="fixture/model",
            source_revision="0123456789abcdef",
            source_model_page_url="https://example.org/fixture/model",
            source_file_url="https://example.org/fixture/model/model.safetensors",
            weight_filename="model.safetensors",
            license_id="FIXTURE",
            license_url="https://example.org/license",
            license_snapshot_sha256=_sha256(license_snapshot),
            license_usage_lane=weight_lane,
            training_description="Synthetic test lineage.",
            training_description_url="https://example.org/training",
            training_description_snapshot_sha256=_sha256(training_snapshot),
            expected_file_bytes=weight.stat().st_size,
            expected_sha256=_sha256(weight),
            checksum_authority=(
                PretrainedWeightChecksumAuthority.PUBLISHED_SHA256
            ),
            target_lane=weight_lane,
            file_format=PretrainedWeightFileFormat.SAFETENSORS,
        )
        weight_receipt = audit_pretrained_weight_file(
            weight_path=weight,
            license_snapshot_path=license_snapshot,
            training_description_snapshot_path=training_snapshot,
            source=weight_source,
        )
        source = PretrainedSupportingAssetSourceContract(
            source_model_id=weight_source.source_model_id,
            source_revision=weight_source.source_revision,
            source_model_page_url=weight_source.source_model_page_url,
            source_file_url=(
                "https://example.org/fixture/model/preprocessor_config.json"
            ),
            asset_filename="preprocessor_config.json",
            asset_kind=PretrainedSupportingAssetKind.PREPROCESSOR_CONFIG,
            expected_file_bytes=asset.stat().st_size,
            expected_sha256=_sha256(asset),
            license_id="FIXTURE",
            license_url="https://example.org/license",
            license_snapshot_sha256=_sha256(license_snapshot),
            license_usage_lane=asset_lane,
            associated_pretrained_weight_receipt_sha256=(
                weight_receipt.receipt_sha256
            ),
            target_lane=asset_lane,
        )
        return asset, license_snapshot, source, weight_source, weight_receipt

    def test_partial_asset_passes_and_round_trips_without_model_admission(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            receipt = audit_pretrained_supporting_asset(
                asset_path=fixture[0],
                license_snapshot_path=fixture[1],
                source=fixture[2],
                associated_weight_source=fixture[3],
                associated_weight_receipt=fixture[4],
            )
            self.assertEqual(
                receipt.decision,
                "PASS_EXACT_BYTE_AND_JSON_DEPLOYMENT_CANDIDATE",
            )
            self.assertEqual(
                receipt.interpretation,
                "CONFIG_BYTE_AND_JSON_STRUCTURE_INTAKE_ONLY_NOT_PREPROCESSING_"
                "MODEL_OR_PERFORMANCE_ADMISSION",
            )
            self.assertEqual(
                PretrainedSupportingAssetSourceContract.from_dict(
                    fixture[2].to_dict()
                ),
                fixture[2],
            )
            self.assertEqual(
                PretrainedSupportingAssetIntakeReceipt.from_dict(
                    receipt.to_dict()
                ),
                receipt,
            )
            validate_pretrained_supporting_asset_receipt_binding(
                receipt,
                fixture[2],
            )

    def test_json_parser_rejects_noncanonical_and_unbounded_structures(self) -> None:
        cases = {
            "duplicate JSON object key": b'{"a":1,"a":2}',
            "must be UTF-8": b'{"a":"\xff"}',
            "root must be an object": b"[]",
            "non-standard JSON numeric constant": b'{"a":NaN}',
            "number must be finite": b'{"a":1e999}',
            "depth exceeds": (
                ("[" * MAXIMUM_JSON_DEPTH)
                + "0"
                + ("]" * MAXIMUM_JSON_DEPTH)
            ).join(('{"a":', "}")).encode(),
            "key count exceeds": json.dumps(
                {str(index): 0 for index in range(MAXIMUM_JSON_KEYS + 1)}
            ).encode(),
            "array exceeds": json.dumps(
                {"a": [0] * (MAXIMUM_JSON_ARRAY_LENGTH + 1)}
            ).encode(),
            "string exceeds": json.dumps(
                {"a": "x" * (MAXIMUM_JSON_STRING_CHARACTERS + 1)}
            ).encode(),
            "integer token exceeds": (
                b'{"a":' + (b"9" * 129) + b"}"
            ),
        }
        for message, payload in cases.items():
            with self.subTest(message=message), self.assertRaisesRegex(
                ValueError,
                message,
            ):
                parse_bounded_strict_json_object(payload)

    def test_source_schema_url_name_size_lane_and_receipt_are_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "URL basename"):
                replace(
                    fixture[2],
                    source_file_url="https://example.org/fixture/model/config.json",
                )
            with self.assertRaisesRegex(ValueError, "positive and bounded"):
                replace(fixture[2], expected_file_bytes=4_194_305)
            with self.assertRaisesRegex(ValueError, "research-only license"):
                replace(
                    fixture[2],
                    license_usage_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
                )
            payload = fixture[2].to_dict()
            payload["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "fields differ"):
                PretrainedSupportingAssetSourceContract.from_dict(payload)

            wrong_receipt = replace(
                fixture[2],
                associated_pretrained_weight_receipt_sha256="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "receipt SHA-256"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=wrong_receipt,
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                )

    def test_model_revision_and_weight_lane_mismatch_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "model ID"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=replace(
                        fixture[3], source_model_id="fixture/other"
                    ),
                    associated_weight_receipt=fixture[4],
                )
            with self.assertRaisesRegex(ValueError, "revision"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=replace(
                        fixture[3], source_revision="fedcba9876543210"
                    ),
                    associated_weight_receipt=fixture[4],
                )

        with TemporaryDirectory() as temporary:
            fixture = self._fixture(
                Path(temporary),
                asset_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
                weight_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
            )
            with self.assertRaisesRegex(ValueError, "research-only weight"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                )

    def test_exact_asset_and_license_bytes_are_required(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            cases = (
                (replace(fixture[2], expected_file_bytes=1), "byte size"),
                (replace(fixture[2], expected_sha256="0" * 64), "SHA-256"),
                (
                    replace(fixture[2], license_snapshot_sha256="0" * 64),
                    "license snapshot SHA-256",
                ),
            )
            for source, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    audit_pretrained_supporting_asset(
                        asset_path=fixture[0],
                        license_snapshot_path=fixture[1],
                        source=source,
                        associated_weight_source=fixture[3],
                        associated_weight_receipt=fixture[4],
                    )

    def test_repository_dinov2_preprocessor_contract_fixes_exact_source(self) -> None:
        contract_path = (
            Path(__file__).parents[1]
            / "configs"
            / "pretrained-weights"
            / "dinov2-small-preprocessor-hf-ed25f3a3.json"
        )
        contract = PretrainedSupportingAssetSourceContract.from_dict(
            json.loads(contract_path.read_text(encoding="utf-8"))
        )
        self.assertEqual(
            contract.source_revision,
            "ed25f3a31f01632728cabb09d1542f84ab7b0056",
        )
        self.assertEqual(contract.expected_file_bytes, 436)
        self.assertEqual(
            contract.expected_sha256,
            "14e780d86fa1861f8751f868d7f45425b5feb55c38ca26f152ca5097ab30f828",
        )
        self.assertEqual(
            contract.associated_pretrained_weight_receipt_sha256,
            "5c9ef4247c7b04daf90de2d0b44136145275d5163c5a5cb4961cb206b9a59e92",
        )

    def test_symlink_fifo_file_parent_and_aba_mutation_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            link = root / "preprocessor_config.json"
            link.symlink_to(fixture[0])
            with self.assertRaises(OSError):
                audit_pretrained_supporting_asset(
                    asset_path=link,
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                )
            fifo = root / "config.json"
            os.mkfifo(fifo)
            fifo_source = replace(
                fixture[2],
                asset_filename="config.json",
                asset_kind=PretrainedSupportingAssetKind.CONFIG,
                source_file_url=(
                    "https://example.org/fixture/model/config.json"
                ),
            )
            with self.assertRaisesRegex(ValueError, "regular file"):
                audit_pretrained_supporting_asset(
                    asset_path=fifo,
                    license_snapshot_path=fixture[1],
                    source=fifo_source,
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)

            def mutate(_: str) -> None:
                fixture[0].write_bytes(b"X" * fixture[2].expected_file_bytes)

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                    audit_phase_callback=mutate,
                )

        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            fixture = self._fixture(source_root)
            parked = base / "parked"
            replacement = base / "replacement"
            replacement.mkdir()
            (replacement / fixture[0].name).write_bytes(
                b"R" * fixture[2].expected_file_bytes
            )

            def replace_parent(_: str) -> None:
                os.replace(source_root, parked)
                os.replace(replacement, source_root)

            with self.assertRaisesRegex(RuntimeError, "parent"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                    audit_phase_callback=replace_parent,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            parked = root / "parked.json"
            replacement = root / "replacement.json"
            replacement.write_bytes(b"Z" * fixture[2].expected_file_bytes)

            def replace_then_restore(_: str) -> None:
                os.replace(fixture[0], parked)
                os.replace(replacement, fixture[0])
                os.replace(fixture[0], replacement)
                os.replace(parked, fixture[0])

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pretrained_supporting_asset(
                    asset_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    source=fixture[2],
                    associated_weight_source=fixture[3],
                    associated_weight_receipt=fixture[4],
                    audit_phase_callback=replace_then_restore,
                )

    def test_cli_writes_once_and_help_does_not_require_model_framework(self) -> None:
        tool = (
            Path(__file__).parents[1]
            / "tools"
            / "audit_pretrained_supporting_asset.py"
        )
        help_result = subprocess.run(
            [sys.executable, str(tool), "--help"],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--weight-intake-receipt", help_result.stdout)

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            source_path = root / "asset-contract.json"
            source_path.write_text(
                json.dumps(fixture[2].to_dict()),
                encoding="utf-8",
            )
            provenance = {"fixture": True}
            weight_bundle = {
                "schema_version": "cvi.pretrained_weight_intake_bundle.v1",
                "source_contract_sha256": fixture[3].contract_sha256,
                "source_contract": fixture[3].to_dict(),
                "receipt_sha256": fixture[4].receipt_sha256,
                "receipt": fixture[4].to_dict(),
                "tool_provenance": provenance,
                "tool_provenance_sha256": content_sha256(provenance),
            }
            weight_bundle_path = root / "weight-receipt.json"
            weight_bundle_path.write_text(
                json.dumps(weight_bundle),
                encoding="utf-8",
            )
            output = root / "asset-receipt.json"
            command = [
                sys.executable,
                str(tool),
                "--source-contract",
                str(source_path),
                "--asset",
                str(fixture[0]),
                "--license-snapshot",
                str(fixture[1]),
                "--weight-intake-receipt",
                str(weight_bundle_path),
                "--receipt",
                str(output),
            ]
            first = subprocess.run(
                command,
                check=True,
                capture_output=True,
                text=True,
            )
            self.assertIn("PASS_EXACT_BYTE_AND_JSON", first.stdout)
            second = subprocess.run(command, capture_output=True, text=True)
            self.assertNotEqual(second.returncode, 0)
            self.assertIn("refusing to overwrite", second.stderr)


if __name__ == "__main__":
    unittest.main()
