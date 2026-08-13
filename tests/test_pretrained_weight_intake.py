from __future__ import annotations

import hashlib
import json
import os
import pickle
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from contracts.pretrained_weight_intake import (
    PretrainedWeightChecksumAuthority,
    PretrainedWeightFileFormat,
    PretrainedWeightIntakeReceipt,
    PretrainedWeightSourceContract,
    PretrainedWeightUsageLane,
    audit_pretrained_weight_file,
    validate_pretrained_weight_receipt_binding,
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class _AdversarialPickle:
    def __init__(self, sentinel: Path) -> None:
        self.sentinel = sentinel

    def __reduce__(self) -> tuple[object, tuple[str]]:
        return os.system, (f"touch {self.sentinel}",)


class PretrainedWeightIntakeTests(unittest.TestCase):
    def test_cli_help_executes_without_loading_a_tensor_framework(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(
                    Path(__file__).parents[1]
                    / "workflows"
                    / "audit_pretrained_weight.py"
                ),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--training-snapshot", completed.stdout)

    def _fixture(
        self,
        root: Path,
        *,
        file_format: PretrainedWeightFileFormat = (
            PretrainedWeightFileFormat.SAFETENSORS
        ),
        license_lane: PretrainedWeightUsageLane = (
            PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ),
        target_lane: PretrainedWeightUsageLane = (
            PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE
        ),
        checksum_authority: PretrainedWeightChecksumAuthority = (
            PretrainedWeightChecksumAuthority.PUBLISHED_SHA256
        ),
        payload: bytes = b"synthetic tensor container bytes",
    ) -> tuple[
        Path,
        Path,
        Path,
        PretrainedWeightSourceContract,
    ]:
        root.mkdir(parents=True, exist_ok=True)
        filename = (
            "weights.safetensors"
            if file_format is PretrainedWeightFileFormat.SAFETENSORS
            else "weights.pth"
        )
        weight = root / filename
        license_snapshot = root / "LICENSE.snapshot"
        training_snapshot = root / "TRAINING.snapshot"
        weight.write_bytes(payload)
        license_snapshot.write_text("synthetic license evidence", encoding="utf-8")
        training_snapshot.write_text(
            "synthetic training-data description evidence",
            encoding="utf-8",
        )
        source = PretrainedWeightSourceContract(
            source_model_id="fixture/model-small",
            source_revision="0123456789abcdef",
            source_model_page_url="https://example.org/models/model-small",
            source_file_url=f"https://example.org/files/{filename}",
            weight_filename=filename,
            license_id="FIXTURE-PERMISSIVE",
            license_url="https://example.org/licenses/fixture",
            license_snapshot_sha256=_sha256(license_snapshot),
            license_usage_lane=license_lane,
            training_description="Synthetic, non-model test bytes only.",
            training_description_url="https://example.org/models/model-small/data",
            training_description_snapshot_sha256=_sha256(training_snapshot),
            expected_file_bytes=weight.stat().st_size,
            expected_sha256=_sha256(weight),
            checksum_authority=checksum_authority,
            target_lane=target_lane,
            file_format=file_format,
        )
        return weight, license_snapshot, training_snapshot, source

    def test_published_safetensors_bytes_pass_and_round_trip(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            receipt = audit_pretrained_weight_file(
                weight_path=fixture[0],
                license_snapshot_path=fixture[1],
                training_description_snapshot_path=fixture[2],
                source=fixture[3],
            )
            self.assertEqual(
                receipt.decision,
                "PASS_PUBLISHED_SHA256_DEPLOYMENT_CANDIDATE",
            )
            self.assertEqual(receipt.weight_sha256, fixture[3].expected_sha256)
            self.assertEqual(
                PretrainedWeightSourceContract.from_dict(fixture[3].to_dict()),
                fixture[3],
            )
            self.assertEqual(
                PretrainedWeightIntakeReceipt.from_dict(receipt.to_dict()),
                receipt,
            )
            validate_pretrained_weight_receipt_binding(receipt, fixture[3])

    def test_pytorch_state_dict_claim_is_never_deserialized(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentinel = root / "DESERIALIZATION_OCCURRED"
            payload = pickle.dumps(_AdversarialPickle(sentinel))
            fixture = self._fixture(
                root,
                file_format=PretrainedWeightFileFormat.PYTORCH_STATE_DICT,
                target_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
                payload=payload,
            )
            receipt = audit_pretrained_weight_file(
                weight_path=fixture[0],
                license_snapshot_path=fixture[1],
                training_description_snapshot_path=fixture[2],
                source=fixture[3],
            )
            self.assertFalse(sentinel.exists())
            self.assertEqual(
                receipt.interpretation,
                "WEIGHT_BYTE_INTAKE_ONLY_NOT_DESERIALIZATION_MODEL_OR_PERFORMANCE_ADMISSION",
            )

    def test_research_license_and_unverified_checksum_cannot_target_deployment(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(
                Path(temporary),
                target_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
            )
            with self.assertRaisesRegex(ValueError, "research-only license"):
                replace(
                    fixture[3],
                    license_usage_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
                    target_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
                )
            with self.assertRaisesRegex(ValueError, "unverified checksum"):
                replace(
                    fixture[3],
                    checksum_authority=(
                        PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256
                    ),
                    target_lane=PretrainedWeightUsageLane.DEPLOYMENT_CANDIDATE,
                )

            research_source = replace(
                fixture[3],
                license_usage_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
                checksum_authority=(
                    PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256
                ),
                target_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
            )
            receipt = audit_pretrained_weight_file(
                weight_path=fixture[0],
                license_snapshot_path=fixture[1],
                training_description_snapshot_path=fixture[2],
                source=research_source,
            )
            self.assertEqual(
                receipt.decision,
                "PASS_UNVERIFIED_SHA256_RESEARCH_ONLY",
            )

    def test_exact_size_hash_and_snapshot_bindings_are_required(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            cases = (
                (replace(fixture[3], expected_file_bytes=1), "byte size"),
                (replace(fixture[3], expected_sha256="0" * 64), "SHA-256"),
                (
                    replace(fixture[3], license_snapshot_sha256="0" * 64),
                    "license snapshot",
                ),
                (
                    replace(
                        fixture[3],
                        training_description_snapshot_sha256="0" * 64,
                    ),
                    "training description snapshot",
                ),
            )
            for source, message in cases:
                with self.subTest(message=message), self.assertRaisesRegex(
                    ValueError,
                    message,
                ):
                    audit_pretrained_weight_file(
                        weight_path=fixture[0],
                        license_snapshot_path=fixture[1],
                        training_description_snapshot_path=fixture[2],
                        source=source,
                    )

    def test_symlink_mutation_and_path_replacement_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            link = root / "link.safetensors"
            link.symlink_to(fixture[0])
            linked_source = replace(
                fixture[3],
                weight_filename=link.name,
                source_file_url="https://example.org/files/link.safetensors",
            )
            with self.assertRaises(OSError):
                audit_pretrained_weight_file(
                    weight_path=link,
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=linked_source,
                )

            fifo = root / "fifo.safetensors"
            os.mkfifo(fifo)
            fifo_source = replace(
                fixture[3],
                weight_filename=fifo.name,
                source_file_url="https://example.org/files/fifo.safetensors",
                expected_file_bytes=1,
                expected_sha256="0" * 64,
            )
            with self.assertRaisesRegex(ValueError, "regular file"):
                audit_pretrained_weight_file(
                    weight_path=fifo,
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=fifo_source,
                )

        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))

            def mutate(_: str) -> None:
                fixture[0].write_bytes(b"X" * fixture[3].expected_file_bytes)

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pretrained_weight_file(
                    weight_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=mutate,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            replacement = root / "replacement.safetensors"
            replacement.write_bytes(b"Y" * fixture[3].expected_file_bytes)

            def replace_path(_: str) -> None:
                os.replace(replacement, fixture[0])

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pretrained_weight_file(
                    weight_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=replace_path,
                )

    def test_replace_then_restore_aba_and_cross_contract_receipt_replay_fail(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            parked = root / "parked.safetensors"
            replacement = root / "replacement.safetensors"
            replacement.write_bytes(b"Z" * fixture[3].expected_file_bytes)

            def replace_then_restore(_: str) -> None:
                os.replace(fixture[0], parked)
                os.replace(replacement, fixture[0])
                os.replace(fixture[0], replacement)
                os.replace(parked, fixture[0])

            with self.assertRaisesRegex(RuntimeError, "changed during intake"):
                audit_pretrained_weight_file(
                    weight_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=replace_then_restore,
                )

        with TemporaryDirectory() as first, TemporaryDirectory() as second:
            fixture_a = self._fixture(Path(first))
            fixture_b = self._fixture(
                Path(second),
                target_lane=PretrainedWeightUsageLane.RESEARCH_ONLY,
                payload=b"different source model bytes",
            )
            receipt = audit_pretrained_weight_file(
                weight_path=fixture_a[0],
                license_snapshot_path=fixture_a[1],
                training_description_snapshot_path=fixture_a[2],
                source=fixture_a[3],
            )
            with self.assertRaisesRegex(ValueError, "source_contract_sha256"):
                validate_pretrained_weight_receipt_binding(receipt, fixture_b[3])

    def test_parent_directory_path_replacement_fails_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            source_root = base / "source"
            fixture = self._fixture(source_root)
            parked = base / "parked"
            replacement = base / "replacement"
            replacement.mkdir()
            (replacement / fixture[0].name).write_bytes(
                b"R" * fixture[3].expected_file_bytes
            )

            def replace_parent(_: str) -> None:
                os.replace(source_root, parked)
                os.replace(replacement, source_root)

            with self.assertRaisesRegex(RuntimeError, "parent"):
                audit_pretrained_weight_file(
                    weight_path=fixture[0],
                    license_snapshot_path=fixture[1],
                    training_description_snapshot_path=fixture[2],
                    source=fixture[3],
                    audit_phase_callback=replace_parent,
                )

    def test_url_filename_format_and_exact_schema_are_enforced(self) -> None:
        with TemporaryDirectory() as temporary:
            fixture = self._fixture(Path(temporary))
            with self.assertRaisesRegex(ValueError, "URL basename"):
                replace(
                    fixture[3],
                    source_file_url="https://example.org/files/other.safetensors",
                )
            with self.assertRaisesRegex(ValueError, "suffix"):
                replace(
                    fixture[3],
                    file_format=PretrainedWeightFileFormat.PYTORCH_STATE_DICT,
                )
            with self.assertRaisesRegex(ValueError, "frozen revision"):
                replace(fixture[3], source_revision="main")
            with self.assertRaisesRegex(TypeError, "target_lane"):
                replace(fixture[3], target_lane="RESEARCH_ONLY")
            payload = fixture[3].to_dict()
            payload["unexpected"] = True
            with self.assertRaisesRegex(ValueError, "fields differ"):
                PretrainedWeightSourceContract.from_dict(payload)

    def test_repository_b2_contract_records_only_observed_checksum_authority(
        self,
    ) -> None:
        contract_path = (
            Path(__file__).parents[1]
            / "contracts"
            / "configs"
            / "pretrained-weights"
            / "torchvision-resnet18-imagenet1k-v1-336d36e8.json"
        )
        contract = PretrainedWeightSourceContract.from_dict(
            json.loads(contract_path.read_text(encoding="utf-8"))
        )

        self.assertEqual(
            contract.source_model_id,
            "torchvision.models.ResNet18_Weights.IMAGENET1K_V1",
        )
        self.assertEqual(
            contract.source_revision,
            "torchvision-v0.26.0@336d36e8db990a905498c73933e35231876e28bc",
        )
        self.assertEqual(
            contract.source_file_url,
            "https://download.pytorch.org/models/resnet18-f37072fd.pth",
        )
        self.assertEqual(contract.weight_filename, "resnet18-f37072fd.pth")
        self.assertEqual(contract.expected_file_bytes, 46_830_571)
        self.assertEqual(
            contract.expected_sha256,
            "f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
        )
        self.assertIs(
            contract.checksum_authority,
            PretrainedWeightChecksumAuthority.UNVERIFIED_SHA256,
        )
        self.assertIs(
            contract.license_usage_lane,
            PretrainedWeightUsageLane.RESEARCH_ONLY,
        )
        self.assertIs(contract.target_lane, PretrainedWeightUsageLane.RESEARCH_ONLY)
        self.assertEqual(contract.license_id, "UNVERIFIED_WEIGHT_LICENSE_SCOPE")
        self.assertIn("official filename hash prefix", contract.training_description)
        self.assertIn("Supervised ImageNet-1K ResNet18", contract.training_description)
        self.assertEqual(
            contract.contract_sha256,
            "d6b36cb256ab2ecf1b16dd13ec8f929ad707c83439146e6313ee201faef04aa6",
        )


if __name__ == "__main__":
    unittest.main()
