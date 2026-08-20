from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace

from enrollment.registry.identity_registry import compute_public_subject_token
from shared.foundation.provenance import content_sha256
from evaluation.splits.split_role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
    verify_declaration_source_links,
    verify_split_role_exposure_inputs,
)

def _token(domain: str, value: str) -> str:
    return hashlib.sha256(f"{domain}\0{value}".encode()).hexdigest()

def _fixture():
    dataset_identity_id = "yt-bb-dog:v1:video-track:1"
    sample = SimpleNamespace(
        sample_token=_token("sample", "one"),
        identity_token=_token("identity", dataset_identity_id),
        dataset_identity_id=dataset_identity_id,
    )
    artifact = {
        "schema_version": "fixture.v1",
        "records": [
            {
                "sample_token": sample.sample_token,
                "identity_token": sample.identity_token,
                "public_subject_token": compute_public_subject_token(
                    dataset_identity_id
                ),
            }
        ],
    }
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=content_sha256(artifact),
        kind=ExposureDeclarationKind.PRIOR_EVALUATION,
        revoked=False,
        records=(
            RoleExposureDeclarationRecord(
                sample_token=sample.sample_token,
                identity_token=sample.identity_token,
                public_subject_token=compute_public_subject_token(
                    dataset_identity_id
                ),
                stage=ExposureStage.CALIBRATION_SCORED,
            ),
        ),
    )
    return sample, artifact, declaration

class SplitRoleExposureTests(unittest.TestCase):
    def test_receipt_and_source_links_are_required(self) -> None:
        sample, artifact, declaration = _fixture()
        verify_declaration_source_links(declaration, artifact)
        ledger = merge_role_exposure_declarations((declaration,))
        receipt = create_role_exposure_receipt(ledger)
        self.assertEqual(
            verify_split_role_exposure_inputs((sample,), ledger, receipt),
            {sample.identity_token: ExposureStage.CALIBRATION_SCORED},
        )

        changed = dict(artifact)
        changed["records"] = []
        with self.assertRaisesRegex(ValueError, "artifact hash differs"):
            verify_declaration_source_links(declaration, changed)

        other = SimpleNamespace(
            sample_token=sample.sample_token,
            identity_token=_token("identity", "other"),
            dataset_identity_id=sample.dataset_identity_id,
        )
        with self.assertRaisesRegex(ValueError, "source links differ"):
            verify_split_role_exposure_inputs((other,), ledger, receipt)

    def test_assembly_cli_refuses_overwrite_before_reading_inputs(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            ledger = root / "ledger.json"
            ledger.write_text(json.dumps({"existing": True}), encoding="utf-8")
            completed = subprocess.run(
                [
                    sys.executable,
                    "evaluation/splits/assemble_role_exposure_ledger.py",
                    "--source-bundle",
                    str(root / "missing-source.json"),
                    "--declaration",
                    str(root / "missing-artifact.json"),
                    str(root / "missing-declaration.json"),
                    "--ledger-output",
                    str(ledger),
                    "--receipt-output",
                    str(root / "receipt.json"),
                ],
                capture_output=True,
                text=True,
            )
            self.assertNotEqual(completed.returncode, 0)
            self.assertIn("refusing to overwrite", completed.stderr)
            self.assertEqual(json.loads(ledger.read_text()), {"existing": True})

if __name__ == "__main__":
    unittest.main()
