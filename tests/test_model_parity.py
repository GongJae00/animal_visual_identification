from __future__ import annotations

import json
import tempfile
import unittest
from hashlib import sha256
from pathlib import Path

from cvi.evidence.model_parity import (
    ModelParityError,
    ModelParityReceipt,
    ModelUsageLane,
    ParityFixtureKind,
    ParityFixtureResult,
    ParityThresholds,
    load_model_parity_receipt,
    validate_parity_binding,
)
from tools.export_pretrained_to_onnx import export_dinov2_small


def _receipt(*, lane: ModelUsageLane = ModelUsageLane.RESEARCH_ONLY) -> ModelParityReceipt:
    return ModelParityReceipt(
        model_id="fixture/model",
        artifact_sha256="a" * 64,
        source_weights_sha256="b" * 64,
        weight_intake_receipt_sha256=None,
        preprocessing_sha256="c" * 64,
        preprocessor_intake_receipt_sha256=None,
        usage_lane=lane,
        reference_backend="fixture-reference",
        candidate_backend="fixture-candidate",
        thresholds=ParityThresholds(1e-4, 1e-3, 1e-4, 0.999),
        fixture_panel_receipt_sha256=None,
        fixtures=(
            ParityFixtureResult(
                fixture_id="synthetic-gradient",
                fixture_kind=ParityFixtureKind.SYNTHETIC,
                input_sha256="d" * 64,
                reference_output_sha256="e" * 64,
                candidate_output_sha256="f" * 64,
                maximum_absolute_error=1e-6,
                maximum_relative_error=1e-5,
                cosine_similarity=1.0,
                decision="PASS",
            ),
        ),
        decision="PASS",
    )


class ModelParityTests(unittest.TestCase):
    def test_receipt_requires_exact_keys_pass_and_external_file_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "parity.json"
            payload = _receipt().to_dict()
            path.write_text(
                json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8"
            )
            expected = sha256(path.read_bytes()).hexdigest()
            loaded = load_model_parity_receipt(path, expected_sha256=expected)
            self.assertEqual(loaded, _receipt())

            unknown = dict(payload)
            unknown["undeclared"] = True
            with self.assertRaisesRegex(ModelParityError, "keys mismatch"):
                ModelParityReceipt.from_dict(unknown)
            failed = dict(payload)
            failed["decision"] = "FAIL"
            with self.assertRaisesRegex(ModelParityError, "must be PASS"):
                ModelParityReceipt.from_dict(failed)
            with self.assertRaises(Exception):
                load_model_parity_receipt(path, expected_sha256="0" * 64)

    def test_public_production_rejects_fixture_and_synthetic_only_lanes(self) -> None:
        fixture = _receipt(lane=ModelUsageLane.TEST_FIXTURE)
        with self.assertRaisesRegex(ModelParityError, "TEST_FIXTURE"):
            validate_parity_binding(
                fixture,
                model_id=fixture.model_id,
                artifact_sha256=fixture.artifact_sha256,
                source_weights_sha256=fixture.source_weights_sha256,
                preprocessing_sha256=fixture.preprocessing_sha256,
                usage_lane=fixture.usage_lane,
                public_production=True,
            )
        deployment = _receipt(lane=ModelUsageLane.DEPLOYMENT_CANDIDATE)
        with self.assertRaisesRegex(ModelParityError, "receipt-bound crop"):
            validate_parity_binding(
                deployment,
                model_id=deployment.model_id,
                artifact_sha256=deployment.artifact_sha256,
                source_weights_sha256=deployment.source_weights_sha256,
                preprocessing_sha256=deployment.preprocessing_sha256,
                usage_lane=deployment.usage_lane,
                public_production=True,
            )

    def test_export_refuses_existing_bundle_before_loading_model(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            existing = root / "dino.onnx"
            existing.write_bytes(b"preserve")
            with self.assertRaisesRegex(FileExistsError, "refusing to overwrite"):
                export_dinov2_small(
                    model_directory=root / "missing-model",
                    weight_intake_bundle=root / "missing-weight.json",
                    preprocessor_intake_bundle=root / "missing-processor.json",
                    output_directory=root,
                    artifact_stem="dino",
                    thresholds=ParityThresholds(1e-4, 1e-3, 1e-4, 0.999),
                )
            self.assertEqual(existing.read_bytes(), b"preserve")
            self.assertEqual(tuple(root.iterdir()), (existing,))


if __name__ == "__main__":
    unittest.main()
