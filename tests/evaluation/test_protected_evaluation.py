from __future__ import annotations

import hashlib
import json
import subprocess
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from evaluation.protected_evaluation import (
    REPORT_INTERPRETATION,
    REPORT_PROTOCOL_STATUS,
    ProtectedEmbeddingManifest,
    ProtectedEmbeddingRecord,
    ProtectedEvaluationPolicy,
    ProtectedEvaluationRoleBinding,
    prepare_protected_evaluation,
    validate_protected_report,
    verify_protected_evaluation_output,
)
from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256
from evaluation.splits.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
)

def _token(*values: object) -> str:
    return hashlib.sha256("\0".join(map(str, values)).encode()).hexdigest()

def _write(path: Path, payload: object) -> None:
    path.write_text(json.dumps(payload, sort_keys=True, indent=2) + "\n", encoding="utf-8")

class ProtectedEvaluationTests(unittest.TestCase):
    def test_protected_cli_help_surfaces_are_available(self) -> None:
        commands = (
            [sys.executable, "evaluation/protected_prepare.py", "--help"],
            [sys.executable, "-m", "evaluation.commands.evaluate", "protected", "--help"],
            [sys.executable, "evaluation/protected_verify.py", "--help"],
        )
        for command in commands:
            completed = subprocess.run(
                command, capture_output=True, text=True, timeout=30
            )
            self.assertEqual(completed.returncode, 0, completed.stderr)

    def _fixture(
        self,
        root: Path,
        *,
        score_cap: int = 16,
        prior_stage: ExposureStage = ExposureStage.BYTES_EXPORTED,
    ) -> dict[str, Path | str]:
        policy = ProtectedEvaluationPolicy(
            role_bindings=(
                ProtectedEvaluationRoleBinding("gallery", "FIXTURE", "PRIMARY", 2, 1, "GALLERY"),
                ProtectedEvaluationRoleBinding("queries", "FIXTURE", "PRIMARY", 2, 1, "KNOWN_QUERY"),
            ),
            rank_ks=(1, 2),
            bootstrap_resamples=20,
            bootstrap_seed=3,
            maximum_json_bytes=1_048_576,
            maximum_json_depth=32,
            maximum_json_nodes=10_000,
            maximum_json_keys=10_000,
            maximum_json_array_length=10_000,
            maximum_json_string_characters=4096,
            maximum_json_number_characters=64,
            maximum_samples_per_input=4,
            maximum_embedding_dimension=4,
            maximum_total_embedding_values=16,
            maximum_score_matrix_elements=score_cap,
            score_dtype="float64",
            metric="cosine",
            self_match_policy="exclude",
        )
        identities = [_token("identity", index) for index in range(2)]
        subjects = [_token("subject", index) for index in range(2)]
        gallery_records = tuple(sorted((
            ProtectedEmbeddingRecord(_token("g", index), identities[index], subjects[index], _token("gt", index), tuple(vector))
            for index, vector in enumerate(((1.0, 0.0), (0.0, 1.0)))
        ), key=lambda item: item.sample_token))
        query_records = tuple(sorted((
            ProtectedEmbeddingRecord(_token("q", index), identities[index], subjects[index], _token("qt", index), tuple(vector))
            for index, vector in enumerate(((0.9, 0.1), (0.1, 0.9)))
        ), key=lambda item: item.sample_token))
        gallery = ProtectedEmbeddingManifest("gallery", _token("gallery-production"), gallery_records)
        queries = ProtectedEmbeddingManifest("queries", _token("query-production"), query_records)
        uses = {
            "gallery": {
                "protocol": "FIXTURE", "episode": "PRIMARY", "gallery_size": 2,
                "shot": 1, "role": "GALLERY",
            },
            "queries": {
                "protocol": "FIXTURE", "episode": "PRIMARY", "gallery_size": 2,
                "shot": 1, "role": "KNOWN_QUERY",
            },
        }
        assignment = {
            "schema_version": "cvi.protected_public_split_assignment.v1",
            "status": "PASS_PROTECTED_SPLIT_CONSTRUCTION",
            "records": [
                {
                    "sample_token": record.sample_token,
                    "identity_token": record.identity_token,
                    "uses": [uses[name]],
                }
                for name, records in (("gallery", gallery.records), ("queries", queries.records))
                for record in records
            ],
        }
        declaration = RoleExposureDeclaration(
            source_artifact_sha256=_token("initial-exposure"),
            kind=ExposureDeclarationKind.PRIOR_ASSIGNMENT,
            revoked=False,
            records=tuple(sorted((
                RoleExposureDeclarationRecord(
                    record.sample_token,
                    record.identity_token,
                    record.public_subject_token,
                    prior_stage,
                )
                for record in (*gallery.records, *queries.records)
            ), key=lambda item: item.sample_token)),
        )
        ledger = merge_role_exposure_declarations((declaration,))
        receipt = create_role_exposure_receipt(ledger)
        paths = {name: root / f"{name}.json" for name in (
            "policy", "split_assignment", "split_receipt", "exposure_ledger",
            "exposure_receipt", "gallery", "queries", "external_pins",
        )}
        _write(paths["policy"], policy.to_dict())
        _write(paths["split_assignment"], assignment)
        _write(paths["split_receipt"], {
            "schema_version": "cvi.protected_public_split_receipt.v2",
            "assignment_sha256": content_sha256(assignment),
            "role_exposure_ledger_sha256": ledger.ledger_sha256,
            "role_exposure_receipt_sha256": receipt.receipt_sha256,
        })
        _write(paths["exposure_ledger"], ledger.to_dict())
        _write(paths["exposure_receipt"], receipt.to_dict())
        _write(paths["gallery"], gallery.to_dict())
        _write(paths["queries"], queries.to_dict())
        raw = {name: read_strict_json_document(path).raw_sha256 for name, path in paths.items() if name != "external_pins"}
        pins = {
            "policy_raw_sha256": raw["policy"],
            "split_assignment_raw_sha256": raw["split_assignment"],
            "split_receipt_raw_sha256": raw["split_receipt"],
            "exposure_ledger_raw_sha256": raw["exposure_ledger"],
            "exposure_receipt_raw_sha256": raw["exposure_receipt"],
            "gallery_raw_sha256": raw["gallery"],
            "gallery_production_receipt_sha256": gallery.production_receipt_sha256,
            "queries_raw_sha256": raw["queries"],
            "queries_production_receipt_sha256": queries.production_receipt_sha256,
            "schema_version": "cvi.protected_evaluation_external_pins.v1",
        }
        _write(paths["external_pins"], pins)
        return {
            **paths,
            "external_pins_raw_sha256": read_strict_json_document(paths["external_pins"]).raw_sha256,
        }

    def _prepare(self, fixture: dict[str, Path | str], output: Path):
        return prepare_protected_evaluation(
            policy_path=fixture["policy"],
            external_pins_path=fixture["external_pins"],
            expected_external_pins_raw_sha256=fixture["external_pins_raw_sha256"],
            split_assignment_path=fixture["split_assignment"],
            split_receipt_path=fixture["split_receipt"],
            exposure_ledger_path=fixture["exposure_ledger"],
            exposure_receipt_path=fixture["exposure_receipt"],
            gallery_path=fixture["gallery"],
            queries_path=fixture["queries"],
            output_directory=output,
            tool_provenance={"schema_version": "fixture", "source": "unit-test"},
        )

    def test_bounded_same_read_hashes_raw_and_canonical_payload(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "value.json"
            path.write_text('{"b": 2, "a": 1}\n', encoding="utf-8")
            document = read_strict_json_document(path, maximum_bytes=64)
            self.assertEqual(document.raw_sha256, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(document.canonical_payload_sha256, content_sha256({"a": 1, "b": 2}))
            with self.assertRaisesRegex(ValueError, "byte limit"):
                read_strict_json_document(path, maximum_bytes=4)
            path.write_text('{"a": 1, "a": 2}', encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate"):
                read_strict_json_document(path)

    def test_exact_key_dataclasses_reject_unknown_fields(self) -> None:
        payload = json.loads(Path("archive/shared_helpers/configs/contracts/protected_evaluation_policy.example.json").read_text())
        payload["unknown"] = True
        with self.assertRaisesRegex(ValueError, "keys differ"):
            ProtectedEvaluationPolicy.from_dict(payload)

    def test_complete_prepare_evaluate_verify_chain(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root)
            preparation = root / "prepared"
            plan = self._prepare(fixture, preparation)
            declaration = json.loads((preparation / "advanced_exposure_declaration.json").read_text())
            self.assertEqual(
                {record["stage"] for record in declaration["records"]},
                {"FINAL_TEST_SCORED"},
            )
            output = root / "output"
            command = [
                sys.executable, "-m", "evaluation.commands.evaluate", "protected",
                "--preparation-directory", str(preparation),
                "--expected-plan-receipt-sha256", plan.receipt_sha256,
                "--expected-advanced-exposure-declaration-sha256", plan.advanced_exposure_declaration_sha256,
                "--policy", str(fixture["policy"]),
                "--split-assignment", str(fixture["split_assignment"]),
                "--split-receipt", str(fixture["split_receipt"]),
                "--exposure-ledger", str(fixture["exposure_ledger"]),
                "--exposure-receipt", str(fixture["exposure_receipt"]),
                "--gallery", str(fixture["gallery"]),
                "--queries", str(fixture["queries"]),
                "--output-directory", str(output),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, timeout=30)
            self.assertEqual(completed.returncode, 0, completed.stdout + completed.stderr)
            output_anchor = json.loads(completed.stdout)["output_receipt_sha256"]
            receipt = verify_protected_evaluation_output(
                preparation_directory=preparation,
                output_directory=output,
                expected_plan_receipt_sha256=plan.receipt_sha256,
                expected_advanced_exposure_declaration_sha256=plan.advanced_exposure_declaration_sha256,
                expected_output_receipt_sha256=output_anchor,
            )
            report = json.loads((output / "report.json").read_text())
            self.assertEqual(report["protocol_status"], REPORT_PROTOCOL_STATUS)
            self.assertTrue(report["receipt_chain_verified"])
            self.assertFalse(report["valid_for_model_selection"])
            self.assertFalse(report["valid_for_final_reporting"])
            self.assertEqual(report["interpretation"], REPORT_INTERPRETATION)
            self.assertEqual(receipt.status, REPORT_PROTOCOL_STATUS)
            self.assertEqual(receipt.interpretation, REPORT_INTERPRETATION)
            self.assertEqual(
                receipt.schema_version,
                "cvi.protected_evaluation_output_receipt.v2",
            )
            self.assertEqual(receipt.report_raw_sha256, hashlib.sha256((output / "report.json").read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, "external output anchor"):
                verify_protected_evaluation_output(
                    preparation_directory=preparation,
                    output_directory=output,
                    expected_plan_receipt_sha256=plan.receipt_sha256,
                    expected_advanced_exposure_declaration_sha256=plan.advanced_exposure_declaration_sha256,
                    expected_output_receipt_sha256="0" * 64,
                )

    def test_external_raw_pin_and_resource_cap_fail_before_publication(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(root, score_cap=3)
            with self.assertRaisesRegex(ValueError, "score matrix"):
                self._prepare(fixture, root / "not-published")
            self.assertFalse((root / "not-published").exists())
            second = root / "second"
            second.mkdir()
            fixture = self._fixture(second)
            policy_path = fixture["policy"]
            policy_path.write_text(policy_path.read_text() + " ", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "raw-byte pin"):
                self._prepare(fixture, second / "not-published")

    def test_actual_prior_scoring_exposure_blocks_final_evaluation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            fixture = self._fixture(
                root, prior_stage=ExposureStage.MODEL_SELECTION_SCORED
            )
            with self.assertRaisesRegex(ValueError, "previously advanced"):
                self._prepare(fixture, root / "not-published")
            self.assertFalse((root / "not-published").exists())

    def test_recursive_v3_schema_rejects_nested_unknown_key(self) -> None:
        report = {
            "schema_version": "cvi.evaluation.report.v3",
            "protocol": "protected_retrieval",
            "protocol_status": REPORT_PROTOCOL_STATUS,
            "receipt_chain_verified": True,
            "valid_for_model_selection": False,
            "valid_for_final_reporting": False,
            "evaluation_token": "0" * 64,
            "receipt_chain": {
                "plan_receipt_sha256": "0" * 64,
                "policy_receipt_sha256": "0" * 64,
                "input_receipt_sha256": "0" * 64,
                "advanced_exposure_declaration_sha256": "0" * 64,
                "split_assignment_sha256": "0" * 64,
                "prior_exposure_ledger_sha256": "0" * 64,
                "prior_exposure_receipt_sha256": "0" * 64,
                "unexpected": True,
            },
            "protocol_configuration": {
                "metric": "cosine", "score_dtype": "float64", "self_match_policy": "exclude",
                "aggregation": "max", "tie_policy": "stable_first_gallery_identity_occurrence",
                "rank_ks": [1], "bootstrap_resamples": 1, "bootstrap_seed": 0,
            },
            "input_summary": {
                "gallery_templates": 1, "query_templates": 1, "gallery_identities": 1,
                "query_identities": 1, "embedding_dimension": 1,
                "total_embedding_values": 2, "score_matrix_elements": 1,
            },
            "metrics": {
                "mAP": 1.0, "mINP": 1.0, "MRR": 1.0,
                "rank_at_k": [{"k": 1, "value": 1.0}],
                "identity_clustered_bootstrap": [{
                    "metric": "AP", "estimate": 1.0, "lower_bound": 1.0,
                    "upper_bound": 1.0, "confidence_level": 0.95,
                    "cluster_unit": "query_identity", "cluster_count": 2,
                    "query_row_count": 2, "resamples": 1, "seed": 0,
                    "interval_method": "whole_identity_percentile_bootstrap",
                }],
            },
            "resource_bounds": {
                "maximum_samples_per_input": 1, "maximum_embedding_dimension": 1,
                "maximum_total_embedding_values": 2, "maximum_score_matrix_elements": 1,
            },
            "evaluator_provenance_sha256": "0" * 64,
            "interpretation": REPORT_INTERPRETATION,
        }
        with self.assertRaisesRegex(ValueError, "Additional properties"):
            validate_protected_report(report)

        report["receipt_chain"].pop("unexpected")
        validate_protected_report(report)
        report["valid_for_final_reporting"] = True
        with self.assertRaisesRegex(ValueError, "False was expected"):
            validate_protected_report(report)

if __name__ == "__main__":
    unittest.main()
