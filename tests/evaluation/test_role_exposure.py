from __future__ import annotations

import copy
import hashlib
import unittest

from evaluation.splits.role_exposure import (
    CandidateRoleAssignment,
    CandidateRoleRecord,
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    RoleExposureLedger,
    RoleExposureReceipt,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
    validate_candidate_assignment,
    verify_role_exposure_receipt,
)

def _token(kind: str, value: object) -> str:
    return hashlib.sha256(f"{kind}\0{value}".encode()).hexdigest()

def _record(
    sample: int,
    stage: ExposureStage,
    *,
    identity: int = 1,
    subject: int = 1,
) -> RoleExposureDeclarationRecord:
    return RoleExposureDeclarationRecord(
        sample_token=_token("sample", sample),
        identity_token=_token("identity", identity),
        public_subject_token=_token("public-subject", subject),
        stage=stage,
    )

def _kind(stage: ExposureStage) -> ExposureDeclarationKind:
    if stage in {ExposureStage.BYTES_EXPORTED, ExposureStage.MODEL_TRAINING_USED}:
        return ExposureDeclarationKind.PRIOR_ASSIGNMENT
    return ExposureDeclarationKind.PRIOR_EVALUATION

def _declaration(
    artifact: int,
    *records: RoleExposureDeclarationRecord,
    revoked: bool = False,
) -> RoleExposureDeclaration:
    return RoleExposureDeclaration(
        source_artifact_sha256=_token("artifact", artifact),
        kind=_kind(records[0].stage),
        revoked=revoked,
        records=tuple(sorted(records, key=lambda record: record.sample_token)),
    )

def _candidate(
    artifact: int,
    sample: int,
    stage: ExposureStage,
    *,
    identity: int = 1,
    subject: int = 1,
) -> CandidateRoleAssignment:
    return CandidateRoleAssignment(
        source_artifact_sha256=_token("candidate", artifact),
        records=(
            CandidateRoleRecord(
                sample_token=_token("sample", sample),
                identity_token=_token("identity", identity),
                public_subject_token=_token("public-subject", subject),
                assigned_stage=stage,
            ),
        ),
    )

class RoleExposureTests(unittest.TestCase):
    def test_merge_is_order_independent_monotonic_and_round_trips(self) -> None:
        exported = _declaration(
            1,
            _record(1, ExposureStage.BYTES_EXPORTED),
            _record(2, ExposureStage.BYTES_EXPORTED),
        )
        selected = _declaration(
            2,
            _record(1, ExposureStage.MODEL_SELECTION_SCORED),
        )

        left = merge_role_exposure_declarations((exported, selected))
        right = merge_role_exposure_declarations((selected, exported))

        self.assertEqual(left, right)
        self.assertEqual(left.ledger_sha256, right.ledger_sha256)
        self.assertEqual(
            left.records[0].maximum_historical_stage,
            ExposureStage.MODEL_SELECTION_SCORED,
        )
        self.assertEqual(
            left.records[0].source_artifact_sha256s,
            tuple(sorted((_token("artifact", 1), _token("artifact", 2)))),
        )
        restored = RoleExposureLedger.from_dict(left.to_dict())
        self.assertEqual(restored, left)
        self.assertIn("NO_AUTOMATIC_DISCOVERY", exported.interpretation)

    def test_revoked_artifact_still_counts_and_receipt_binds_every_source(self) -> None:
        exported = _declaration(1, _record(1, ExposureStage.BYTES_EXPORTED))
        final = _declaration(
            2,
            _record(1, ExposureStage.FINAL_TEST_SCORED),
            revoked=True,
        )
        ledger = merge_role_exposure_declarations((final, exported))
        receipt = create_role_exposure_receipt(ledger)

        self.assertEqual(
            ledger.records[0].maximum_historical_stage,
            ExposureStage.FINAL_TEST_SCORED,
        )
        self.assertEqual(
            receipt.source_artifact_sha256s,
            tuple(sorted((_token("artifact", 1), _token("artifact", 2)))),
        )
        self.assertEqual(
            receipt.revoked_source_artifact_sha256s,
            (_token("artifact", 2),),
        )
        verify_role_exposure_receipt(ledger, receipt)
        self.assertEqual(RoleExposureReceipt.from_dict(receipt.to_dict()), receipt)

        other_ledger = merge_role_exposure_declarations((exported,))
        with self.assertRaisesRegex(ValueError, "differs from ledger"):
            verify_role_exposure_receipt(other_ledger, receipt)

    def test_candidate_assignment_rejects_every_less_protected_lane(self) -> None:
        regressions = {
            ExposureStage.MODEL_SELECTION_SCORED: (
                ExposureStage.MODEL_TRAINING_USED,
            ),
            ExposureStage.CALIBRATION_SCORED: (
                ExposureStage.MODEL_TRAINING_USED,
                ExposureStage.MODEL_SELECTION_SCORED,
            ),
            ExposureStage.FINAL_TEST_SCORED: (
                ExposureStage.MODEL_TRAINING_USED,
                ExposureStage.MODEL_SELECTION_SCORED,
                ExposureStage.CALIBRATION_SCORED,
            ),
        }
        artifact = 10
        for historical, forbidden in regressions.items():
            with self.subTest(historical=historical):
                ledger = merge_role_exposure_declarations(
                    (_declaration(artifact, _record(1, historical)),)
                )
                validate_candidate_assignment(
                    ledger, _candidate(artifact, 1, historical)
                )
                for proposed in forbidden:
                    with self.assertRaisesRegex(ValueError, "role regression"):
                        validate_candidate_assignment(
                            ledger, _candidate(artifact, 1, proposed)
                        )
            artifact += 1

    def test_identity_history_protects_new_samples_and_sample_links_are_stable(self) -> None:
        ledger = merge_role_exposure_declarations(
            (
                _declaration(
                    1,
                    _record(1, ExposureStage.FINAL_TEST_SCORED),
                ),
            )
        )
        with self.assertRaisesRegex(ValueError, "role regression"):
            validate_candidate_assignment(
                ledger,
                _candidate(2, 2, ExposureStage.CALIBRATION_SCORED),
            )
        with self.assertRaisesRegex(ValueError, "historical identity links"):
            validate_candidate_assignment(
                ledger,
                _candidate(
                    3,
                    1,
                    ExposureStage.FINAL_TEST_SCORED,
                    identity=2,
                    subject=2,
                ),
            )

    def test_duplicate_and_conflicting_declarations_fail_closed(self) -> None:
        first = _declaration(1, _record(1, ExposureStage.BYTES_EXPORTED))
        with self.assertRaisesRegex(ValueError, "source artifact hashes must be unique"):
            merge_role_exposure_declarations((first, first))

        duplicate = _record(1, ExposureStage.BYTES_EXPORTED)
        with self.assertRaisesRegex(ValueError, "sample tokens must be unique"):
            _declaration(2, duplicate, duplicate)

        conflicting_sample = _declaration(
            3,
            _record(
                1,
                ExposureStage.MODEL_SELECTION_SCORED,
                identity=2,
                subject=2,
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting identity declarations"):
            merge_role_exposure_declarations((first, conflicting_sample))

        conflicting_identity = _declaration(
            4,
            _record(
                2,
                ExposureStage.MODEL_SELECTION_SCORED,
                identity=1,
                subject=2,
            ),
        )
        with self.assertRaisesRegex(ValueError, "conflicting public subject"):
            merge_role_exposure_declarations((first, conflicting_identity))

    def test_strict_schema_rejects_malformed_input(self) -> None:
        declaration = _declaration(
            1, _record(1, ExposureStage.MODEL_SELECTION_SCORED)
        )
        payload = declaration.to_dict()
        payload["undeclared_discovery"] = True
        with self.assertRaisesRegex(ValueError, "keys mismatch"):
            RoleExposureDeclaration.from_dict(payload)

        record_payload = declaration.records[0].to_dict()
        record_payload["sample_token"] = "A" * 64
        with self.assertRaisesRegex(ValueError, "lowercase SHA-256"):
            RoleExposureDeclarationRecord.from_dict(record_payload)

        unknown_kind = copy.deepcopy(declaration.to_dict())
        unknown_kind["kind"] = "AUTOMATIC_DISCOVERY"
        with self.assertRaises(ValueError):
            RoleExposureDeclaration.from_dict(unknown_kind)

        with self.assertRaisesRegex(ValueError, "explicit role exposure declaration"):
            merge_role_exposure_declarations(())

if __name__ == "__main__":
    unittest.main()
