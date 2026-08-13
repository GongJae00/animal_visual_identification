from __future__ import annotations

import ast
import copy
import hashlib
import math
import unittest
from dataclasses import replace
from pathlib import Path

from evaluation.open_set_calibration import (
    AuthenticatedOpenSetCalibrationPanel,
    BlindOpenSetScoreRow,
    DistinctIdentityScore,
    OpenSetCalibrationPolicy,
    OpenSetCalibrationReceipt,
    OpenSetDisposition,
    apply_open_set_boundary,
    freeze_open_set_threshold,
    maximum_allowed_calibration_accepts,
    top_identity_evidence,
    zero_event_one_sided_upper_bound,
)
from identity.splits.protected_public_split import ProtectedPublicSplitPolicy


def _token(kind: str, value: object) -> str:
    return hashlib.sha256(f"{kind}\0{value}".encode()).hexdigest()


def _row(index: int, *, top: float = 0.8, second: float = 0.5, n: int = 300, shot: int = 3) -> BlindOpenSetScoreRow:
    scores = [
        DistinctIdentityScore(_token("slot", slot), -0.5 + slot / 1000.0)
        for slot in range(n)
    ]
    scores[-1] = DistinctIdentityScore(scores[-1].identity_slot_token, top)
    scores[-2] = DistinctIdentityScore(scores[-2].identity_slot_token, second)
    return BlindOpenSetScoreRow(
        query_token=_token("query", index),
        gallery_size=n,
        shot=shot,
        scores=tuple(sorted(scores, key=lambda item: item.identity_slot_token)),
    )


def _rows(**kwargs: object) -> tuple[BlindOpenSetScoreRow, ...]:
    return tuple(sorted((_row(index, **kwargs) for index in range(300)), key=lambda row: row.query_token))


def _panel(*, n: int = 300, shot: int = 3) -> AuthenticatedOpenSetCalibrationPanel:
    slots = tuple(sorted(_token("slot", slot) for slot in range(n)))
    queries = tuple(sorted(_token("query", index) for index in range(300)))
    return AuthenticatedOpenSetCalibrationPanel(
        split_assignment_sha256="1" * 64,
        split_policy_sha256=ProtectedPublicSplitPolicy().policy_sha256,
        gallery_size=n,
        shot=shot,
        gallery_identity_slot_tokens=slots,
        unknown_query_event_tokens=queries,
        episode=f"N_{n}",
    )


def _freeze(rows: tuple[BlindOpenSetScoreRow, ...], *, margin: float = 0.1):
    return freeze_open_set_threshold(
        rows,
        policy=OpenSetCalibrationPolicy(),
        panel=_panel(),
        margin_threshold=margin,
        margin_selection_receipt_sha256="3" * 64,
        calibration_score_receipt_sha256="4" * 64,
        model_sha256="5" * 64,
        preprocessing_sha256="6" * 64,
        scoring_semantics_sha256="7" * 64,
        precision="FP32",
        score_dtype="IEEE754_BINARY32_ACCUMULATED_FP32",
    )


class OpenSetCalibrationTests(unittest.TestCase):
    def test_exact_binomial_golden_capacity_and_zero_event_bounds(self) -> None:
        self.assertIsNone(maximum_allowed_calibration_accepts(trials=298, target_fpir=0.01, one_sided_alpha=0.05))
        self.assertEqual(maximum_allowed_calibration_accepts(trials=299, target_fpir=0.01, one_sided_alpha=0.05), 0)
        self.assertEqual(maximum_allowed_calibration_accepts(trials=300, target_fpir=0.01, one_sided_alpha=0.05), 0)
        self.assertAlmostEqual(zero_event_one_sided_upper_bound(trials=300, alpha=0.05), 0.009936, places=6)
        self.assertAlmostEqual(zero_event_one_sided_upper_bound(trials=423, alpha=0.05), 0.007057, places=6)
        self.assertAlmostEqual(zero_event_one_sided_upper_bound(trials=32, alpha=0.05), 0.0893682, places=6)
        self.assertAlmostEqual(zero_event_one_sided_upper_bound(trials=20, alpha=0.05), 0.139108, places=6)

    def test_threshold_is_strictly_above_maximum_effective_score(self) -> None:
        receipt = _freeze(_rows())
        self.assertEqual(receipt.status, "PASS_EXACT_OPEN_SET_CALIBRATION")
        self.assertIsNotNone(receipt.boundary)
        assert receipt.boundary is not None
        self.assertEqual(receipt.summary.allowed_calibration_accepts, 0)
        self.assertEqual(receipt.summary.observed_calibration_accepts, 0)
        self.assertEqual(receipt.summary.order_statistic_rank_one_based, 300)
        self.assertEqual(receipt.summary.order_statistic_value, 0.8)
        self.assertEqual(receipt.boundary.score_threshold, math.nextafter(0.8, math.inf))
        self.assertEqual(OpenSetCalibrationReceipt.from_dict(receipt.to_dict()), receipt)

    def test_score_one_uses_representable_above_one_threshold(self) -> None:
        receipt = _freeze(_rows(top=1.0))
        assert receipt.boundary is not None
        self.assertFalse(receipt.boundary.automatic_accept_enabled)
        self.assertGreater(receipt.boundary.score_threshold, 1.0)
        result = apply_open_set_boundary(_row(999, top=1.0), receipt.boundary)
        self.assertEqual(result.disposition, OpenSetDisposition.REVIEW_REQUIRED)

    def test_distinct_identity_top_tie_is_review_not_token_winner(self) -> None:
        row = _row(0, top=0.8, second=0.8)
        evidence = top_identity_evidence(row, margin_threshold=0.1)
        self.assertFalse(evidence.unique_top_identity)
        self.assertIsNone(evidence.top_identity_slot_token)
        receipt = _freeze(_rows())
        assert receipt.boundary is not None
        result = apply_open_set_boundary(row, receipt.boundary)
        self.assertEqual(result.disposition, OpenSetDisposition.REVIEW_REQUIRED)
        self.assertIsNone(result.predicted_identity_slot_token)

    def test_margin_failure_is_review_and_threshold_failure_is_unknown(self) -> None:
        receipt = _freeze(_rows())
        assert receipt.boundary is not None
        review = apply_open_set_boundary(_row(1, top=0.55, second=0.5), receipt.boundary)
        unknown = apply_open_set_boundary(_row(2, top=0.7, second=0.5), receipt.boundary)
        self.assertEqual(review.disposition, OpenSetDisposition.REVIEW_REQUIRED)
        self.assertEqual(unknown.disposition, OpenSetDisposition.UNKNOWN)

    def test_capacity_failure_has_no_boundary_and_no_silent_backfill(self) -> None:
        receipt = _freeze(_rows()[:-1])
        self.assertEqual(receipt.status, "CALIBRATION_CAPACITY_FAILED")
        self.assertIsNone(receipt.boundary)
        self.assertEqual(receipt.summary.unknown_identity_events, 299)
        empty = _freeze(())
        self.assertEqual(empty.status, "CALIBRATION_CAPACITY_FAILED")
        self.assertEqual(empty.summary.zero_event_fpir_upper_bound, 1.0)

    def test_registered_gallery_and_shot_are_exact_no_interpolation(self) -> None:
        with self.assertRaisesRegex(ValueError, "RECALIBRATION_REQUIRED"):
            freeze_open_set_threshold(
                (),
                policy=OpenSetCalibrationPolicy(),
                panel=_panel(n=200),
                margin_threshold=0.1,
                margin_selection_receipt_sha256="3" * 64,
                calibration_score_receipt_sha256="4" * 64,
                model_sha256="5" * 64,
                preprocessing_sha256="6" * 64,
                scoring_semantics_sha256="7" * 64,
                precision="FP32",
                score_dtype="FP32",
            )
        with self.assertRaisesRegex(ValueError, "RECALIBRATION_REQUIRED"):
            freeze_open_set_threshold(
                (),
                policy=OpenSetCalibrationPolicy(),
                panel=_panel(shot=5),
                margin_threshold=0.1,
                margin_selection_receipt_sha256="3" * 64,
                calibration_score_receipt_sha256="4" * 64,
                model_sha256="5" * 64,
                preprocessing_sha256="6" * 64,
                scoring_semantics_sha256="7" * 64,
                precision="FP32",
                score_dtype="FP32",
            )
        receipt = _freeze(_rows())
        assert receipt.boundary is not None
        with self.assertRaisesRegex(ValueError, "RECALIBRATION_REQUIRED"):
            apply_open_set_boundary(_row(0, n=100), receipt.boundary)

    def test_rows_are_label_free_distinct_identity_aggregated_and_canonical(self) -> None:
        row = _row(0)
        self.assertEqual(BlindOpenSetScoreRow.from_dict(row.to_dict()), row)
        with self.assertRaisesRegex(ValueError, "canonically sorted"):
            replace(row, scores=tuple(reversed(row.scores)))
        with self.assertRaisesRegex(ValueError, "cardinality"):
            replace(row, scores=row.scores[:-1])
        with self.assertRaisesRegex(ValueError, "semantics"):
            replace(row, score_semantics="PROTOTYPE_LEVEL")
        source = ast.parse(Path("evaluation/open_set_calibration.py").read_text())
        forbidden = {"dog_id", "known_role", "unknown_role", "true_identity", "test_score"}
        names = {node.id for node in ast.walk(source) if isinstance(node, ast.Name)}
        self.assertFalse(forbidden & names)

    def test_arbitrary_query_or_gallery_slots_cannot_rebind_calibration(self) -> None:
        rows = list(_rows())
        rows[0] = replace(rows[0], query_token=_token("foreign-query", 0))
        rows.sort(key=lambda row: row.query_token)
        with self.assertRaisesRegex(ValueError, "queries differ"):
            _freeze(tuple(rows))

        rows = list(_rows())
        scores = list(rows[0].scores)
        scores[0] = DistinctIdentityScore(_token("foreign-slot", 0), scores[0].score)
        scores.sort(key=lambda item: item.identity_slot_token)
        rows[0] = replace(rows[0], scores=tuple(scores))
        with self.assertRaisesRegex(ValueError, "gallery slots differ"):
            _freeze(tuple(rows))

    def test_secondary_panel_requires_precommitted_familywise_allocation(self) -> None:
        with self.assertRaisesRegex(ValueError, "FAMILYWISE_ALLOCATION_REQUIRED"):
            freeze_open_set_threshold(
                _rows(n=100),
                policy=OpenSetCalibrationPolicy(),
                panel=_panel(n=100),
                margin_threshold=0.1,
                margin_selection_receipt_sha256="3" * 64,
                calibration_score_receipt_sha256="4" * 64,
                model_sha256="5" * 64,
                preprocessing_sha256="6" * 64,
                scoring_semantics_sha256="7" * 64,
                precision="FP32",
                score_dtype="FP32",
            )

    def test_policy_and_lineage_are_strictly_bound(self) -> None:
        policy = OpenSetCalibrationPolicy()
        self.assertEqual(OpenSetCalibrationPolicy.from_dict(policy.to_dict()), policy)
        payload = policy.to_dict()
        payload["registered_shots"] = [1, 3, 5]
        payload["primary_shot"] = 5
        with self.assertRaisesRegex(ValueError, "constants differ"):
            OpenSetCalibrationPolicy.from_dict(payload)
        with self.assertRaisesRegex(ValueError, "SHA-256"):
            freeze_open_set_threshold(
                _rows(),
                policy=policy,
                panel=replace(_panel(), split_assignment_sha256="x" * 64),
                margin_threshold=0.1,
                margin_selection_receipt_sha256="3" * 64,
                calibration_score_receipt_sha256="4" * 64,
                model_sha256="5" * 64,
                preprocessing_sha256="6" * 64,
                scoring_semantics_sha256="7" * 64,
                precision="FP32",
                score_dtype="FP32",
            )

    def test_receipt_recomputes_exact_allowance_rank_threshold_and_upper_bound(self) -> None:
        receipt = _freeze(_rows())
        mutations = (
            ("allowed", ("summary", "allowed_calibration_accepts"), 1),
            ("rank", ("summary", "order_statistic_rank_one_based"), 299),
            ("upper bound", ("summary", "zero_event_fpir_upper_bound"), 0.5),
            ("threshold", ("boundary", "score_threshold"), 0.8),
            ("policy", ("boundary", "policy_sha256"), "0" * 64),
        )
        for label, path, value in mutations:
            with self.subTest(label=label):
                payload = copy.deepcopy(receipt.to_dict())
                payload[path[0]][path[1]] = value
                with self.assertRaises(ValueError):
                    OpenSetCalibrationReceipt.from_dict(payload)

        disabled = _freeze(_rows(top=1.0))
        payload = copy.deepcopy(disabled.to_dict())
        payload["boundary"]["automatic_accept_enabled"] = True
        with self.assertRaisesRegex(ValueError, "automatic accept state"):
            OpenSetCalibrationReceipt.from_dict(payload)


if __name__ == "__main__":
    unittest.main()
