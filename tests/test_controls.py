from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from data_pipeline.acquisition import sha256_file
from evaluation.controls import (
    ControlMaskEntry,
    ControlMaskManifest,
    ControlMaskVerification,
    MaskEvidence,
    MaskReviewStatus,
    MaskRole,
    VisualControlKind,
    VisualControlPanel,
    VisualControlPolicy,
    VisualControlRecipe,
    plan_visual_control_audit,
    verify_control_mask_files,
)
from evaluation.pairing import (
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairScoringRequest,
    PairStratum,
)
from evaluation.mask_semantics import (
    MaskEntrySemanticReceipt,
    MaskPixelStats,
    MaskSemanticVerification,
)
from foundation.provenance import content_sha256
from evaluation.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def construction() -> PairConstructionResult:
    requests = (
        PairScoringRequest("pair-0", "q1", "r1"),
        PairScoringRequest("pair-1", "q1", "r2"),
        PairScoringRequest("pair-2", "q2", "r3"),
    )
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=HASH_B,
        attributes_sha256=HASH_C,
        eligible_query_count=2,
        selected_query_count=2,
        dropped_query_count=0,
        scoring_requests=requests,
        artifact_bindings=tuple(
            PairArtifactBinding(token, f"sample-{token}")
            for token in ("q1", "q2", "r1", "r2", "r3")
        ),
        ground_truth=(
            PairGroundTruth(
                "pair-0", "dog-1", "dog-1", "sq1", "sr1",
                PairStratum.POSITIVE,
            ),
            PairGroundTruth(
                "pair-1", "dog-1", "dog-2", "sq1", "sr2",
                PairStratum.SAME_BREED,
            ),
            PairGroundTruth(
                "pair-2", "dog-3", "dog-4", "sq2", "sr3",
                PairStratum.RANDOM,
            ),
        ),
        quotas=(),
    )


def artifact_manifest(
    pairs: PairConstructionResult,
) -> PairArtifactManifest:
    return PairArtifactManifest(
        pair_set_sha256=pairs.result_sha256,
        artifact_bindings_sha256=content_sha256(
            pairs.artifact_binding_payload()
        ),
        entries=tuple(
            PairArtifactEntry(
                token,
                f"{token}.png",
                content_sha256({"token": token}),
                100,
                "image/png",
            )
            for token in ("q1", "q2", "r1", "r2", "r3")
        ),
    )


def mask(
    token: str,
    role: MaskRole,
    *,
    status: MaskReviewStatus = MaskReviewStatus.VERIFIED,
) -> MaskEvidence:
    mask_token = f"mask-{token}-{role.value.casefold()}"
    return MaskEvidence(
        role=role,
        artifact_token=mask_token,
        relative_path=f"{mask_token}.png",
        content_sha256=content_sha256(
            {"token": token, "role": role.value}
        ),
        byte_size=20,
        width=100,
        height=80,
        annotation_version="annotation-v1",
        provenance_kind="manual-reviewed",
        provenance_reference_sha256=HASH_D,
        review_status=status,
    )


def mask_manifest(
    base: PairArtifactManifest,
) -> ControlMaskManifest:
    return ControlMaskManifest(
        base_artifact_manifest_sha256=base.manifest_sha256,
        entries=(
            ControlMaskEntry(
                "q1",
                (
                    mask("q1", MaskRole.DOG),
                    mask("q1", MaskRole.ACCESSORY),
                ),
            ),
            ControlMaskEntry(
                "q2",
                (
                    mask(
                        "q2",
                        MaskRole.DOG,
                        status=MaskReviewStatus.UNVERIFIED,
                    ),
                ),
            ),
            ControlMaskEntry(
                "r1",
                (
                    mask("r1", MaskRole.DOG),
                    mask("r1", MaskRole.ACCESSORY),
                ),
            ),
            ControlMaskEntry("r2", (mask("r2", MaskRole.DOG),)),
            ControlMaskEntry("r3", ()),
        ),
    )


def base_verification(
    manifest: PairArtifactManifest,
) -> PairArtifactVerification:
    return PairArtifactVerification(
        artifact_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(manifest.entries),
        verified_bytes=sum(entry.byte_size for entry in manifest.entries),
    )


def mask_verification(
    manifest: ControlMaskManifest,
) -> ControlMaskVerification:
    masks = tuple(
        mask for entry in manifest.entries for mask in entry.masks
    )
    return ControlMaskVerification(
        mask_manifest_sha256=manifest.manifest_sha256,
        verified_files=len(masks),
        verified_bytes=sum(mask.byte_size for mask in masks),
    )


def semantic_verification(
    base: PairArtifactManifest,
    base_receipt: PairArtifactVerification,
    masks: ControlMaskManifest,
    mask_receipt: ControlMaskVerification,
) -> MaskSemanticVerification:
    entries = []
    for entry in masks.entries:
        verified = tuple(
            item
            for item in entry.masks
            if item.review_status is MaskReviewStatus.VERIFIED
        )
        if not verified:
            continue
        entries.append(
            MaskEntrySemanticReceipt(
                base_artifact_token=entry.base_artifact_token,
                width=verified[0].width,
                height=verified[0].height,
                masks=tuple(
                    MaskPixelStats(
                        item.role,
                        10,
                        item.width * item.height,
                    )
                    for item in verified
                ),
                accessory_outside_dog_pixels=(
                    0
                    if any(
                        item.role is MaskRole.ACCESSORY
                        for item in verified
                    )
                    else None
                ),
                accessory_outside_dog_fraction=(
                    0.0
                    if any(
                        item.role is MaskRole.ACCESSORY
                        for item in verified
                    )
                    else None
                ),
            )
        )
    return MaskSemanticVerification(
        base_artifact_manifest_sha256=base.manifest_sha256,
        base_artifact_verification_sha256=content_sha256(
            base_receipt.to_dict()
        ),
        mask_manifest_sha256=masks.manifest_sha256,
        mask_file_verification_sha256=content_sha256(
            mask_receipt.to_dict()
        ),
        policy_sha256=HASH_E,
        ffmpeg_version="synthetic-test",
        entries=tuple(entries),
    )


def policy() -> VisualControlPolicy:
    kinds = (
        VisualControlKind.ORIGINAL,
        VisualControlKind.BACKGROUND_ONLY,
        VisualControlKind.ACCESSORY_ONLY,
        VisualControlKind.ACCESSORY_MASKED,
    )
    return VisualControlPolicy(
        name="shortcut-controls",
        recipes=tuple(
            VisualControlRecipe(
                kind,
                content_sha256({"recipe": kind.value}),
                "cvi-control-semantics-v1",
            )
            for kind in kinds
        ),
        panels=(
            VisualControlPanel(
                "background",
                (
                    VisualControlKind.ORIGINAL,
                    VisualControlKind.BACKGROUND_ONLY,
                ),
                minimum_matched_pairs=1,
                maximum_matched_pairs=1,
            ),
            VisualControlPanel(
                "accessory",
                (
                    VisualControlKind.ORIGINAL,
                    VisualControlKind.ACCESSORY_ONLY,
                    VisualControlKind.ACCESSORY_MASKED,
                ),
                minimum_matched_pairs=1,
                maximum_matched_pairs=10,
            ),
        ),
        seed=19,
    )


class VisualControlTests(unittest.TestCase):
    def test_panels_are_pair_matched_blind_and_cache_aware(self) -> None:
        pairs = construction()
        base = artifact_manifest(pairs)
        masks = mask_manifest(base)
        self.assertEqual(
            ControlMaskManifest.from_dict(masks.to_dict()),
            masks,
        )
        self.assertEqual(
            VisualControlPolicy.from_dict(policy().to_dict()),
            policy(),
        )
        base_receipt = base_verification(base)
        mask_receipt = mask_verification(masks)
        plan = plan_visual_control_audit(
            pairs,
            base,
            base_receipt,
            masks,
            mask_receipt,
            semantic_verification(
                base,
                base_receipt,
                masks,
                mask_receipt,
            ),
            policy(),
        )
        background, accessory = plan.panels
        self.assertEqual(background.eligible_pairs, 2)
        self.assertEqual(background.selected_pairs, 1)
        self.assertTrue(background.cap_applied)
        self.assertEqual(accessory.eligible_pairs, 1)
        self.assertEqual(accessory.selected_pairs, 1)
        self.assertTrue(accessory.minimum_met)
        self.assertIn(
            "MISSING_ACCESSORY_MASK",
            {item.reason for item in accessory.exclusions},
        )
        self.assertEqual(plan.cost.scoring_requests, 5)
        self.assertEqual(plan.cost.transform_tasks, 6)
        self.assertGreater(plan.cost.reusable_embedding_calls_saved, 0)
        self.assertEqual(
            plan,
            plan_visual_control_audit(
                pairs,
                base,
                base_receipt,
                masks,
                mask_receipt,
                semantic_verification(
                    base,
                    base_receipt,
                    masks,
                    mask_receipt,
                ),
                policy(),
            ),
        )
        scoring_text = str(plan.scoring_payload())
        self.assertNotIn("dog-", scoring_text)
        self.assertNotIn("session", scoring_text)
        self.assertNotIn("base_artifact_token", scoring_text)
        self.assertNotIn("control_kind", scoring_text)
        self.assertIn(
            "base_artifact_token",
            str(plan.protected_transform_payload()),
        )
        self.assertIn(
            "control_kind",
            str(plan.sealed_evaluation_payload()),
        )

    def test_mask_manifest_and_pair_set_must_match_exactly(self) -> None:
        pairs = construction()
        base = artifact_manifest(pairs)
        masks = mask_manifest(base)
        incomplete = ControlMaskManifest(
            base_artifact_manifest_sha256=base.manifest_sha256,
            entries=masks.entries[:-1],
        )
        with self.assertRaisesRegex(ValueError, "every base artifact"):
            plan_visual_control_audit(
                pairs,
                base,
                base_verification(base),
                incomplete,
                mask_verification(incomplete),
                semantic_verification(
                    base,
                    base_verification(base),
                    incomplete,
                    mask_verification(incomplete),
                ),
                policy(),
            )
        stale = ControlMaskManifest(
            base_artifact_manifest_sha256=HASH_E,
            entries=masks.entries,
        )
        with self.assertRaisesRegex(ValueError, "base-artifact hash"):
            plan_visual_control_audit(
                pairs,
                base,
                base_verification(base),
                stale,
                mask_verification(stale),
                semantic_verification(
                    base,
                    base_verification(base),
                    stale,
                    mask_verification(stale),
                ),
                policy(),
            )
        bad_base_receipt = PairArtifactVerification(
            artifact_manifest_sha256=base.manifest_sha256,
            verified_files=len(base.entries) - 1,
            verified_bytes=sum(
                entry.byte_size for entry in base.entries
            ),
        )
        with self.assertRaisesRegex(ValueError, "verification receipt"):
            plan_visual_control_audit(
                pairs,
                base,
                bad_base_receipt,
                masks,
                mask_verification(masks),
                semantic_verification(
                    base,
                    bad_base_receipt,
                    masks,
                    mask_verification(masks),
                ),
                policy(),
            )

    def test_insufficient_panel_is_explicit_blocker(self) -> None:
        pairs = construction()
        base = artifact_manifest(pairs)
        controls = (
            VisualControlKind.ORIGINAL,
            VisualControlKind.ACCESSORY_ONLY,
            VisualControlKind.ACCESSORY_MASKED,
        )
        strict = VisualControlPolicy(
            name="strict",
            recipes=tuple(
                recipe
                for recipe in policy().recipes
                if recipe.kind in controls
            ),
            panels=(
                VisualControlPanel(
                    "accessory",
                    controls,
                    minimum_matched_pairs=2,
                    maximum_matched_pairs=2,
                ),
            ),
            seed=19,
        )
        masks = mask_manifest(base)
        base_receipt = base_verification(base)
        mask_receipt = mask_verification(masks)
        plan = plan_visual_control_audit(
            pairs,
            base,
            base_receipt,
            masks,
            mask_receipt,
            semantic_verification(
                base,
                base_receipt,
                masks,
                mask_receipt,
            ),
            strict,
        )
        self.assertEqual(len(plan.gate_blockers), 1)
        self.assertIn("insufficient", plan.gate_blockers[0])

    def test_mask_files_are_closed_and_rehashed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = root / "mask-a-dog.png"
            second = root / "mask-b-dog.png"
            first.write_bytes(b"first-mask")
            second.write_bytes(b"second-mask")
            manifest = ControlMaskManifest(
                base_artifact_manifest_sha256=HASH_A,
                entries=(
                    ControlMaskEntry(
                        "a",
                        (
                            MaskEvidence(
                                MaskRole.DOG,
                                "mask-a-dog",
                                first.name,
                                sha256_file(first),
                                first.stat().st_size,
                                10,
                                10,
                                "v1",
                                "manual-reviewed",
                                HASH_B,
                                MaskReviewStatus.VERIFIED,
                            ),
                        ),
                    ),
                    ControlMaskEntry(
                        "b",
                        (
                            MaskEvidence(
                                MaskRole.DOG,
                                "mask-b-dog",
                                second.name,
                                sha256_file(second),
                                second.stat().st_size,
                                10,
                                10,
                                "v1",
                                "manual-reviewed",
                                HASH_B,
                                MaskReviewStatus.VERIFIED,
                            ),
                        ),
                    ),
                ),
            )
            verified = verify_control_mask_files(root, manifest)
            self.assertEqual(verified.verified_files, 2)
            (root / "extra.png").write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "entries mismatch"):
                verify_control_mask_files(root, manifest)


if __name__ == "__main__":
    unittest.main()
