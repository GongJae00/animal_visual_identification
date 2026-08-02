from __future__ import annotations

import hashlib
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from PIL import Image

from data_pipeline.public_crop_manifest import (
    PublicCropArtifact,
    PublicCropManifest,
    canonical_rgb_pixel_sha256,
    read_verified_crop_artifact,
    verify_public_crop_manifest,
)
from identity_governance.training_admission import (
    TrainingAdmissionManifest,
    TrainingAdmissionReceipt,
    TrainingCropRow,
    admit_training,
    verify_training_admission_receipt,
)
from identity_governance.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
)


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(
    root: Path,
    *,
    sample: str,
    subject: str,
    component: str,
    color: tuple[int, int, int],
    source_variant: str = "original",
) -> PublicCropArtifact:
    path = root / f"{sample}.png"
    with Image.new("RGB", (2, 2), color) as image:
        image.save(path, format="PNG")
        pixel_sha256 = canonical_rgb_pixel_sha256(
            2, 2, image.tobytes("raw", "RGB")
        )
    payload = path.read_bytes()
    return PublicCropArtifact(
        sample_token=sample,
        public_subject_token=subject,
        component_token=component,
        source_variant=source_variant,
        relative_path=path.name,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        pixel_sha256=pixel_sha256,
        width=2,
        height=2,
        mode="RGB",
        format="PNG",
    )


def _row(artifact: PublicCropArtifact, lane: str, role: str) -> TrainingCropRow:
    return TrainingCropRow(
        sample_token=artifact.sample_token,
        identity_token=_token(f"identity-{artifact.public_subject_token}"),
        public_subject_token=artifact.public_subject_token,
        component_token=artifact.component_token,
        lane=lane,
        role=role,
        crop_relative_path=artifact.relative_path,
        crop_artifact_sha256=artifact.artifact_sha256,
    )


def _exposure(rows, stage: ExposureStage = ExposureStage.BYTES_EXPORTED):
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=_token(f"exposure-{stage.value}"),
        kind=(
            ExposureDeclarationKind.PRIOR_ASSIGNMENT
            if stage in {ExposureStage.BYTES_EXPORTED, ExposureStage.MODEL_TRAINING_USED}
            else ExposureDeclarationKind.PRIOR_EVALUATION
        ),
        revoked=False,
        records=tuple(
            RoleExposureDeclarationRecord(
                sample_token=row.sample_token,
                identity_token=row.identity_token,
                public_subject_token=row.public_subject_token,
                stage=stage,
            )
            for row in rows
        ),
    )
    ledger = merge_role_exposure_declarations((declaration,))
    return ledger, create_role_exposure_receipt(ledger)


def _fixture(root: Path) -> tuple[PublicCropManifest, TrainingAdmissionManifest]:
    artifacts = tuple(sorted((
        _artifact(
            root,
            sample=_token("train-sample"),
            subject=_token("train-subject"),
            component=_token("train-component"),
            color=(255, 0, 0),
        ),
        _artifact(
            root,
            sample=_token("development-sample"),
            subject=_token("development-subject"),
            component=_token("development-component"),
            color=(0, 255, 0),
        ),
    ), key=lambda item: item.sample_token))
    crop_manifest = PublicCropManifest(artifacts)
    rows = tuple(sorted((
        _row(artifacts[0], "MODEL_SELECTION", "YT_DEVELOPMENT")
        if artifacts[0].sample_token == _token("development-sample")
        else _row(artifacts[0], "MODEL_TRAINING", "YT_FIT"),
        _row(artifacts[1], "MODEL_SELECTION", "YT_DEVELOPMENT")
        if artifacts[1].sample_token == _token("development-sample")
        else _row(artifacts[1], "MODEL_TRAINING", "YT_FIT"),
    ), key=lambda item: item.sample_token))
    _, exposure_receipt = _exposure(rows)
    admission = TrainingAdmissionManifest(
        split_receipt_sha256=_token("split-receipt"),
        crop_manifest_sha256=crop_manifest.manifest_sha256,
        crop_receipt_sha256=verify_public_crop_manifest(
            root, crop_manifest
        ).receipt_sha256,
        exposure_receipt_sha256=exposure_receipt.receipt_sha256,
        model_receipt_sha256=_token("model-receipt"),
        rows=rows,
    )
    return crop_manifest, admission


def _admit(
    root: Path,
    crops: PublicCropManifest,
    admission: TrainingAdmissionManifest,
) -> TrainingAdmissionReceipt:
    exposure_ledger, exposure_receipt = _exposure(admission.rows)
    return admit_training(
        admission,
        crops,
        crop_root=root,
        exposure_ledger=exposure_ledger,
        exposure_receipt=exposure_receipt,
        expected_sample_tokens=tuple(row.sample_token for row in admission.rows),
        expected_split_receipt_sha256=admission.split_receipt_sha256,
        expected_crop_receipt_sha256=admission.crop_receipt_sha256,
        expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
        expected_model_receipt_sha256=admission.model_receipt_sha256,
    )


class TrainingAdmissionTests(unittest.TestCase):
    def test_exact_receipt_bound_population_passes_and_round_trips(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, admission = _fixture(root)
            self.assertEqual(
                TrainingAdmissionManifest.from_dict(admission.to_dict()), admission
            )
            receipt = _admit(root, crops, admission)
            self.assertEqual(TrainingAdmissionReceipt.from_dict(receipt.to_dict()), receipt)
            self.assertEqual(receipt.state, "PASS")
            self.assertEqual(receipt.admitted_rows, 2)
            self.assertEqual(receipt.admitted_public_subjects, 2)

    def test_public_verified_read_returns_only_matching_bytes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, _ = _fixture(root)
            artifact = crops.artifacts[0]
            expected = (root / artifact.relative_path).read_bytes()
            self.assertEqual(read_verified_crop_artifact(root, artifact), expected)
            (root / artifact.relative_path).write_bytes(expected + b"tampered")
            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                read_verified_crop_artifact(root, artifact)

    def test_external_admission_receipt_and_hashes_are_required(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, admission = _fixture(root)
            receipt = _admit(root, crops, admission)
            verified = verify_training_admission_receipt(
                admission,
                crops,
                receipt,
                crop_root=root,
                exposure_ledger=_exposure(admission.rows)[0],
                exposure_receipt=_exposure(admission.rows)[1],
                expected_admission_manifest_sha256=admission.manifest_sha256,
                expected_admission_receipt_sha256=receipt.receipt_sha256,
                expected_split_receipt_sha256=admission.split_receipt_sha256,
                expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                expected_model_receipt_sha256=admission.model_receipt_sha256,
            )
            self.assertEqual(verified, receipt)
            with self.assertRaisesRegex(ValueError, "admission receipt hash mismatch"):
                verify_training_admission_receipt(
                    admission,
                    crops,
                    receipt,
                    crop_root=root,
                    exposure_ledger=_exposure(admission.rows)[0],
                    exposure_receipt=_exposure(admission.rows)[1],
                    expected_admission_manifest_sha256=admission.manifest_sha256,
                    expected_admission_receipt_sha256=_token("wrong-admission-receipt"),
                    expected_split_receipt_sha256=admission.split_receipt_sha256,
                    expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                    expected_model_receipt_sha256=admission.model_receipt_sha256,
                )

    def test_registered_identity_field_and_forbidden_roles_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            crops, admission = _fixture(Path(temporary))
            payload = admission.rows[0].to_dict()
            payload["registered_dog_id"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "unknown"):
                TrainingCropRow.from_dict(payload)
            for role in (
                "YT_CALIBRATION_KNOWN",
                "YT_TEST_KNOWN",
                "MPDD_EXTERNAL_KNOWN",
                "YT_RANDOM_BACKGROUND_FIT",
            ):
                with self.subTest(role=role):
                    with self.assertRaisesRegex(ValueError, "forbidden"):
                        replace(admission.rows[0], role=role)
            self.assertEqual(len(crops.artifacts), 2)

    def test_train_development_subject_and_component_overlap_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            _, admission = _fixture(Path(temporary))
            train = next(row for row in admission.rows if row.lane == "MODEL_TRAINING")
            development = next(
                row for row in admission.rows if row.lane == "MODEL_SELECTION"
            )
            for field in ("identity_token", "public_subject_token", "component_token"):
                changed = replace(development, **{field: getattr(train, field)})
                rows = tuple(sorted((train, changed), key=lambda row: row.sample_token))
                with self.subTest(field=field):
                    with self.assertRaisesRegex(ValueError, "tokens overlap"):
                        replace(admission, rows=rows)

    def test_missing_extra_duplicate_and_artifact_rebinding_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, admission = _fixture(root)
            with self.assertRaisesRegex(ValueError, "expected/training sample tokens"):
                exposure_ledger, exposure_receipt = _exposure(admission.rows)
                admit_training(
                    admission,
                    crops,
                    crop_root=root,
                    exposure_ledger=exposure_ledger,
                    exposure_receipt=exposure_receipt,
                    expected_sample_tokens=(admission.rows[0].sample_token,),
                    expected_split_receipt_sha256=admission.split_receipt_sha256,
                    expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                    expected_model_receipt_sha256=admission.model_receipt_sha256,
                )
            with self.assertRaisesRegex(ValueError, "duplicate expected sample tokens"):
                admit_training(
                    admission,
                    crops,
                    crop_root=root,
                    exposure_ledger=exposure_ledger,
                    exposure_receipt=exposure_receipt,
                    expected_sample_tokens=(
                        admission.rows[0].sample_token,
                        admission.rows[0].sample_token,
                    ),
                    expected_split_receipt_sha256=admission.split_receipt_sha256,
                    expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                    expected_model_receipt_sha256=admission.model_receipt_sha256,
                )
            changed = replace(
                admission.rows[0], crop_artifact_sha256=_token("other-artifact")
            )
            rows = tuple(sorted((changed, admission.rows[1]), key=lambda row: row.sample_token))
            with self.assertRaisesRegex(ValueError, "crop binding mismatch"):
                _admit(root, crops, replace(admission, rows=rows))

    def test_receipt_hashes_and_random_background_crops_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, admission = _fixture(root)
            expected = {
                "expected_split_receipt_sha256": admission.split_receipt_sha256,
                "expected_crop_receipt_sha256": admission.crop_receipt_sha256,
                "expected_exposure_receipt_sha256": admission.exposure_receipt_sha256,
                "expected_model_receipt_sha256": admission.model_receipt_sha256,
            }
            exposure_ledger, exposure_receipt = _exposure(admission.rows)
            for argument, label in (
                ("expected_split_receipt_sha256", "split"),
                ("expected_crop_receipt_sha256", "crop"),
                ("expected_exposure_receipt_sha256", "exposure"),
                ("expected_model_receipt_sha256", "model"),
            ):
                mismatched = dict(expected)
                mismatched[argument] = _token(f"wrong-{label}")
                with self.subTest(receipt=label):
                    with self.assertRaisesRegex(
                        ValueError, f"{label} receipt hash mismatch"
                    ):
                        admit_training(
                            admission,
                            crops,
                            crop_root=root,
                            exposure_ledger=exposure_ledger,
                            exposure_receipt=exposure_receipt,
                            expected_sample_tokens=tuple(
                                row.sample_token for row in admission.rows
                            ),
                            **mismatched,
                        )

            changed_artifact = replace(crops.artifacts[0], source_variant="random_background")
            changed_artifacts = tuple(sorted(
                (changed_artifact, crops.artifacts[1]),
                key=lambda item: item.sample_token,
            ))
            changed_crops = PublicCropManifest(changed_artifacts)
            changed_rows = tuple(sorted((
                _row(
                    changed_artifact,
                    admission.rows[0].lane,
                    admission.rows[0].role,
                ),
                admission.rows[1],
            ), key=lambda row: row.sample_token))
            changed_admission = replace(
                admission,
                crop_manifest_sha256=changed_crops.manifest_sha256,
                crop_receipt_sha256=verify_public_crop_manifest(
                    root, changed_crops
                ).receipt_sha256,
                rows=changed_rows,
            )
            with self.assertRaisesRegex(ValueError, "non-original crops are forbidden"):
                _admit(root, changed_crops, changed_admission)

    def test_historical_final_test_exposure_cannot_regress_to_training(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            crops, admission = _fixture(root)
            ledger, receipt = _exposure(
                admission.rows, ExposureStage.FINAL_TEST_SCORED
            )
            changed = replace(
                admission, exposure_receipt_sha256=receipt.receipt_sha256
            )
            with self.assertRaisesRegex(ValueError, "candidate role regression"):
                admit_training(
                    changed,
                    crops,
                    crop_root=root,
                    exposure_ledger=ledger,
                    exposure_receipt=receipt,
                    expected_sample_tokens=tuple(
                        row.sample_token for row in changed.rows
                    ),
                    expected_split_receipt_sha256=changed.split_receipt_sha256,
                    expected_crop_receipt_sha256=changed.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=receipt.receipt_sha256,
                    expected_model_receipt_sha256=changed.model_receipt_sha256,
                )


if __name__ == "__main__":
    unittest.main()
