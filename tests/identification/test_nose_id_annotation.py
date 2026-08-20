from __future__ import annotations

import json
import tempfile
import uuid
from copy import deepcopy
from hashlib import sha256
from pathlib import Path

import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper
from PIL import Image

from shared.contracts.artifact_manifest import (
    ArtifactLicense,
    ImagePreprocessing,
    NoseDetectorManifest,
    UsageLane,
)
from shared.foundation.provenance import content_sha256
from identification.export.nose.data.annotation import (
    ACQUISITION_SCHEMA,
    ANNOTATION_SCHEMA,
    INVALID_MASK_CLASSES,
    SEMANTIC_MASK_CLASSES,
    AcquisitionRecord,
    AnnotationRecord,
    build_admission_receipt,
    canonical_jsonl_bytes,
    load_acquisition_jsonl,
    validate_acquisition_records,
    validate_annotation_records,
)
from identification.export.nose.types import NOSE_KEYPOINTS
from archive.nose.commands.prepare_nose_annotation_batch import (
    create_review_batch,
    validate_completed_batch,
)

class TestNoseAnnotationContracts:
    def setup_method(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.data_root = self.root / "data"
        self.annotation_root = self.root / "annotations"
        (self.data_root / "images").mkdir(parents=True)
        self.annotation_root.mkdir()
        self.identity = str(uuid.uuid5(uuid.NAMESPACE_DNS, "nose-annotation-dog"))
        self.images: list[tuple[Path, str]] = []
        for name, color in (("first.png", (80, 90, 100)), ("second.png", (90, 100, 110))):
            path = self.data_root / "images" / name
            Image.new("RGB", (512, 512), color).save(path)
            self.images.append((path, sha256(path.read_bytes()).hexdigest()))
        self.acquisition_payloads = [
            self._acquisition_payload(0, "session-a1", "capture-a1"),
            self._acquisition_payload(1, "session-a2", "capture-a2"),
        ]
        self.acquisitions = tuple(
            AcquisitionRecord.from_dict(payload) for payload in self.acquisition_payloads
        )

    def teardown_method(self) -> None:
        self.temporary.cleanup()

    def _acquisition_payload(
        self, index: int, session_id: str, capture_id: str
    ) -> dict[str, object]:
        path, digest = self.images[index]
        return {
            "schema_version": ACQUISITION_SCHEMA,
            "sample_id": f"nose-sample-{index + 1}",
            "registered_dog_id": self.identity,
            "session_id": session_id,
            "camera_id": "camera-main",
            "capture_id": capture_id,
            "original_image": {
                "relative_path": path.relative_to(self.data_root).as_posix(),
                "sha256": digest,
                "width": 512,
                "height": 512,
            },
            "consent": {
                "token": "consent-token-1",
                "usage_lane": "RESEARCH_ONLY",
            },
            "license": {
                "license_id": "LicenseRef-Private-Consent",
                "usage_lane": "RESEARCH_ONLY",
            },
            "split_role": "TRAIN",
        }

    def _write_masks(self, stem: str, *, invalid_value: int = 0) -> tuple[Path, Path]:
        semantic = np.zeros((224, 224), dtype=np.uint8)
        semantic[16:208, 16:208] = 1
        semantic[80:144, 72:152] = 2
        invalid = np.full((224, 224), invalid_value, dtype=np.uint8)
        semantic_path = self.annotation_root / f"{stem}-semantic.png"
        invalid_path = self.annotation_root / f"{stem}-invalid.png"
        Image.fromarray(semantic, mode="L").save(semantic_path)
        Image.fromarray(invalid, mode="L").save(invalid_path)
        return semantic_path, invalid_path

    def _annotation_payload(
        self, acquisition: AcquisitionRecord, stem: str
    ) -> dict[str, object]:
        semantic, invalid = self._write_masks(stem)
        return {
            "schema_version": ANNOTATION_SCHEMA,
            "sample_id": acquisition.sample_id,
            "acquisition_sha256": acquisition.record_sha256,
            "nose_bbox_xyxy": [100, 100, 324, 324],
            "native_nose_short_side": 224,
            "nose_points": {
                "order": list(NOSE_KEYPOINTS),
                "xy": [
                    [160.0, 215.0],
                    [264.0, 215.0],
                    [212.0, 130.0],
                    [212.0, 305.0],
                    [125.0, 225.0],
                    [299.0, 225.0],
                ],
                "visibility": [2, 2, 2, 2, 1, 1],
            },
            "semantic_mask": {
                "relative_path": semantic.name,
                "sha256": sha256(semantic.read_bytes()).hexdigest(),
                "box_xyxy": [100, 100, 324, 324],
                "classes": dict(SEMANTIC_MASK_CLASSES),
            },
            "invalid_mask": {
                "relative_path": invalid.name,
                "sha256": sha256(invalid.read_bytes()).hexdigest(),
                "box_xyxy": [100, 100, 324, 324],
                "classes": dict(INVALID_MASK_CLASSES),
            },
            "annotator_token": "annotator-token-a",
            "reviewer_token": "reviewer-token-b",
            "review_status": "APPROVED",
        }

    def test_acquisition_contract_rejects_extra_keys_uuid_paths_and_reused_ids(self) -> None:
        payload = deepcopy(self.acquisition_payloads[0])
        payload["extra"] = True
        with pytest.raises(ValueError, match="keys differ"):
            AcquisitionRecord.from_dict(payload)

        payload = deepcopy(self.acquisition_payloads[0])
        payload["registered_dog_id"] = str(uuid.uuid4())
        with pytest.raises(ValueError, match="UUIDv5"):
            AcquisitionRecord.from_dict(payload)

        payload = deepcopy(self.acquisition_payloads[0])
        payload["original_image"]["relative_path"] = "../first.png"  # type: ignore[index]
        with pytest.raises(ValueError, match="relative path"):
            AcquisitionRecord.from_dict(payload)

        payload = deepcopy(self.acquisition_payloads[0])
        payload["capture_id"] = payload["session_id"]
        with pytest.raises(ValueError, match="must be distinct"):
            AcquisitionRecord.from_dict(payload)

    def test_acquisition_validation_authenticates_images_and_enforces_split_policy(self) -> None:
        validate_acquisition_records(self.acquisitions, self.data_root)

        one_session = (self.acquisitions[0],)
        with pytest.raises(ValueError, match="at least two distinct sessions"):
            validate_acquisition_records(one_session, self.data_root)

        dev_payload = deepcopy(self.acquisition_payloads[1])
        dev_payload["split_role"] = "DEV"
        dev = AcquisitionRecord.from_dict(dev_payload)
        with pytest.raises(ValueError, match="disjoint"):
            validate_acquisition_records((self.acquisitions[0], dev), self.data_root)

        wrong_hash = deepcopy(self.acquisition_payloads[1])
        wrong_hash["original_image"]["sha256"] = "0" * 64  # type: ignore[index]
        with pytest.raises(ValueError, match="SHA256 differs"):
            validate_acquisition_records(
                (self.acquisitions[0], AcquisitionRecord.from_dict(wrong_hash)),
                self.data_root,
            )

    def test_acquisition_validation_rejects_symlinked_path_components(self) -> None:
        target = self.data_root / "real"
        target.mkdir()
        copied = target / "linked.png"
        copied.write_bytes(self.images[0][0].read_bytes())
        (self.data_root / "linked").symlink_to(target, target_is_directory=True)
        first = deepcopy(self.acquisition_payloads[0])
        first["original_image"]["relative_path"] = "linked/linked.png"  # type: ignore[index]
        first["original_image"]["sha256"] = sha256(copied.read_bytes()).hexdigest()  # type: ignore[index]
        linked_record = AcquisitionRecord.from_dict(first)
        with pytest.raises(ValueError, match="real directory"):
            validate_acquisition_records((linked_record, self.acquisitions[1]), self.data_root)

    def test_annotation_contract_and_masks_fail_closed(self) -> None:
        payloads = [
            self._annotation_payload(self.acquisitions[0], "first"),
            self._annotation_payload(self.acquisitions[1], "second"),
        ]
        annotations = tuple(AnnotationRecord.from_dict(payload) for payload in payloads)
        validate_annotation_records(
            self.acquisitions,
            annotations,
            data_root=self.data_root,
            annotation_root=self.annotation_root,
        )

        too_small = deepcopy(payloads[0])
        too_small["nose_bbox_xyxy"] = [100, 100, 323, 324]
        too_small["native_nose_short_side"] = 223
        with pytest.raises(ValueError, match="at least 224"):
            AnnotationRecord.from_dict(too_small)

        outside = deepcopy(payloads[0])
        outside["nose_points"]["xy"][0] = [512.0, 10.0]  # type: ignore[index]
        outside_annotation = AnnotationRecord.from_dict(outside)
        with pytest.raises(ValueError, match="outside original image"):
            validate_annotation_records(
                self.acquisitions,
                (outside_annotation, annotations[1]),
                data_root=self.data_root,
                annotation_root=self.annotation_root,
            )

        bad_mask = np.full((224, 224), 4, dtype=np.uint8)
        invalid_path = self.annotation_root / "bad-invalid.png"
        Image.fromarray(bad_mask, mode="L").save(invalid_path)
        unknown = deepcopy(payloads[0])
        unknown["invalid_mask"]["relative_path"] = invalid_path.name  # type: ignore[index]
        unknown["invalid_mask"]["sha256"] = sha256(invalid_path.read_bytes()).hexdigest()  # type: ignore[index]
        with pytest.raises(ValueError, match="unknown class value"):
            validate_annotation_records(
                self.acquisitions,
                (AnnotationRecord.from_dict(unknown), annotations[1]),
                data_root=self.data_root,
                annotation_root=self.annotation_root,
            )

    def test_mask_shape_hash_and_symlink_are_rejected(self) -> None:
        first_payload = self._annotation_payload(self.acquisitions[0], "shape-first")
        second = AnnotationRecord.from_dict(
            self._annotation_payload(self.acquisitions[1], "shape-second")
        )
        wrong_shape_path = self.annotation_root / "wrong-shape.png"
        Image.fromarray(np.zeros((223, 224), dtype=np.uint8), mode="L").save(
            wrong_shape_path
        )
        first_payload["semantic_mask"]["relative_path"] = wrong_shape_path.name  # type: ignore[index]
        first_payload["semantic_mask"]["sha256"] = sha256(wrong_shape_path.read_bytes()).hexdigest()  # type: ignore[index]
        with pytest.raises(ValueError, match="shape"):
            validate_annotation_records(
                self.acquisitions,
                (AnnotationRecord.from_dict(first_payload), second),
                data_root=self.data_root,
                annotation_root=self.annotation_root,
            )

        target = self.annotation_root / "target-mask.png"
        Image.fromarray(np.zeros((224, 224), dtype=np.uint8), mode="L").save(target)
        symlink = self.annotation_root / "linked-mask.png"
        symlink.symlink_to(target)
        first_payload["semantic_mask"]["relative_path"] = symlink.name  # type: ignore[index]
        first_payload["semantic_mask"]["sha256"] = sha256(target.read_bytes()).hexdigest()  # type: ignore[index]
        with pytest.raises(ValueError, match="symlink"):
            validate_annotation_records(
                self.acquisitions,
                (AnnotationRecord.from_dict(first_payload), second),
                data_root=self.data_root,
                annotation_root=self.annotation_root,
            )

    def test_jsonl_duplicate_keys_and_receipt_bind_exact_records(self) -> None:
        acquisitions_path = self.root / "acquisitions.jsonl"
        acquisitions_path.write_bytes(canonical_jsonl_bytes(self.acquisitions))
        assert load_acquisition_jsonl(acquisitions_path) == self.acquisitions
        duplicate = self.root / "duplicate.jsonl"
        duplicate.write_text(
            '{"schema_version":"x","schema_version":"y"}\n', encoding="utf-8"
        )
        with pytest.raises(ValueError, match="duplicate JSON object key"):
            load_acquisition_jsonl(duplicate)

        annotations = tuple(
            AnnotationRecord.from_dict(self._annotation_payload(acquisition, f"receipt-{index}"))
            for index, acquisition in enumerate(self.acquisitions)
        )
        receipt = build_admission_receipt(
            self.acquisitions,
            annotations,
            acquisition_jsonl_sha256=sha256(canonical_jsonl_bytes(self.acquisitions)).hexdigest(),
            annotation_jsonl_sha256=sha256(canonical_jsonl_bytes(annotations)).hexdigest(),
            batch_manifest_sha256="a" * 64,
        )
        assert receipt["admitted_count"] == 2
        assert receipt["split_role_counts"]["TRAIN"] == 2
        assert receipt["records"][0]["annotation_sha256"] == annotations[0].record_sha256

    def test_real_detector_batch_stays_blank_until_completed_validation(self) -> None:
        acquisitions_path = self.root / "input-acquisitions.jsonl"
        acquisitions_path.write_bytes(canonical_jsonl_bytes(self.acquisitions))
        detector_path = self.root / "detector.onnx"
        self._write_detector(detector_path)
        manifest = self._detector_manifest(detector_path)
        manifest_path = self.root / "detector.json"
        manifest_path.write_text(json.dumps(manifest.to_dict()), encoding="utf-8")
        batch_dir = self.root / "review-batch"

        batch = create_review_batch(
            acquisitions_path=acquisitions_path,
            data_root=self.data_root,
            detector_artifact=detector_path,
            detector_manifest_path=manifest_path,
            output_dir=batch_dir,
        )

        assert batch["admission_status"].startswith("NOT_ADMITTED")
        assert batch["annotation_templates"]["contains_admitted_labels"] is False
        template_rows = [
            json.loads(line)
            for line in (batch_dir / "annotation-templates.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
        ]
        assert all(row["completed_annotation"] is None for row in template_rows)
        assert all(
            row["localizer_prediction"]["source"].endswith("NOT_A_LABEL")
            for row in template_rows
        )
        assert len(tuple((batch_dir / "predicted-crops").iterdir())) == 2

        annotations = tuple(
            AnnotationRecord.from_dict(self._annotation_payload(acquisition, f"cli-{index}"))
            for index, acquisition in enumerate(self.acquisitions)
        )
        completed_path = self.root / "completed.jsonl"
        completed_path.write_bytes(canonical_jsonl_bytes(annotations))
        admitted_dir = self.root / "admitted"
        receipt = validate_completed_batch(
            batch_dir=batch_dir,
            completed_annotations=completed_path,
            data_root=self.data_root,
            annotation_root=self.annotation_root,
            output_dir=admitted_dir,
        )
        assert receipt["decision"] == "ADMITTED_VERIFIED_HUMAN_ANNOTATIONS"
        assert receipt["admitted_count"] == 2
        persisted = json.loads(
            (admitted_dir / "admission-receipt.json").read_text(encoding="utf-8")
        )
        assert persisted == receipt
        assert persisted["batch_manifest_sha256"] == content_sha256(batch)

    @staticmethod
    def _write_detector(path: Path) -> None:
        output = np.asarray(
            [[[100 / 512, 100 / 512, 324 / 512, 324 / 512, 0.95]]],
            dtype=np.float32,
        )
        value = numpy_helper.from_array(output, "constant_value")
        graph = helper.make_graph(
            [helper.make_node("Constant", [], ["output"], value=value)],
            "nose-annotation-detector",
            [helper.make_tensor_value_info("input", TensorProto.FLOAT, (1, 3, 8, 8))],
            [helper.make_tensor_value_info("output", TensorProto.FLOAT, output.shape)],
        )
        onnx.save(
            helper.make_model(graph, opset_imports=[helper.make_opsetid("", 18)]), path
        )

    @staticmethod
    def _detector_manifest(path: Path) -> NoseDetectorManifest:
        return NoseDetectorManifest(
            artifact_id="nose-annotation-test-detector",
            artifact_sha256=sha256(path.read_bytes()).hexdigest(),
            input_name="input",
            input_shape=(1, 3, 8, 8),
            output_name="output",
            output_shape=(1, 1, 5),
            license=ArtifactLicense("LicenseRef-Test-Fixture", UsageLane.TEST_FIXTURE),
            preprocessing=ImagePreprocessing(
                color_mode="RGB",
                layout="NCHW",
                dtype="float32",
                resize="bilinear",
                scale=1.0 / 255.0,
                mean=(0.0, 0.0, 0.0),
                std=(1.0, 1.0, 1.0),
                clahe=None,
            ),
            confidence_threshold=0.5,
        )
