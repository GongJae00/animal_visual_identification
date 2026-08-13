from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

import numpy as np
from PIL import Image

from contracts.three_region_artifact import (
    completion_for_record,
    read_three_region_artifact,
    validate_three_region_artifact_bundle,
)
from data.types import CaptureGroupKind, UnifiedCanidSample
from foundation.protected_io import write_private_json_bundle
from foundation.provenance import content_sha256
from identity_governance.identity_registry import compute_registered_dog_id
from localization.prediction_cache import build_prediction_cache
from localization.roi_manifest import build_roi_manifest
from localization.types import DetectionBox, Keypoint, KeypointSet, LocalizationResult
from workflows.export_three_region_artifacts import export_three_region_artifacts


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _build_roi_fixture(root: Path) -> Path:
    image_path = root / "source.png"
    Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
    sample = UnifiedCanidSample(
        sample_id="sample-1",
        dataset_name="fixture",
        dataset_version="v1",
        source_group_id="source-group-1",
        image_path=image_path.name,
        image_sha256=_sha256(image_path),
        width=100,
        height=60,
        registered_identity_id=compute_registered_dog_id("fixture:v1:dog:1"),
        raw_identity_id="dog-1",
        capture_group_id="capture-1",
        capture_group_kind=CaptureGroupKind.REAL_CAMERA_SESSION,
        split_role="TRAIN",
    )
    pose = KeypointSet(
        {
            "left_eye": Keypoint(30, 15, 0.9),
            "right_eye": Keypoint(50, 15, 0.9),
            "nose_center": Keypoint(40, 25, 0.9),
            "neck": Keypoint(40, 35, 0.8),
        },
        "ap10k-dog-17",
    )
    result = LocalizationResult(
        image_id=sample.sample_id,
        dog_boxes=(DetectionBox(10, 5, 90, 55, 0.9, 16, "dog"),),
        face_boxes=(),
        nose_boxes=(),
        body_keypoints=(pose,),
        face_landmarks=(),
        model_name="fixture-pose",
        model_family="fixture",
        inference_ms=1.0,
    )
    cache = build_prediction_cache(
        (sample,),
        (result,),
        model={
            "family": "fixture",
            "name": "fixture-pose",
            "artifact_sha256": "a" * 64,
            "artifact_size_bytes": 1,
            "license_id": "TEST",
            "device": "cpu",
        },
    )["cache"]
    roi_root = root / "roi"
    bundle = build_roi_manifest(
        (sample,), (cache,), data_root=root, output_dir=roi_root
    )
    roi_path = roi_root / "roi_manifest.json"
    write_private_json_bundle(((roi_path, bundle),))
    return roi_path


def _semantic_mask(
    root: Path,
    name: str,
    *,
    record: dict[str, object],
    values: tuple[int, ...],
    target: str,
    class_map: dict[str, str],
) -> dict[str, object]:
    array = np.zeros((60, 100), dtype=np.uint8)
    stripe = max(1, array.shape[1] // len(values))
    for index, value in enumerate(values):
        array[:, index * stripe : (index + 1) * stripe] = value
    path = root / name
    Image.fromarray(array, mode="L").save(path)
    payload = path.read_bytes()
    artifact = {
        "relative_path": name,
        "sha256": hashlib.sha256(payload).hexdigest(),
        "byte_size": len(payload),
        "width": 100,
        "height": 60,
        "encoding": "PNG",
        "pixel_mode": "L",
        "coordinate_space": "SOURCE_IMAGE_PIXELS",
        "observed_pixel_values": list(values),
        "source_mapping": None,
    }
    source = record["source"]
    receipt = {
        "schema_version": "cvi.semantic_mask_review_receipt.v1",
        "decision": "VERIFIED",
        "sample_id": record["sample_id"],
        "instance_id": record["instance_id"],
        "source_image_sha256": source["image_sha256"],
        "semantic_target": target,
        "mask_artifact_sha256": artifact["sha256"],
        "mask_width": artifact["width"],
        "mask_height": artifact["height"],
        "coordinate_space": artifact["coordinate_space"],
        "source_mapping_sha256": content_sha256(artifact["source_mapping"]),
        "class_map_sha256": content_sha256(class_map),
        "review_reference_sha256": "c" * 64,
        "pixel_verification_reference_sha256": "d" * 64,
    }
    receipt_name = f"{name}.review.json"
    receipt_path = root / receipt_name
    receipt_path.write_text(
        json.dumps(receipt, sort_keys=True, indent=2) + "\n", encoding="utf-8"
    )
    receipt_digest = hashlib.sha256(receipt_path.read_bytes()).hexdigest()
    return {
        "state": "AVAILABLE",
        "qualification": "VERIFIED_SEMANTIC",
        "semantic_target": target,
        "artifact": artifact,
        "class_map": class_map,
        "producer_reference_sha256": "b" * 64,
        "review_reference_sha256": "c" * 64,
        "pixel_verification_reference_sha256": "d" * 64,
        "verification_binding_sha256": content_sha256(
            {
                "semantic_target": target,
                "sample_id": record["sample_id"],
                "instance_id": record["instance_id"],
                "source_image_sha256": source["image_sha256"],
                "artifact": {
                    key: artifact[key]
                    for key in (
                        "sha256",
                        "width",
                        "height",
                        "coordinate_space",
                        "source_mapping",
                    )
                },
                "class_map": class_map,
                "review_reference_sha256": "c" * 64,
                "pixel_verification_reference_sha256": "d" * 64,
                "review_receipt_sha256": receipt_digest,
            }
        ),
        "review_receipt_path": receipt_name,
        "review_receipt_sha256": receipt_digest,
    }


class ThreeRegionArtifactTests(unittest.TestCase):
    def test_roi_export_is_explicitly_incomplete_and_nonsemantic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            summary = export_three_region_artifacts(roi_path, output)
            manifest = read_three_region_artifact(output)
            record = manifest["records"][0]

            self.assertEqual(summary["complete_records"], 0)
            self.assertEqual(record["completion"]["state"], "INCOMPLETE_REQUIRED_EVIDENCE")
            self.assertEqual(
                record["regions"]["A"]["source_validity_mask"]["qualification"],
                "SOURCE_VALIDITY",
            )
            self.assertEqual(
                record["regions"]["A"]["semantic_mask"],
                {"state": "UNAVAILABLE", "reason": "GENERATOR_NOT_CONFIGURED"},
            )
            self.assertIn(
                "A.SCHEMA_BOUND_SKELETON", record["completion"]["missing"]
            )
            self.assertIn(
                "F.EAR_FACE_NECK_LANDMARKS", record["completion"]["missing"]
            )
            self.assertIn("N.NATIVE_GEOMETRY", record["completion"]["missing"])

    def test_source_validity_cannot_be_relabelled_as_semantic(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            record = bundle["manifest"]["records"][0]
            record["regions"]["A"]["semantic_mask"] = deepcopy(
                record["regions"]["A"]["source_validity_mask"]
            )
            record["regions"]["A"]["semantic_mask"]["semantic_target"] = (
                "FULL_BODY_DOG"
            )
            record["completion"] = completion_for_record(record)
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])
            with self.assertRaisesRegex(ValueError, "source-validity mask cannot"):
                validate_three_region_artifact_bundle(bundle, root=output.parent)

    def test_completion_cannot_be_asserted_without_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            bundle["manifest"]["records"][0]["completion"] = {
                "state": "COMPLETE",
                "missing": [],
            }
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])
            with self.assertRaisesRegex(ValueError, "completion differs"):
                validate_three_region_artifact_bundle(bundle, root=output.parent)

    def test_referenced_artifact_tamper_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            manifest = read_three_region_artifact(output)
            relative = manifest["records"][0]["regions"]["A"][
                "source_validity_mask"
            ]["artifact"]["relative_path"]
            with (output.parent / relative).open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "byte size differs|byte binding differs"):
                read_three_region_artifact(output)

    def test_complete_state_requires_distinct_verified_masks_and_geometry(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            record = bundle["manifest"]["records"][0]
            record["regions"]["A"]["semantic_mask"] = _semantic_mask(
                output.parent,
                "body_semantic.png",
                record=record,
                values=(0, 1),
                target="FULL_BODY_DOG",
                class_map={"0": "background", "1": "dog"},
            )
            record["regions"]["A"]["skeleton"]["schema"] = "ap10k-dog-17"
            record["regions"]["F"]["semantic_mask"] = _semantic_mask(
                output.parent,
                "face_semantic.png",
                record=record,
                values=(0, 1, 2, 3),
                target="EARS_FACE_NECK",
                class_map={
                    "0": "background",
                    "1": "ears",
                    "2": "face",
                    "3": "neck",
                },
            )
            record["regions"]["F"]["landmarks"] = {
                "state": "AVAILABLE",
                "qualification": "VERIFIED_ANNOTATION",
                "kind": "LANDMARKS",
                "schema": "fixture-face-landmarks.v1",
                "coordinate_space": "SOURCE_IMAGE_PIXELS",
                "payload": {
                    "points": {
                        "left_ear": [25.0, 10.0, 1.0],
                        "face_center": [40.0, 20.0, 1.0],
                        "neck": [40.0, 35.0, 1.0],
                    },
                    "coverage": ["EARS", "FACE", "NECK"],
                },
                "producer_reference_sha256": "e" * 64,
            }
            record["regions"]["N"]["semantic_mask"] = _semantic_mask(
                output.parent,
                "nose_semantic.png",
                record=record,
                values=(0, 1, 2),
                target="NOSE",
                class_map={
                    "0": "context",
                    "1": "nasal_surface",
                    "2": "nostril",
                },
            )
            record["regions"]["N"]["native_geometry"] = {
                "state": "AVAILABLE",
                "qualification": "VERIFIED_ANNOTATION",
                "kind": "NATIVE_NOSE_GEOMETRY",
                "schema": "fixture-native-nose.v1",
                "coordinate_space": "SOURCE_IMAGE_PIXELS",
                "payload": {
                    "source_box_xyxy": [30, 20, 50, 40],
                    "crop_width": 20,
                    "crop_height": 20,
                    "keypoints": {"nose_center": [40.0, 30.0, 1.0]},
                },
                "producer_reference_sha256": "f" * 64,
            }
            record["completion"] = completion_for_record(record)
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])

            validate_three_region_artifact_bundle(bundle, root=output.parent)
            self.assertEqual(record["completion"], {"state": "COMPLETE", "missing": []})

    def test_verified_masks_require_foreground_and_cannot_be_reused_across_regions(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            record = bundle["manifest"]["records"][0]
            empty = _semantic_mask(
                output.parent,
                "empty_semantic.png",
                record=record,
                values=(0,),
                target="FULL_BODY_DOG",
                class_map={"0": "background", "1": "dog"},
            )
            record["regions"]["A"]["semantic_mask"] = empty
            record["completion"] = completion_for_record(record)
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])
            with self.assertRaisesRegex(ValueError, "lacks required foreground"):
                validate_three_region_artifact_bundle(bundle, root=output.parent)

    def test_source_validity_pixels_must_match_declared_crop_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            roi_path = _build_roi_fixture(Path(temporary))
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            bundle = json.loads(output.read_text(encoding="utf-8"))
            evidence = bundle["manifest"]["records"][0]["regions"]["A"][
                "source_validity_mask"
            ]
            artifact = evidence["artifact"]
            path = output.parent / artifact["relative_path"]
            Image.new("L", (artifact["width"], artifact["height"]), 255).save(path)
            payload = path.read_bytes()
            artifact["sha256"] = hashlib.sha256(payload).hexdigest()
            artifact["byte_size"] = len(payload)
            artifact["observed_pixel_values"] = [255]
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])
            with self.assertRaisesRegex(ValueError, "pixels differ from source mapping"):
                validate_three_region_artifact_bundle(bundle, root=output.parent)

    def test_export_refuses_overwrite_and_cross_root_output(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            roi_path = _build_roi_fixture(root)
            output = roi_path.parent / "three_region.json"
            export_three_region_artifacts(roi_path, output)
            with self.assertRaises(FileExistsError):
                export_three_region_artifacts(root / "missing.json", output)
            with self.assertRaisesRegex(ValueError, "share the ROI manifest"):
                export_three_region_artifacts(roi_path, root / "elsewhere.json")


if __name__ == "__main__":
    unittest.main()
