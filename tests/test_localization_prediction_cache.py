from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path

from PIL import Image

from cvi.canid_data.types import CaptureGroupKind, UnifiedCanidSample
from cvi.identity_registry import compute_registered_dog_id
from cvi.localization.prediction_cache import (
    build_prediction_cache,
    validate_prediction_cache,
)
from cvi.localization.roi_manifest import build_roi_manifest, read_roi_manifest
from cvi.localization.types import (
    DetectionBox,
    Keypoint,
    KeypointSet,
    LocalizationResult,
)
from cvi.provenance import content_sha256


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _write_bundle(path: Path, bundle: dict[str, object]) -> None:
    path.write_text(json.dumps(bundle), encoding="utf-8")


def _sample(path: Path, digest: str) -> UnifiedCanidSample:
    return UnifiedCanidSample(
        sample_id="sample-1",
        dataset_name="fixture",
        dataset_version="v1",
        source_group_id="source-1",
        image_path=path.name,
        image_sha256=digest,
        width=100,
        height=60,
        registered_identity_id=compute_registered_dog_id("fixture:v1:dog:1"),
        raw_identity_id="dog-1",
        capture_group_id="capture-1",
        capture_group_kind=CaptureGroupKind.REAL_CAMERA_SESSION,
        split_role="TRAIN",
    )


def _result(
    boxes: tuple[DetectionBox, ...] | None = None,
    body_keypoints: tuple[KeypointSet, ...] = (),
) -> LocalizationResult:
    return LocalizationResult(
        image_id="sample-1",
        dog_boxes=boxes or (DetectionBox(10, 5, 90, 55, 0.9, 16, "dog"),),
        face_boxes=(),
        nose_boxes=(),
        body_keypoints=body_keypoints,
        face_landmarks=(),
        model_name="detector",
        model_family="fixture",
        inference_ms=1.5,
    )


def _model(*, digest: str = "b" * 64) -> dict[str, object]:
    return {
        "family": "fixture",
        "name": "detector",
        "artifact_sha256": digest,
        "artifact_size_bytes": 123,
        "license_id": "Unknown",
        "device": "cpu",
    }


class PredictionCacheTests(unittest.TestCase):
    def test_cache_and_roi_manifest_round_trip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            digest = _sha256(image_path)
            sample = _sample(image_path, digest)
            bundle = build_prediction_cache(
                (sample,),
                (_result(),),
                model=_model(),
            )
            validate_prediction_cache(bundle["cache"])
            output = root / "roi"
            manifest = build_roi_manifest(
                (sample,), (bundle["cache"],), data_root=root, output_dir=output
            )
            record = manifest["manifest"]["records"][0]
            self.assertEqual(manifest["manifest"]["source_sample_ids"], ["sample-1"])
            self.assertEqual(record["review_state"], "REVIEW")
            self.assertIsNone(record["face_roi_xyxy"])
            self.assertEqual(
                manifest["schema_version"], "cvi.canid_roi_manifest_bundle.v2"
            )
            self.assertEqual(
                record["dog_crop_sha256"], _sha256(output / record["dog_crop_path"])
            )
            self.assertEqual(
                record["source_valid_mask_sha256"],
                _sha256(output / record["source_valid_mask_path"]),
            )
            self.assertTrue((output / record["dog_crop_path"]).is_file())
            self.assertTrue((output / record["source_valid_mask_path"]).is_file())
            manifest_path = output / "roi_manifest.json"
            _write_bundle(manifest_path, manifest)
            self.assertEqual(read_roi_manifest(manifest_path)["records"], [record])

    def test_cache_rejects_duplicate_result_coverage(self) -> None:
        sample = _sample(Path("image.png"), "a" * 64)
        with self.assertRaisesRegex(ValueError, "exactly cover"):
            build_prediction_cache((sample,), (), model=_model())

    def test_cache_rejects_malformed_nested_objects(self) -> None:
        sample = _sample(Path("image.png"), "a" * 64)
        valid = build_prediction_cache((sample,), (_result(),), model=_model())["cache"]

        malformed = deepcopy(valid)
        malformed["model"].pop("artifact_sha256")
        with self.assertRaisesRegex(ValueError, "model schema"):
            validate_prediction_cache(malformed)

        malformed = deepcopy(valid)
        malformed["records"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "record schema"):
            validate_prediction_cache(malformed)

        malformed = deepcopy(valid)
        malformed["records"][0]["dog_boxes"][0]["unexpected"] = True
        with self.assertRaisesRegex(ValueError, "box schema"):
            validate_prediction_cache(malformed)

        malformed = deepcopy(valid)
        malformed["records"][0]["body_keypoints"] = [
            {
                "schema": "fixture-points",
                "points": {"nose": [10.0, 10.0, 0.9, 1.0]},
            }
        ]
        with self.assertRaisesRegex(ValueError, "three finite numbers"):
            validate_prediction_cache(malformed)

        malformed = deepcopy(valid)
        malformed["records"][0]["body_keypoints"] = [
            {
                "schema": "fixture-points",
                "points": {"nose": [10.0, 10.0, 0.9]},
                "unexpected": True,
            }
        ]
        with self.assertRaisesRegex(ValueError, "keypoint set schema"):
            validate_prediction_cache(malformed)

    def test_cache_requires_content_bound_model_artifact(self) -> None:
        sample = _sample(Path("image.png"), "a" * 64)
        model = _model()
        model["artifact_sha256"] = "detector.pt"
        with self.assertRaisesRegex(ValueError, "model artifact SHA256"):
            build_prediction_cache((sample,), (_result(),), model=model)

    def test_multi_dog_manifest_does_not_inherit_source_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            sample = _sample(image_path, _sha256(image_path))
            result = _result(
                (
                    DetectionBox(5, 5, 40, 55, 0.9, 16, "dog"),
                    DetectionBox(60, 5, 95, 55, 0.8, 16, "dog"),
                )
            )
            cache = build_prediction_cache((sample,), (result,), model=_model())[
                "cache"
            ]
            output = root / "roi"
            bundle = build_roi_manifest(
                (sample,), (cache,), data_root=root, output_dir=output
            )
            records = bundle["manifest"]["records"]
            self.assertEqual(len(records), 2)
            self.assertTrue(
                all(record["registered_identity_id"] is None for record in records)
            )

            mislabeled = deepcopy(bundle)
            mislabeled["manifest"]["records"][0]["registered_identity_id"] = (
                compute_registered_dog_id("fixture:v1:dog:1")
            )
            mislabeled["manifest_sha256"] = content_sha256(mislabeled["manifest"])
            mislabeled_path = output / "mislabeled.json"
            _write_bundle(mislabeled_path, mislabeled)
            with self.assertRaisesRegex(ValueError, "multi-instance"):
                read_roi_manifest(mislabeled_path)

    def test_face_outputs_bind_hashes_and_actual_crop_rect(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            sample = _sample(image_path, _sha256(image_path))
            pose = KeypointSet(
                {
                    "left_eye": Keypoint(5, 10, 0.9),
                    "right_eye": Keypoint(15, 10, 0.9),
                    "nose_center": Keypoint(5, 20, 0.9),
                },
                "ap10k",
            )
            cache = build_prediction_cache(
                (sample,),
                (_result((DetectionBox(0, 0, 60, 60, 0.9, 16, "dog"),), (pose,)),),
                model=_model(),
            )["cache"]
            output = root / "roi"
            bundle = build_roi_manifest(
                (sample,), (cache,), data_root=root, output_dir=output
            )
            record = bundle["manifest"]["records"][0]
            self.assertIsNotNone(record["face_crop_rect_xyxy"])
            self.assertIsNotNone(record["face_quality"])
            self.assertNotEqual(record["face_crop_rect_xyxy"], record["face_roi_xyxy"])
            for path_field, hash_field in (
                ("face_crop_path", "face_crop_sha256"),
                ("face_source_valid_mask_path", "face_source_valid_mask_sha256"),
                ("weak_nose_crop_path", "weak_nose_crop_sha256"),
                (
                    "weak_nose_source_valid_mask_path",
                    "weak_nose_source_valid_mask_sha256",
                ),
            ):
                self.assertEqual(
                    record[hash_field], _sha256(output / record[path_field])
                )

    def test_reader_rejects_nested_traversal_and_artifact_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            sample = _sample(image_path, _sha256(image_path))
            cache = build_prediction_cache((sample,), (_result(),), model=_model())[
                "cache"
            ]
            output = root / "roi"
            bundle = build_roi_manifest(
                (sample,), (cache,), data_root=root, output_dir=output
            )

            traversal = deepcopy(bundle)
            traversal["manifest"]["records"][0]["dog_crop_path"] = "../outside.jpg"
            traversal["manifest_sha256"] = content_sha256(traversal["manifest"])
            traversal_path = output / "traversal.json"
            _write_bundle(traversal_path, traversal)
            with self.assertRaisesRegex(ValueError, "canonical safe relative path"):
                read_roi_manifest(traversal_path)

            outside = root / "outside.jpg"
            Image.new("RGB", (224, 224), (0, 0, 0)).save(outside)
            escape = output / "dog_crops" / "escape.jpg"
            escape.symlink_to(outside)
            containment = deepcopy(bundle)
            containment_record = containment["manifest"]["records"][0]
            containment_record["dog_crop_path"] = "dog_crops/escape.jpg"
            containment_record["dog_crop_sha256"] = _sha256(outside)
            containment["manifest_sha256"] = content_sha256(containment["manifest"])
            containment_path = output / "containment.json"
            _write_bundle(containment_path, containment)
            with self.assertRaisesRegex(ValueError, "under the manifest parent"):
                read_roi_manifest(containment_path)

            manifest_path = output / "roi_manifest.json"
            _write_bundle(manifest_path, bundle)
            crop_path = output / bundle["manifest"]["records"][0]["dog_crop_path"]
            with crop_path.open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "dog_crop_sha256 differs"):
                read_roi_manifest(manifest_path)

    def test_reader_rejects_artifact_dimension_and_nested_schema_changes(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            sample = _sample(image_path, _sha256(image_path))
            cache = build_prediction_cache((sample,), (_result(),), model=_model())[
                "cache"
            ]
            output = root / "roi"
            bundle = build_roi_manifest(
                (sample,), (cache,), data_root=root, output_dir=output
            )
            valid_bundle = deepcopy(bundle)
            record = bundle["manifest"]["records"][0]
            crop_path = output / record["dog_crop_path"]
            Image.new("RGB", (32, 32), (0, 0, 0)).save(crop_path)
            record["dog_crop_sha256"] = _sha256(crop_path)
            bundle["manifest_sha256"] = content_sha256(bundle["manifest"])
            dimension_path = output / "dimension.json"
            _write_bundle(dimension_path, bundle)
            with self.assertRaisesRegex(ValueError, "dimensions differ"):
                read_roi_manifest(dimension_path)

            Image.new("RGB", (224, 224), (120, 100, 80)).save(crop_path, quality=95)
            valid_record = valid_bundle["manifest"]["records"][0]
            valid_record["dog_crop_sha256"] = _sha256(crop_path)
            valid_bundle["manifest_sha256"] = content_sha256(valid_bundle["manifest"])
            malformed = deepcopy(valid_bundle)
            del malformed["manifest"]["records"][0]["quality"]["overall"]
            malformed["manifest_sha256"] = content_sha256(malformed["manifest"])
            malformed_path = output / "malformed.json"
            _write_bundle(malformed_path, malformed)
            with self.assertRaisesRegex(ValueError, "quality schema differs"):
                read_roi_manifest(malformed_path)

    def test_roi_manifest_rejects_duplicate_teacher_caches(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (100, 60), (120, 100, 80)).save(image_path)
            digest = _sha256(image_path)
            sample = _sample(image_path, digest)
            cache = build_prediction_cache((sample,), (_result(),), model=_model())[
                "cache"
            ]

            with self.assertRaisesRegex(ValueError, "duplicate teacher artifact"):
                build_roi_manifest(
                    (sample,), (cache, cache), data_root=root, output_dir=root / "roi"
                )


if __name__ == "__main__":
    unittest.main()
