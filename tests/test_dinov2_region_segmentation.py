from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path
from typing import ClassVar

import numpy as np
from PIL import Image

from data.types import UnifiedCanidSample
from parsing.regions.dinov2_region_segmentation import (
    EMBEDDING_DIMENSION,
    PATCH_GRID,
    derive_patch_region_candidates,
    produce_dataset_region_candidates,
    read_region_candidates,
)


def _features() -> np.ndarray:
    yy, xx = np.mgrid[0:PATCH_GRID, 0:PATCH_GRID]
    basis = np.stack(
        (
            xx / (PATCH_GRID - 1),
            yy / (PATCH_GRID - 1),
            np.sin(xx / 2.0),
            np.cos(yy / 2.0),
        ),
        axis=2,
    ).astype(np.float32)
    tiled = np.tile(basis, (1, 1, EMBEDDING_DIMENSION // basis.shape[2]))
    return tiled / np.linalg.norm(tiled, axis=2, keepdims=True).clip(1e-6)


class _FakeRuntime:
    binding: ClassVar[dict[str, str]] = {
        "model_id": "fixture-dinov2",
        "source_revision": "fixture",
        "model_sha256": "a" * 64,
        "weight_intake_receipt_sha256": "b" * 64,
        "preprocessor_sha256": "c" * 64,
        "preprocessor_intake_receipt_sha256": "d" * 64,
        "config_sha256": "e" * 64,
        "device": "cpu",
        "output": "FINAL_LAYER_PATCH_TOKENS_16X16X384",
    }

    def patch_features(self, images):
        return np.stack([_features() for _ in images])


class Dinov2RegionSegmentationTests(unittest.TestCase):
    def test_body_and_face_candidates_are_distinct_model_generated_masks(self) -> None:
        candidates = derive_patch_region_candidates(
            _features(),
            dataset_name="ap10k-dog",
            image_width=200,
            image_height=100,
            dog_box=(10, 5, 190, 95),
            body_keypoints={
                "left_eye": (80, 25, 1.0),
                "right_eye": (120, 25, 1.0),
                "nose_center": (100, 40, 1.0),
                "neck": (100, 60, 1.0),
            },
            face_box=None,
            face_landmarks=None,
            geometry_source="PUBLISHER_ANNOTATION",
        )
        for region in ("A", "F", "N"):
            self.assertEqual(candidates[region]["state"], "AVAILABLE")
            self.assertEqual(
                candidates[region]["qualification"], "MODEL_GENERATED_CANDIDATE"
            )
            self.assertEqual(
                candidates[region]["mask"].shape, (PATCH_GRID, PATCH_GRID)
            )
            self.assertEqual(
                candidates[region]["embedding"].shape, (EMBEDDING_DIMENSION,)
            )
            self.assertAlmostEqual(
                float(np.linalg.norm(candidates[region]["embedding"])), 1.0, places=5
            )

    def test_face_crop_does_not_claim_full_body(self) -> None:
        candidates = derive_patch_region_candidates(
            _features(),
            dataset_name="dogfacenet224",
            image_width=224,
            image_height=224,
            dog_box=None,
            body_keypoints=None,
            face_box=None,
            face_landmarks=None,
            geometry_source="SOURCE_REGION_PRIOR",
        )
        self.assertEqual(
            candidates["A"],
            {
                "state": "UNAVAILABLE",
                "reason": "SOURCE_REGION_DOES_NOT_CONTAIN_FULL_BODY",
            },
        )
        self.assertEqual(candidates["F"]["state"], "AVAILABLE")
        self.assertEqual(candidates["N"]["state"], "AVAILABLE")

    def test_dataset_production_and_reader_bind_arrays(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image_path = root / "image.png"
            Image.new("RGB", (64, 48), (120, 100, 80)).save(image_path)
            digest = hashlib.sha256(image_path.read_bytes()).hexdigest()
            sample = UnifiedCanidSample(
                sample_id="a" * 64,
                dataset_name="dogfacenet224",
                dataset_version="fixture-v1",
                source_group_id="group-1",
                image_path=image_path.name,
                image_sha256=digest,
                width=64,
                height=48,
                registered_identity_id="31b674c5-b4b7-50bb-84e1-b64573d1f15f",
                raw_identity_id="dog-1",
            )
            output = root / "output"
            bundle = produce_dataset_region_candidates(
                (sample,),
                data_root=root,
                output_dir=output,
                runtime=_FakeRuntime(),
                batch_size=1,
            )
            manifest, arrays = read_region_candidates(
                output / "region_candidates.json"
            )
            self.assertEqual(manifest["record_count"], 1)
            self.assertEqual(bundle["manifest"], manifest)
            self.assertTrue(np.isnan(arrays["A_embeddings"][0]).all())
            self.assertTrue(np.isfinite(arrays["F_embeddings"][0]).all())

            with (output / "region_candidates.npz").open("ab") as stream:
                stream.write(b"tamper")
            with self.assertRaisesRegex(ValueError, "byte size differs|SHA-256 differs"):
                read_region_candidates(output / "region_candidates.json")


if __name__ == "__main__":
    unittest.main()
