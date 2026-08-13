from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
import uuid
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import numpy as np
from PIL import Image

from canine_identity.engine import IdentityEngine, Match
from contracts.artifact_manifest import (
    ArtifactLicense,
    ImagePreprocessing,
    LandmarkGraphManifest,
    LandmarkGraphPreprocessing,
    LandmarkKeypointManifest,
    UsageLane,
)
from contracts.model_contract import (
    ConvNeXtModelManifest,
    DogFaceNetModelManifest,
    OnnxEvidenceContractError,
    OnnxModelLicenseState,
    OnnxModelUsageLane,
    OnnxPreprocessingContract,
    PetReIDModelManifest,
)
from evidence_fusion.base import AbstractEvidencer
from identity_governance.identity_registry import compute_registered_dog_id
from identity_retrieval.gallery import IdentityGallery
from identity_retrieval.pipeline.extraction import EvidenceExtractionPipeline
from identity_retrieval.pipeline.retrieval import IdentityRetrievalPipeline


class _FixedEvidencer(AbstractEvidencer):
    name = "fixed"
    output_dim = 2

    def __init__(self, value: np.ndarray) -> None:
        self._value = np.asarray(value, dtype=np.float32)

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._value.copy()

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([self._value for _ in images])


class _ConfiguredOnnxEvidencer(AbstractEvidencer):
    calls: ClassVar[list[tuple[Path, object, bool]]] = []

    def __init__(self, model_path: Path, manifest: object, *, use_cuda: bool):
        self.calls.append((model_path, manifest, use_cuda))
        self.output_dim = manifest.output_dim  # type: ignore[attr-defined]

    def extract(self, image: Image.Image) -> np.ndarray:
        return np.ones(self.output_dim, dtype=np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), self.output_dim), dtype=np.float32)


class _ConfiguredLocalDinov2Evidencer(AbstractEvidencer):
    calls: ClassVar[list[dict]] = []
    output_dim = 384
    gallery_contract_fields: ClassVar[dict[str, str]] = {
        "model_sha256": "1" * 64,
        "preprocessor_sha256": "2" * 64,
        "weight_intake_receipt_sha256": "3" * 64,
        "preprocessor_intake_receipt_sha256": "4" * 64,
    }

    def __init__(self, **kwargs):
        self.calls.append(kwargs)

    def extract(self, image: Image.Image) -> np.ndarray:
        return np.ones(self.output_dim, dtype=np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), self.output_dim), dtype=np.float32)


class _ConfiguredLandmarkEvidencer(AbstractEvidencer):
    calls: ClassVar[list[tuple[Path, object, Path, object, bool]]] = []

    def __init__(
        self,
        keypoint_path: Path,
        keypoint_manifest: object,
        graph_path: Path,
        graph_manifest: object,
        *,
        use_cuda: bool,
    ) -> None:
        self.calls.append(
            (keypoint_path, keypoint_manifest, graph_path, graph_manifest, use_cuda)
        )
        self.output_dim = graph_manifest.output_shape[1]  # type: ignore[attr-defined]

    def extract(self, image: Image.Image) -> np.ndarray:
        return np.ones(self.output_dim, dtype=np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), self.output_dim), dtype=np.float32)


class _PublicSearchRecorder:
    def __init__(self) -> None:
        self.search_arguments = None
        self.enroll_arguments = None

    def search(self, image, top_k, breed_filter):
        self.search_arguments = (image, top_k, breed_filter)
        return []

    def enroll(self, *arguments):
        self.enroll_arguments = arguments
        return 7


def _manifest(manifest_type, model_kind: str):
    return manifest_type(
        model_id=f"test-{model_kind}",
        model_sha256="a" * 64,
        input_name="images",
        input_shape=("batch", 3, 8, 8),
        output_name="embedding",
        output_dim=4,
        preprocessing=OnnxPreprocessingContract(
            color_mode="RGB",
            layout="NCHW",
            dtype="float32",
            resize="bilinear",
            scale=1.0 / 255.0,
            mean=(0.0, 0.0, 0.0),
            std=(1.0, 1.0, 1.0),
        ),
        model_kind=model_kind,
        usage_lane=OnnxModelUsageLane.RESEARCH_ONLY,
        license_state=OnnxModelLicenseState.UNVERIFIED,
    )


def _landmark_manifests():
    license_contract = ArtifactLicense("LicenseRef-Test", UsageLane.TEST_FIXTURE)
    keypoint_order = ("left_eye", "right_eye")
    keypoint = LandmarkKeypointManifest(
        artifact_id="keypoints",
        artifact_sha256="b" * 64,
        input_name="images",
        input_shape=(1, 3, 8, 8),
        output_name="heatmaps",
        output_shape=(1, 2, 2, 2),
        license=license_contract,
        preprocessing=ImagePreprocessing(
            "RGB", "NCHW", "float32", "bilinear", 1.0 / 255.0,
            (0.0, 0.0, 0.0), (1.0, 1.0, 1.0), None,
        ),
        keypoint_order=keypoint_order,
        visibility_threshold=0.5,
        min_visible_keypoints=2,
    )
    graph = LandmarkGraphManifest(
        artifact_id="graph",
        artifact_sha256="c" * 64,
        input_name="landmarks",
        input_shape=(1, 2, 4),
        output_name="embedding",
        output_shape=(1, 8),
        license=license_contract,
        keypoint_order=keypoint_order,
        preprocessing=LandmarkGraphPreprocessing(
            "crop_normalized_xy", (0.0, 1.0), "binary_0_1"
        ),
    )
    return keypoint, graph


def _gallery_contract(
    channels: list[tuple[str, int]], weights: list[float] | None = None
) -> dict:
    weights = weights or [1.0 / len(channels)] * len(channels)
    return {
        "schema_version": "cvi.gallery_embedding_contract.v1",
        "dimension": sum(dimension for _, dimension in channels),
        "channels": [
            {"name": name, "dimension": dimension, "optional": False}
            for name, dimension in channels
        ],
        "fusion": {
            "type": "weighted_concatenated_cosine",
            "weights": weights,
            "embedding_scales": [float(np.sqrt(weight)) for weight in weights],
        },
    }


def _dog_id(label: str) -> str:
    return compute_registered_dog_id(f"fixture:v1:runtime-contract:{label}")


class GalleryContractTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_versioned_gallery_round_trip(self) -> None:
        dog_one = compute_registered_dog_id("fixture:v1:dog:1")
        dog_two = compute_registered_dog_id("fixture:v1:dog:2")
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), dog_one, {"display_name": "one"})
        index.enroll(np.array([0.0, 1.0]), dog_one)
        index.enroll(np.array([1.0, 1.0]), dog_two)
        index.save()
        loaded = IdentityGallery(self.root, dim=2, read_only=True)
        self.assertEqual(loaded.size, 3)
        result = loaded.search(np.array([1.0, 0.0]), top_k=1)[0]
        self.assertEqual(result[2]["registered_dog_id"], dog_one)
        manifest = json.loads(
            (self.root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["template_count"], 3)
        self.assertEqual(manifest["identity_count"], 2)
        self.assertEqual(manifest["identity_aggregation"], "max")
        loaded.close()
        index.close()

    def test_gallery_load_rejects_noncanonical_persisted_identity(self) -> None:
        valid_id = _dog_id("persisted-identity")
        cases = (valid_id.upper(), str(uuid.uuid4()), "dog-1")
        for invalid_id in cases:
            with (
                self.subTest(invalid_id=invalid_id),
                tempfile.TemporaryDirectory() as directory,
            ):
                root = Path(directory)
                index = IdentityGallery(root, dim=2)
                index.enroll(np.array([1.0, 0.0]), valid_id)
                index.save()
                index.close()

                manifest_path = root / "gallery_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                metadata_path = root / manifest["files"]["metadata"]["name"]
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                metadata["0"]["registered_dog_id"] = invalid_id
                payload = (
                    json.dumps(
                        metadata,
                        sort_keys=True,
                        separators=(",", ":"),
                        allow_nan=False,
                    )
                    + "\n"
                ).encode("utf-8")
                metadata_path.write_bytes(payload)
                manifest["files"]["metadata"]["sha256"] = hashlib.sha256(
                    payload
                ).hexdigest()
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
                )

                with self.assertRaisesRegex(RuntimeError, "UUIDv5"):
                    IdentityGallery(root, dim=2, read_only=True)

    def test_direct_enrollment_requires_identity_but_metadata_labels_reload(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        for invalid_id in ("source-label", str(uuid.uuid4()), _dog_id("valid").upper()):
            with self.subTest(invalid_id=invalid_id), self.assertRaisesRegex(
                ValueError, "UUIDv5"
            ):
                index.enroll_with_breed(
                    np.array([1.0, 0.0]), invalid_id, "unknown"
                )
        self.assertEqual(index.size, 0)

        dog_id = _dog_id("metadata-label")
        index.enroll(
            np.array([1.0, 0.0]),
            dog_id,
            {"dog_id": "source-label", "registered_dog_id": "source-registration"},
        )
        index.save()
        index.close()

        loaded = IdentityGallery(self.root, dim=2, read_only=True)
        try:
            result = loaded.search(np.array([1.0, 0.0]), top_k=1)[0][2]
            self.assertEqual(result["registered_dog_id"], dog_id)
            self.assertEqual(result["metadata"]["dog_id"], "source-label")
            self.assertEqual(
                result["metadata"]["registered_dog_id"], "source-registration"
            )
        finally:
            loaded.close()

    def test_gallery_rejects_bad_vectors_and_accepts_multiple_templates(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        dog_id = _dog_id("bad-vectors")
        with self.assertRaises(ValueError):
            index.enroll(np.array([0.0, 0.0]), dog_id)
        with self.assertRaises(ValueError):
            index.enroll(np.array([np.nan, 1.0]), dog_id)
        index.enroll(np.array([1.0, 0.0]), dog_id)
        index.enroll(np.array([0.0, 1.0]), dog_id)
        self.assertEqual(index.size, 2)
        with self.assertRaises(ValueError):
            index.search(np.ones(3, dtype=np.float32))

    def test_gallery_enrollment_duplicate_semantics(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        dog_one = _dog_id("duplicate-one")
        dog_two = _dog_id("duplicate-two")
        with self.assertRaisesRegex(ValueError, "idempotency_key.*bounded"):
            index.enroll(
                np.array([1.0, 0.0]),
                dog_one,
                idempotency_key="\u00e9" * 32_769,
            )
        with self.assertRaisesRegex(ValueError, "metadata.*(bounded|size limit)"):
            index.enroll(
                np.array([1.0, 0.0]),
                dog_one,
                metadata={"note": "\u00e9" * 32_769},
            )
        metadata = {"capture": {"camera": "front"}}
        first = index.enroll(
            np.array([1.0, 0.0]),
            dog_one,
            metadata,
            idempotency_key="request-1",
        )
        metadata["capture"]["camera"] = "mutated"
        retry = index.enroll(
            np.array([1.0, 0.0]),
            dog_one,
            {"capture": {"camera": "front"}},
            idempotency_key="request-1",
        )
        self.assertEqual(retry, first)
        self.assertEqual(index.size, 1)

        with self.assertRaisesRegex(ValueError, "idempotency key"):
            index.enroll(
                np.array([0.0, 1.0]),
                dog_one,
                idempotency_key="request-1",
            )
        second = index.enroll(
            np.array([0.0, 1.0]),
            dog_one,
            idempotency_key="request-2",
        )
        self.assertNotEqual(second, first)
        third = index.enroll(
            np.array([0.0, 1.0]),
            dog_one,
            idempotency_key="request-3",
            content_sha256="a" * 64,
        )
        self.assertNotEqual(third, second)
        with self.assertRaisesRegex(ValueError, "different registered identity"):
            index.enroll(
                np.array([1.0, 0.0]),
                dog_two,
                idempotency_key="request-4",
            )

        result = index.search(np.array([1.0, 0.0]), top_k=1)[0]
        self.assertEqual(result[2]["metadata"]["capture"]["camera"], "front")
        self.assertEqual(len(result[2]["template_id"]), 64)
        self.assertEqual(len(result[2]["content_sha256"]), 64)

    def test_search_aggregates_templates_before_top_k_and_tie_breaks_by_id(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        dog_c = _dog_id("aggregate-c")
        dog_d = _dog_id("aggregate-d")
        index.enroll(np.array([0.99, 0.1]), dog_c)
        winning_index = index.enroll(np.array([1.0, 0.0]), dog_c)
        index.enroll(np.array([0.9, 0.4]), dog_d)
        results = index.search(np.array([1.0, 0.0]), top_k=2)
        self.assertEqual([row[2]["registered_dog_id"] for row in results], [
            dog_c, dog_d,
        ])
        self.assertEqual(results[0][0], winning_index)

        tied = IdentityGallery(self.root / "ties", dim=2)
        tied_ids = [_dog_id("tie-b"), _dog_id("tie-a")]
        tied.enroll(np.array([1.0, 1.0]), tied_ids[0])
        tied.enroll(np.array([1.0, -1.0]), tied_ids[1])
        tied_results = tied.search(np.array([1.0, 0.0]), top_k=2)
        self.assertEqual(
            [row[2]["registered_dog_id"] for row in tied_results],
            sorted(tied_ids),
        )

    def test_gallery_rejects_corrupted_metadata(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("corrupted-metadata"))
        index.save()
        manifest = json.loads(
            (self.root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        metadata = self.root / manifest["files"]["metadata"]["name"]
        metadata.write_text("{}\n", encoding="utf-8")
        index.close()
        with self.assertRaisesRegex(RuntimeError, "corrupted"):
            IdentityGallery(self.root, dim=2, read_only=True)

    def test_gallery_manifest_validates_template_and_identity_counts(self) -> None:
        for field, value, message in (
            ("template_count", 3, "template count"),
            ("identity_count", 2, "identity count"),
        ):
            with self.subTest(field=field), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                index = IdentityGallery(root, dim=2)
                index.enroll(np.array([1.0, 0.0]), _dog_id(f"count-{field}"))
                index.save()
                manifest_path = root / "gallery_manifest.json"
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                manifest[field] = value
                if field == "template_count":
                    manifest["count"] = value
                manifest_path.write_text(
                    json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
                )
                index.close()
                with self.assertRaisesRegex(RuntimeError, message):
                    IdentityGallery(root, dim=2, read_only=True)

    def test_gallery_resave_preserves_corrupted_generation_file(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("corrupted-generation"))
        index.save()
        manifest_path = self.root / "gallery_manifest.json"
        manifest_bytes = manifest_path.read_bytes()
        manifest = json.loads(manifest_bytes)
        metadata = self.root / manifest["files"]["metadata"]["name"]
        metadata.write_text("{}\n", encoding="utf-8")
        corrupted_bytes = metadata.read_bytes()

        with self.assertRaisesRegex(RuntimeError, "refusing to overwrite"):
            index.save()

        self.assertEqual(metadata.read_bytes(), corrupted_bytes)
        self.assertEqual(manifest_path.read_bytes(), manifest_bytes)
        index.close()

    def test_gallery_rejects_symlink_root_and_sidecar(self) -> None:
        real_root = self.root / "real"
        root_link = self.root / "root-link"
        real_root.mkdir()
        root_link.symlink_to(real_root, target_is_directory=True)
        with self.assertRaisesRegex(RuntimeError, "root.*symbolic link"):
            IdentityGallery(root_link, dim=2)

        index = IdentityGallery(real_root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("symlink"))
        index.save()
        manifest = json.loads(
            (real_root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        metadata = real_root / manifest["files"]["metadata"]["name"]
        target = real_root / "metadata-target.json"
        target.write_bytes(metadata.read_bytes())
        metadata.unlink()
        metadata.symlink_to(target)
        index.close()

        with self.assertRaisesRegex(RuntimeError, "without following links"):
            IdentityGallery(real_root, dim=2, read_only=True)

    def test_gallery_rejects_bounded_manifest_and_cardinality_before_sidecars(
        self,
    ) -> None:
        manifest_path = self.root / "gallery_manifest.json"
        manifest_path.write_bytes(b" " * 17)
        with (
            patch("identity_retrieval.gallery._MAXIMUM_MANIFEST_BYTES", 16),
            self.assertRaisesRegex(RuntimeError, "byte limit"),
        ):
            IdentityGallery(self.root, dim=2, read_only=True)

        manifest_path.unlink()
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("cardinality"))
        index.save()
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        manifest["count"] = 1_000_001
        manifest["template_count"] = 1_000_001
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        for entry in manifest["files"].values():
            (self.root / entry["name"]).unlink()
        index.close()

        with self.assertRaisesRegex(RuntimeError, "count is invalid"):
            IdentityGallery(self.root, dim=2, read_only=True)

    def test_gallery_rejects_oversized_json_sidecar(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("oversized-sidecar"))
        index.save()
        index.close()

        with (
            patch("identity_retrieval.gallery._MAXIMUM_SIDECAR_JSON_BYTES", 16),
            self.assertRaisesRegex(RuntimeError, "byte limit"),
        ):
            IdentityGallery(self.root, dim=2, read_only=True)

    def test_gallery_rejects_oversized_faiss_before_deserialization(self) -> None:
        dog_id = compute_registered_dog_id("fixture:v1:dog:oversized-faiss")
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), dog_id)
        index.save()
        manifest = json.loads(
            (self.root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        required_index = self.root / manifest["files"]["required_index"]["name"]
        with required_index.open("r+b") as stream:
            stream.truncate(1024 * 1024 + 2 * 4 + 1)
        index.close()

        with (
            patch("identity_retrieval.gallery.faiss.read_index") as read_index,
            self.assertRaisesRegex(RuntimeError, "required index.*byte limit"),
        ):
            IdentityGallery(self.root, dim=2, read_only=True)
        read_index.assert_not_called()

    def test_publication_limits_do_not_mutate_prior_generation(self) -> None:
        for limit_name, message in (
            ("_MAXIMUM_SIDECAR_JSON_BYTES", "sidecar.*byte limit"),
            ("_MAXIMUM_MANIFEST_BYTES", "manifest.*byte limit"),
        ):
            with self.subTest(limit=limit_name):
                root = self.root / limit_name
                dog_one = compute_registered_dog_id(f"fixture:v1:{limit_name}:1")
                dog_two = compute_registered_dog_id(f"fixture:v1:{limit_name}:2")
                index = IdentityGallery(root, dim=2)
                index.enroll(np.array([1.0, 0.0]), dog_one)
                index.save()
                snapshot = {
                    path.name: path.read_bytes()
                    for path in root.iterdir()
                    if path.is_file()
                }
                index.enroll(np.array([0.0, 1.0]), dog_two)

                with (
                    patch(f"identity_retrieval.gallery.{limit_name}", 16),
                    self.assertRaisesRegex(RuntimeError, message),
                ):
                    index.save()

                self.assertEqual(
                    {
                        path.name: path.read_bytes()
                        for path in root.iterdir()
                        if path.is_file()
                    },
                    snapshot,
                )
                index.close()

    def test_gallery_sidecar_rejects_duplicate_keys(self) -> None:
        index = IdentityGallery(self.root, dim=2)
        index.enroll(np.array([1.0, 0.0]), _dog_id("duplicate-sidecar"))
        index.save()
        manifest_path = self.root / "gallery_manifest.json"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        metadata = self.root / manifest["files"]["metadata"]["name"]
        duplicate_payload = b'{"0":{},"0":{}}\n'
        metadata.write_bytes(duplicate_payload)
        manifest["files"]["metadata"]["sha256"] = hashlib.sha256(
            duplicate_payload
        ).hexdigest()
        manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
        index.close()

        with self.assertRaisesRegex(RuntimeError, "duplicate JSON object key"):
            IdentityGallery(self.root, dim=2, read_only=True)

    def test_gallery_rejects_a_different_embedding_contract(self) -> None:
        first_contract = {
            "schema_version": "cvi.gallery_embedding_contract.v1",
            "dimension": 2,
            "channels": ["a", "b"],
            "weights": [0.9, 0.1],
        }
        second_contract = {
            **first_contract,
            "weights": [0.1, 0.9],
        }
        index = IdentityGallery(
            self.root, dim=2, embedding_contract=first_contract
        )
        index.enroll(np.array([1.0, 0.0]), _dog_id("different-contract"))
        index.save()
        index.close()
        with self.assertRaisesRegex(RuntimeError, "embedding contract"):
            IdentityGallery(
                self.root, dim=2, embedding_contract=second_contract,
                read_only=True,
            )

    def test_unversioned_gallery_is_rejected(self) -> None:
        (self.root / "metadata.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(RuntimeError, "unversioned"):
            IdentityGallery(self.root, dim=2)


class SearchContractTests(unittest.TestCase):
    def test_evidence_uses_the_qk_channel_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = IdentityGallery(
                Path(directory), dim=4,
                embedding_contract=_gallery_contract(
                    [("b", 2), ("a", 2)], [0.5, 0.5]
                ),
            )
            candidate = {
                "a": np.array([1.0, 0.0]),
                "b": np.array([0.0, 1.0]),
            }
            query = {
                "a": np.array([1.0, 0.0]),
                "b": np.array([1.0, 0.0]),
            }
            extraction = EvidenceExtractionPipeline({
                name: _FixedEvidencer(embedding)
                for name, embedding in query.items()
            })
            pipeline = IdentityRetrievalPipeline(extraction, index)
            index.enroll(candidate, _dog_id("fusion-order"))
            result = pipeline.search(Image.new("RGB", (4, 4)), top_k=1)[0]
            self.assertAlmostEqual(result.evidence["a"], 1.0)
            self.assertAlmostEqual(result.evidence["b"], 0.0)
            self.assertIn("template_id", result.metadata)

    def test_pipeline_content_hash_distinguishes_images_with_same_embedding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            index = IdentityGallery(
                Path(directory), dim=2,
                embedding_contract=_gallery_contract([("appearance", 2)]),
            )
            evidence = EvidenceExtractionPipeline({
                "appearance": _FixedEvidencer(np.array([1.0, 0.0]))
            })
            pipeline = IdentityRetrievalPipeline(evidence, index)
            dog_id = _dog_id("same-embedding")
            pipeline.enroll(Image.new("RGB", (2, 2), "black"), dog_id)
            pipeline.enroll(Image.new("RGB", (2, 2), "white"), dog_id)
            self.assertEqual(index.size, 2)

class PublicConfigurationTests(unittest.TestCase):
    @staticmethod
    def _config(channels: dict) -> dict:
        return {
            "schema_version": "cvi.retrieval_config.v1",
            "mode": "closed_set_retrieval",
            "index_dir": "/tmp/cvi-public-runtime-contract-gallery",
            "channels": channels,
        }

    def test_match_metadata_is_backward_compatible_and_serialized_when_present(self) -> None:
        legacy = Match("dog-1", 0.5, {"visual": 0.5})
        self.assertEqual(
            legacy.to_dict(),
            {"dog_id": "dog-1", "similarity": 0.5, "evidence": {"visual": 0.5}},
        )
        match = Match("dog-1", 0.5, {}, {"template_id": "template-1"})
        self.assertEqual(match.to_dict()["metadata"]["template_id"], "template-1")

    def test_configuration_requires_explicit_mode_and_channels(self) -> None:
        with self.assertRaisesRegex(ValueError, "explicit"):
            IdentityEngine()
        with self.assertRaisesRegex(ValueError, "explicit index_dir"):
            IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "channels": {},
            })
        with self.assertRaisesRegex(ValueError, "channels"):
            IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "index_dir": "/tmp/cvi-public-runtime-contract-gallery",
            })
        with self.assertRaisesRegex(ValueError, "schema_version"):
            IdentityEngine({"mode": "closed_set_retrieval", "channels": {}})

    def test_configuration_rejects_unknown_models(self) -> None:
        with self.assertRaisesRegex(ValueError, "unsupported channel"):
            IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "index_dir": "/tmp/cvi-public-runtime-contract-gallery",
                "channels": {"visual": {"type": "dinov3"}},
            })

    def test_unpinned_appearance_types_are_always_rejected(self) -> None:
        for kind in ("dinov2", "appearance"):
            for spec in (
                {"type": kind},
                {"type": kind, "allow_unpinned_research_model": True},
            ):
                with (
                    self.subTest(spec=spec),
                    self.assertRaisesRegex(ValueError, "unsupported channel type"),
                ):
                    IdentityEngine(self._config({"visual": spec}))

    def test_public_local_dinov2_is_receipt_bound_and_gallery_bound(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channel = {
                "type": "dinov2_local",
                "model_dir": str(root / "model"),
                "weight_intake_bundle": str(root / "weight-intake.json"),
                "preprocessor_intake_bundle": str(
                    root / "preprocessor-intake.json"
                ),
                "device": "cuda",
            }
            _ConfiguredLocalDinov2Evidencer.calls.clear()
            with patch(
                "identity_methods.appearance.ReceiptBoundDinov2Small",
                _ConfiguredLocalDinov2Evidencer,
            ):
                runtime = IdentityEngine({
                    **self._config({"visual": channel}),
                    "index_dir": str(root / "index"),
                })

            self.assertEqual(
                _ConfiguredLocalDinov2Evidencer.calls,
                [{
                    "model_directory": root / "model",
                    "weight_intake_bundle": root / "weight-intake.json",
                    "preprocessor_intake_bundle": (
                        root / "preprocessor-intake.json"
                    ),
                    "device": "cuda",
                }],
            )
            contract_channel = runtime._gallery._embedding_contract["channels"][0]
            self.assertEqual(contract_channel["configuration"], channel)
            for field, digest in (
                _ConfiguredLocalDinov2Evidencer.gallery_contract_fields.items()
            ):
                self.assertEqual(contract_channel[field], digest)

            runtime.close()
            changed_receipt_fields = {
                **_ConfiguredLocalDinov2Evidencer.gallery_contract_fields,
                "weight_intake_receipt_sha256": "5" * 64,
            }
            with (
                patch(
                    "identity_methods.appearance.ReceiptBoundDinov2Small",
                    _ConfiguredLocalDinov2Evidencer,
                ),
                patch.object(
                    _ConfiguredLocalDinov2Evidencer,
                    "gallery_contract_fields",
                    changed_receipt_fields,
                ),
                self.assertRaisesRegex(RuntimeError, "embedding contract"),
            ):
                IdentityEngine({
                    **self._config({"visual": channel}),
                    "index_dir": str(root / "index"),
                })

    def test_public_local_dinov2_requires_exact_schema_and_explicit_device(self) -> None:
        valid = {
            "type": "dinov2_local",
            "model_dir": "/local/model",
            "weight_intake_bundle": "/local/weight.json",
            "preprocessor_intake_bundle": "/local/preprocessor.json",
            "device": "cpu",
        }
        invalid_specs = (
            {key: value for key, value in valid.items() if key != "device"},
            {**valid, "network_model_id": "facebook/dinov2-small"},
            {**valid, "device": "auto"},
            {**valid, "device": ["cpu"]},
            {**valid, "model_dir": ""},
        )
        for spec in invalid_specs:
            with self.subTest(spec=spec), self.assertRaises(ValueError):
                IdentityEngine(self._config({"visual": spec}))

    def test_configuration_rejects_heuristic_open_set(self) -> None:
        with self.assertRaisesRegex(ValueError, "frozen calibration"):
            IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "channels": {"visual": {"type": "dinov2"}},
                "open_set": {"enabled": True},
            })

    def test_file_configuration_rejects_duplicate_keys_and_nonfinite_dict(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            path.write_text(
                '{"schema_version":"cvi.retrieval_config.v1",'
                '"mode":"closed_set_retrieval","mode":"other",'
                '"channels":{}}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                IdentityEngine(path)
            path.write_text(
                '{"schema_version":"cvi.retrieval_config.v1",'
                '"mode":"closed_set_retrieval",'
                '"index_dir":"/tmp/cvi-index","channels":{},'
                '"fusion_weights":[NaN]}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "non-finite"):
                IdentityEngine(path)
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "channels": {"visual": {"type": "dinov3"}},
                "fusion_weights": [float("nan")],
            })

    def test_file_configuration_is_bounded_and_does_not_follow_symlinks(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            oversized = root / "oversized.json"
            oversized.write_bytes(b" " * (1_048_576 + 1))
            with self.assertRaisesRegex(ValueError, "bounded regular file"):
                IdentityEngine(oversized)

            target = root / "config-target.json"
            target.write_text("{}", encoding="utf-8")
            link = root / "config.json"
            link.symlink_to(target)
            with self.assertRaisesRegex(ValueError, "without following links"):
                IdentityEngine(link)

    def test_manifest_contracts_round_trip_and_reject_schema_substitution(self) -> None:
        cases = (
            (DogFaceNetModelManifest, "dogfacenet_onnx"),
            (ConvNeXtModelManifest, "convnext_onnx"),
            (PetReIDModelManifest, "petreid_nose_onnx"),
        )
        for manifest_type, model_kind in cases:
            with self.subTest(model_kind=model_kind):
                manifest = _manifest(manifest_type, model_kind)
                payload = manifest.to_dict()
                self.assertEqual(manifest_type.from_dict(payload), manifest)
                self.assertEqual(
                    set(payload),
                    {
                        "schema_version", "model_kind", "model_id",
                        "model_sha256", "input_name", "input_shape",
                        "output_name", "output_dim", "preprocessing",
                        "usage_lane", "license_state",
                    },
                )
                with self.assertRaisesRegex(
                    OnnxEvidenceContractError, "exact-key"
                ):
                    manifest_type.from_dict({**payload, "input_size": 8})
                wrong_kind = {
                    **payload,
                    "model_kind": "generic_onnx",
                }
                with self.assertRaisesRegex(
                    OnnxEvidenceContractError, "model_kind"
                ):
                    manifest_type.from_dict(wrong_kind)

        preprocessing = _manifest(
            DogFaceNetModelManifest, "dogfacenet_onnx"
        ).preprocessing.to_dict()
        with self.assertRaisesRegex(OnnxEvidenceContractError, "exact-key"):
            OnnxPreprocessingContract.from_dict({**preprocessing, "size": 8})

    def test_public_onnx_channels_load_strict_manifests_and_devices(self) -> None:
        cases = (
            ("dogfacenet_onnx", DogFaceNetModelManifest, False),
            ("convnext_onnx", ConvNeXtModelManifest, True),
            ("petreid_nose_onnx", PetReIDModelManifest, False),
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            channels = {}
            for index, (kind, manifest_type, use_cuda) in enumerate(cases):
                model_path = root / f"{kind}.onnx"
                manifest_path = root / f"{kind}.manifest.json"
                model_path.write_bytes(b"fixture")
                manifest_path.write_text(
                    json.dumps(_manifest(manifest_type, kind).to_dict()),
                    encoding="utf-8",
                )
                spec = {
                    "type": kind,
                    "model_path": str(model_path),
                    "manifest_path": str(manifest_path),
                }
                if use_cuda:
                    spec["device"] = "cuda"
                channels[f"channel-{index}"] = spec

            _ConfiguredOnnxEvidencer.calls.clear()
            with (
                patch(
                    "identity_methods.backbones.extractors.DogFaceNetExtractor",
                    _ConfiguredOnnxEvidencer,
                ),
                patch(
                    "identity_methods.backbones.extractors.ConvNeXtExtractor",
                    _ConfiguredOnnxEvidencer,
                ),
                patch(
                    "identity_methods.backbones.extractors.PetReIDExtractor",
                    _ConfiguredOnnxEvidencer,
                ),
            ):
                runtime = IdentityEngine({
                    **self._config(channels),
                    "index_dir": str(root / "index"),
                })

            self.assertEqual(runtime.size, 0)
            self.assertEqual(
                [call[2] for call in _ConfiguredOnnxEvidencer.calls],
                [False, True, False],
            )
            self.assertEqual(
                [call[1].model_kind for call in _ConfiguredOnnxEvidencer.calls],
                [case[0] for case in cases],
            )

    def test_public_landmark_channel_loads_exact_paired_manifests(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            keypoint_model = root / "keypoints.onnx"
            graph_model = root / "graph.onnx"
            keypoint_manifest_path = root / "keypoints.manifest.json"
            graph_manifest_path = root / "graph.manifest.json"
            keypoint_model.write_bytes(b"fixture")
            graph_model.write_bytes(b"fixture")
            keypoint_manifest, graph_manifest = _landmark_manifests()
            keypoint_manifest_path.write_text(
                json.dumps(keypoint_manifest.to_dict()), encoding="utf-8"
            )
            graph_manifest_path.write_text(
                json.dumps(graph_manifest.to_dict()), encoding="utf-8"
            )
            channel = {
                "type": "landmark_onnx",
                "keypoint_model_path": str(keypoint_model),
                "keypoint_manifest_path": str(keypoint_manifest_path),
                "graph_model_path": str(graph_model),
                "graph_manifest_path": str(graph_manifest_path),
                "device": "cuda",
            }
            _ConfiguredLandmarkEvidencer.calls.clear()
            with patch(
                "localization.landmark_graph.LandmarkEvidencer",
                _ConfiguredLandmarkEvidencer,
            ):
                runtime = IdentityEngine({
                    **self._config({"landmark": channel}),
                    "index_dir": str(root / "index"),
                })
            self.assertEqual(runtime.size, 0)
            call = _ConfiguredLandmarkEvidencer.calls[0]
            self.assertEqual(call[0], keypoint_model)
            self.assertEqual(call[2], graph_model)
            self.assertEqual(call[1], keypoint_manifest)
            self.assertEqual(call[3], graph_manifest)
            self.assertTrue(call[4])

    def test_public_onnx_channels_reject_shape_only_and_duplicate_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_path = root / "model.onnx"
            manifest_path = root / "manifest.json"
            model_path.write_bytes(b"fixture")
            channel = {
                "type": "dogfacenet_onnx",
                "model_path": str(model_path),
                "manifest_path": str(manifest_path),
            }
            with self.assertRaisesRegex(ValueError, "exact ONNX channel schema"):
                IdentityEngine(self._config({
                    "visual": {
                        "type": "dogfacenet_onnx",
                        "model_path": str(model_path),
                        "input_shape": ["batch", 3, 8, 8],
                    }
                }))

            payload = json.dumps(
                _manifest(
                    DogFaceNetModelManifest, "dogfacenet_onnx"
                ).to_dict()
            )
            manifest_path.write_text(
                payload[:-1] + ',"model_kind":"dogfacenet_onnx"}',
                encoding="utf-8",
            )
            with self.assertRaisesRegex(ValueError, "duplicate"):
                IdentityEngine(self._config({"visual": channel}))

    def test_public_arguments_are_bounded_canonical_and_finite(self) -> None:
        runtime = IdentityEngine.__new__(IdentityEngine)
        runtime._retrieval = _PublicSearchRecorder()
        image = Image.new("RGB", (2, 2))
        with self.assertRaisesRegex(ValueError, "top_k"):
            runtime.search(image, top_k=1_001)
        for breed_filter in (["beagle", "beagle"], [" beagle"], [None]):
            with self.subTest(breed_filter=breed_filter), self.assertRaises(ValueError):
                runtime.search(image, breed_filter=breed_filter)  # type: ignore[arg-type]
        runtime.search(image, top_k=2, breed_filter=["beagle"])
        self.assertEqual(runtime._retrieval.search_arguments[2], ["beagle"])

        dog_id = "877d96de-ba43-542d-9523-5c20213bfc09"
        with self.assertRaisesRegex(ValueError, "finite JSON"):
            runtime.enroll(image, dog_id, metadata={"quality": float("nan")})
        with self.assertRaisesRegex(ValueError, "keys must be strings"):
            runtime.enroll(image, dog_id, metadata={"capture": {1: "front"}})
        with self.assertRaisesRegex(ValueError, "breed"):
            runtime.enroll(image, dog_id, breed=" beagle")
        with self.assertRaisesRegex(ValueError, "idempotency_key.*byte limit"):
            runtime.enroll(image, dog_id, idempotency_key="\u00e9" * 32_769)
        metadata = {"capture": {"camera": "front"}}
        self.assertEqual(runtime.enroll(image, dog_id, metadata=metadata), 7)
        metadata["capture"]["camera"] = "mutated"
        self.assertEqual(
            runtime._retrieval.enroll_arguments[3],
            {"capture": {"camera": "front"}},
        )

    def test_enrollment_identity_requires_canonical_uuidv5(self) -> None:
        value = "877d96de-ba43-542d-9523-5c20213bfc09"
        self.assertEqual(IdentityEngine._validate_registered_dog_id(value), value)
        for invalid in ("bbo-bi", value.upper(), str(uuid.uuid4())):
            with self.subTest(invalid=invalid), self.assertRaisesRegex(
                ValueError, "UUIDv5"
            ):
                IdentityEngine._validate_registered_dog_id(invalid)


if __name__ == "__main__":
    unittest.main()
