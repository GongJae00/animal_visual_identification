from __future__ import annotations

import hashlib
import io
import json
import tempfile
import unittest
import zipfile
from pathlib import Path
from typing import ClassVar
from unittest.mock import patch

import faiss
import numpy as np
from PIL import Image

from canine_identity.engine import IdentityEngine, Match
from evidence_fusion.base import (
    AbstractEvidencer,
    EvidenceObservation,
    EvidenceUnavailableReason,
    RequiredEvidenceUnavailableError,
)
from identity_governance.identity_registry import compute_registered_dog_id
from retrieval.gallery import IdentityGallery
from retrieval.pipeline.extraction import EvidenceExtractionPipeline
from retrieval.pipeline.retrieval import IdentityRetrievalPipeline
from retrieval.qkv import RetrievalQuery
from workflows.migrate_gallery_v3_to_v4 import migrate_gallery


class _FixedEvidence(AbstractEvidencer):
    output_dim = 2

    def __init__(self, value: np.ndarray | EvidenceObservation) -> None:
        self._value = value

    def extract(self, image: Image.Image) -> np.ndarray | EvidenceObservation:
        if isinstance(self._value, EvidenceObservation):
            return self._value
        return self._value.copy()

    def extract_batch(self, images: list[Image.Image]):
        return [self.extract(image) for image in images]


class _BrokenEvidence(_FixedEvidence):
    def extract(self, image: Image.Image) -> np.ndarray:
        raise RuntimeError("runtime failed")


class _ConfiguredNoseEvidence(AbstractEvidencer):
    calls: ClassVar[list[tuple[tuple, dict]]] = []
    output_dim = 3

    def __init__(self, *args, **kwargs) -> None:
        self.calls.append((args, kwargs))

    def extract(self, image: Image.Image) -> np.ndarray:
        return np.ones(3, dtype=np.float32)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.ones((len(images), 3), dtype=np.float32)


def _contract() -> dict:
    return {
        "schema_version": "cvi.gallery_embedding_contract.v1",
        "dimension": 4,
        "channels": [
            {"name": "required", "dimension": 2, "optional": False},
            {"name": "optional", "dimension": 2, "optional": True},
        ],
        "fusion": {
            "type": "weighted_concatenated_cosine",
            "weights": [0.1, 0.9],
            "embedding_scales": [float(np.sqrt(0.1)), float(np.sqrt(0.9))],
        },
    }


def _dog_id(label: str) -> str:
    return compute_registered_dog_id(f"fixture:v1:optional-runtime:{label}")


def _replace_optional_npz_member(
    root: Path, member_name: str, replacement: bytes
) -> Path:
    manifest_path = root / "gallery_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    optional_vectors = root / manifest["files"]["optional_vectors"]["name"]
    with zipfile.ZipFile(optional_vectors, mode="r") as archive:
        members = {
            member.filename: archive.read(member) for member in archive.infolist()
        }
    members[member_name] = replacement
    with zipfile.ZipFile(
        optional_vectors, mode="w", compression=zipfile.ZIP_DEFLATED
    ) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    manifest["files"]["optional_vectors"]["sha256"] = hashlib.sha256(
        optional_vectors.read_bytes()
    ).hexdigest()
    manifest_path.write_text(
        json.dumps(manifest, sort_keys=True) + "\n", encoding="utf-8"
    )
    return optional_vectors


class ObservationBoundaryTests(unittest.TestCase):
    def test_optional_insufficiency_is_unavailable_but_required_fails(self) -> None:
        unavailable = EvidenceObservation.unavailable(
            "optional", EvidenceUnavailableReason.NO_ROI
        )
        pipeline = EvidenceExtractionPipeline(
            {
                "required": _FixedEvidence(np.asarray([1.0, 0.0], np.float32)),
                "optional": _FixedEvidence(unavailable),
            },
            {"optional"},
        )
        observations = pipeline.extract_observations(Image.new("RGB", (2, 2)))
        self.assertTrue(observations["required"].is_available)
        self.assertFalse(observations["optional"].is_available)
        self.assertEqual(set(pipeline.extract_all(Image.new("RGB", (2, 2)))), {
            "required"
        })

        required = EvidenceExtractionPipeline(
            {"optional": _FixedEvidence(unavailable)}
        )
        with self.assertRaises(RequiredEvidenceUnavailableError):
            required.extract_all(Image.new("RGB", (2, 2)))

    def test_operational_errors_are_not_converted_to_unavailable(self) -> None:
        pipeline = EvidenceExtractionPipeline(
            {
                "required": _FixedEvidence(np.asarray([1.0, 0.0], np.float32)),
                "optional": _BrokenEvidence(np.asarray([1.0, 0.0], np.float32)),
            },
            {"optional"},
        )
        with self.assertRaisesRegex(RuntimeError, "runtime failed"):
            pipeline.extract_observations(Image.new("RGB", (2, 2)))


class ExactOptionalGalleryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_prebuilt_query_cannot_omit_required_channels(self) -> None:
        gallery = IdentityGallery(
            self.root, dim=4, embedding_contract=_contract()
        )
        try:
            gallery.enroll(
                {"required": np.asarray([1.0, 0.0], np.float32)},
                _dog_id("required-query"),
            )
            query = RetrievalQuery(
                vectors={"optional": np.asarray([1.0, 0.0], np.float32)},
                availability={"required": False, "optional": True},
            )
            with self.assertRaisesRegex(ValueError, "required embedding channels"):
                gallery.search(query)
        finally:
            gallery.close()

    def test_query_vectors_are_immutable_defensive_copies(self) -> None:
        source = np.asarray([1.0, 0.0], np.float32)
        query = RetrievalQuery(
            vectors={"required": source},
            availability={"required": True, "optional": False},
        )
        source[0] = np.nan
        np.testing.assert_array_equal(
            query.vectors["required"], np.asarray([1.0, 0.0], np.float32)
        )
        with self.assertRaisesRegex(ValueError, "read-only"):
            query.vectors["required"][0] = np.nan
        with self.assertRaisesRegex(ValueError, "WRITEABLE"):
            query.vectors["required"].setflags(write=True)

    def test_exact_search_is_exhaustive_and_renormalizes_intersection(self) -> None:
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        try:
            for number in range(20):
                index.enroll(
                    {"required": np.asarray([0.7, 0.714], np.float32)},
                    _dog_id(f"decoy-{number:02d}"),
                    content_sha256=f"{number + 1:064x}",
                )
            target_id = _dog_id("target")
            target_index = index.enroll(
                {
                    "required": np.asarray([-1.0, 0.0], np.float32),
                    "optional": np.asarray([1.0, 0.0], np.float32),
                },
                target_id,
                content_sha256="f" * 64,
            )
            result = index.search({
                "required": np.asarray([1.0, 0.0], np.float32),
                "optional": np.asarray([1.0, 0.0], np.float32),
            }, top_k=1)[0]
            self.assertEqual(result[0], target_index)
            self.assertEqual(result[2]["registered_dog_id"], target_id)
            self.assertTrue(result[2]["_exact"])
            self.assertEqual(len(result[2]["_scorer_hash"]), 64)
            self.assertEqual(result[2]["_evidence_availability"], {
                "required": True, "optional": True,
            })

            decoy = index.explain_identity(
                {"required": np.asarray([1.0, 0.0], np.float32)},
                _dog_id("decoy-00"),
            )
            self.assertIsNotNone(decoy)
            self.assertAlmostEqual(decoy[1], 0.7 / np.linalg.norm([0.7, 0.714]))
            self.assertEqual(decoy[2]["_evidence_availability"], {
                "required": True, "optional": False,
            })
        finally:
            index.close()

    def test_identity_max_uses_one_complete_winning_template(self) -> None:
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        try:
            dog_id = _dog_id("identity-max")
            index.enroll(
                {
                    "required": np.asarray([1.0, 0.0], np.float32),
                    "optional": np.asarray([-1.0, 0.0], np.float32),
                },
                dog_id,
                content_sha256="1" * 64,
            )
            winning = index.enroll(
                {"required": np.asarray([0.8, 0.6], np.float32)},
                dog_id,
                content_sha256="2" * 64,
            )
            result = index.search({
                "required": np.asarray([1.0, 0.0], np.float32),
                "optional": np.asarray([1.0, 0.0], np.float32),
            }, top_k=1)[0]
            self.assertEqual(result[0], winning)
            self.assertEqual(set(result[2]["_evidence"]), {"required"})
            self.assertAlmostEqual(result[1], 0.8)
        finally:
            index.close()

    def test_v4_round_trip_sidecar_and_single_writer(self) -> None:
        dog_one = compute_registered_dog_id("fixture:v1:optional:dog:1")
        dog_two = compute_registered_dog_id("fixture:v1:optional:dog:2")
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        index.enroll(
            {"required": np.asarray([1.0, 0.0], np.float32)}, dog_one
        )
        index.enroll(
            {
                "required": np.asarray([0.0, 1.0], np.float32),
                "optional": np.asarray([1.0, 0.0], np.float32),
            },
            dog_two,
        )
        with self.assertRaisesRegex(RuntimeError, "active writer"):
            IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        index.save()
        manifest = json.loads(
            (self.root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        self.assertEqual(manifest["schema_version"], "cvi.gallery_manifest.v4")
        self.assertTrue(manifest["scorer"]["exact"])
        availability_path = self.root / manifest["files"]["availability"]["name"]
        availability = json.loads(availability_path.read_text(encoding="utf-8"))
        self.assertEqual(availability["0"], {"optional": False, "required": True})
        self.assertEqual(availability["1"], {"optional": True, "required": True})
        loaded = IdentityGallery(
            self.root, dim=4, embedding_contract=_contract(), read_only=True
        )
        try:
            self.assertEqual(loaded.size, 2)
        finally:
            loaded.close()
            index.close()

    def test_oversized_optional_vectors_rejected_before_npz_load(self) -> None:
        dog_id = compute_registered_dog_id("fixture:v1:optional:oversized")
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        index.enroll(
            {
                "required": np.asarray([1.0, 0.0], np.float32),
                "optional": np.asarray([0.0, 1.0], np.float32),
            },
            dog_id,
        )
        index.save()
        manifest = json.loads(
            (self.root / "gallery_manifest.json").read_text(encoding="utf-8")
        )
        optional_vectors = self.root / manifest["files"]["optional_vectors"]["name"]
        maximum_bytes = 1024 * 1024 + 64 * 1024 + 8 + 2 * 4
        with optional_vectors.open("r+b") as stream:
            stream.truncate(maximum_bytes + 1)
        index.close()

        with (
            patch("retrieval.gallery.np.load") as np_load,
            self.assertRaisesRegex(RuntimeError, "optional vectors.*byte limit"),
        ):
            IdentityGallery(
                self.root, dim=4, embedding_contract=_contract(), read_only=True
            )
        np_load.assert_not_called()

    def test_compressed_oversized_npz_member_fails_before_materialization(self) -> None:
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        index.enroll(
            {
                "required": np.asarray([1.0, 0.0], np.float32),
                "optional": np.asarray([0.0, 1.0], np.float32),
            },
            _dog_id("zip-bomb"),
        )
        index.save()
        replacement = b"\0" * (64 * 1024 + 2 * 4 + 1)
        optional_vectors = _replace_optional_npz_member(
            self.root, "c0_vectors.npy", replacement
        )
        self.assertLess(optional_vectors.stat().st_size, len(replacement))
        index.close()

        with (
            patch("retrieval.gallery.np.load") as np_load,
            self.assertRaisesRegex(RuntimeError, "member.*invalid"),
        ):
            IdentityGallery(
                self.root, dim=4, embedding_contract=_contract(), read_only=True
            )
        np_load.assert_not_called()

    def test_forged_huge_npy_shape_fails_before_materialization(self) -> None:
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        index.enroll(
            {
                "required": np.asarray([1.0, 0.0], np.float32),
                "optional": np.asarray([0.0, 1.0], np.float32),
            },
            _dog_id("forged-shape"),
        )
        index.save()
        forged = io.BytesIO()
        np.lib.format.write_array_header_1_0(forged, {
            "descr": np.lib.format.dtype_to_descr(np.dtype(np.float32)),
            "fortran_order": False,
            "shape": (1_000_000, 2),
        })
        _replace_optional_npz_member(
            self.root, "c0_vectors.npy", forged.getvalue()
        )
        index.close()

        with (
            patch("retrieval.gallery.np.load") as np_load,
            self.assertRaisesRegex(RuntimeError, "dtype or shape"),
        ):
            IdentityGallery(
                self.root, dim=4, embedding_contract=_contract(), read_only=True
            )
        np_load.assert_not_called()

    def test_pipeline_returns_auditable_fields_and_exact_explain(self) -> None:
        index = IdentityGallery(self.root, dim=4, embedding_contract=_contract())
        evidence = EvidenceExtractionPipeline(
            {
                "required": _FixedEvidence(np.asarray([1.0, 0.0], np.float32)),
                "optional": _FixedEvidence(EvidenceObservation.unavailable(
                    "optional", EvidenceUnavailableReason.NO_ROI
                )),
            },
            {"optional"},
        )
        pipeline = IdentityRetrievalPipeline(evidence, index)
        image = Image.new("RGB", (2, 2))
        try:
            dog_id = _dog_id("pipeline")
            pipeline.enroll(image, dog_id)
            result = pipeline.search(image, top_k=1)[0]
            self.assertTrue(result.exact)
            self.assertEqual(result.evidence_availability, {
                "required": True, "optional": False,
            })
            self.assertEqual(len(result.scorer_hash), 64)
            explanation = pipeline.explain(image, dog_id)
            self.assertTrue(explanation["exact"])
            self.assertEqual(explanation["template_id"], result.metadata["template_id"])
            serialized = Match(
                dog_id, result.similarity, result.evidence, result.metadata,
                result.evidence_availability, result.scorer_hash, result.exact,
            ).to_dict()
            self.assertTrue(serialized["exact"])
            self.assertIn("evidence_availability", serialized)
        finally:
            index.close()


class ConfigAndMigrationTests(unittest.TestCase):
    def test_config_v2_requires_explicit_optional_and_one_required(self) -> None:
        fixed = {
            "required": _FixedEvidence(np.asarray([1.0, 0.0], np.float32)),
            "optional": _FixedEvidence(np.asarray([0.0, 1.0], np.float32)),
        }
        with tempfile.TemporaryDirectory() as directory, patch.object(
            IdentityEngine, "_build_evidence", return_value=fixed
        ):
            config = {
                "schema_version": "cvi.retrieval_config.v2",
                "mode": "closed_set_retrieval",
                "index_dir": directory,
                "channels": {"required": {}, "optional": {}},
            }
            with self.assertRaisesRegex(ValueError, "explicit optional_channels"):
                IdentityEngine(config)
            with self.assertRaisesRegex(ValueError, "at least one"):
                IdentityEngine({**config, "optional_channels": ["required", "optional"]})
            runtime = IdentityEngine({**config, "optional_channels": ["optional"]})
            try:
                channels = runtime._gallery._embedding_contract["channels"]
                self.assertEqual([channel["optional"] for channel in channels], [
                    False, True,
                ])
                self.assertEqual(
                    runtime._gallery.scorer_hash,
                    runtime._gallery._scorer.scorer_hash,
                )
                self.assertEqual(len(runtime._gallery.scorer_hash), 64)
                self.assertEqual(
                    runtime._gallery._embedding_contract["fusion"]["type"],
                    "exact_available_intersection_weighted_cosine.v1",
                )
            finally:
                runtime._gallery.close()

    def test_legacy_v1_is_all_required_only(self) -> None:
        fixed = {"required": _FixedEvidence(np.asarray([1.0, 0.0], np.float32))}
        with tempfile.TemporaryDirectory() as directory, patch.object(
            IdentityEngine, "_build_evidence", return_value=fixed
        ):
            runtime = IdentityEngine({
                "schema_version": "cvi.retrieval_config.v1",
                "mode": "closed_set_retrieval",
                "index_dir": directory,
                "channels": {"required": {}},
            })
            try:
                self.assertFalse(
                    runtime._gallery._embedding_contract["channels"][0]["optional"]
                )
            finally:
                runtime._gallery.close()

    def test_nose_bundle_schema_is_composite_and_exact(self) -> None:
        runtime = IdentityEngine.__new__(IdentityEngine)
        runtime._config = {
            "channels": {
                "nose": {
                    "type": "nose_print_onnx",
                    "model_path": "single-model-is-not-a-bundle.onnx",
                }
            }
        }
        with self.assertRaisesRegex(ValueError, "exact composite"):
            runtime._build_evidence()

    def test_nose_bundle_loads_all_composite_members(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            detector_manifest_path = root / "detector.json"
            embedding_manifest_path = root / "embedding.json"
            detector_manifest_path.write_text("{}", encoding="utf-8")
            embedding_manifest_path.write_text("{}", encoding="utf-8")
            detector_manifest = object()
            embedding_manifest = object()
            runtime = IdentityEngine.__new__(IdentityEngine)
            runtime._config = {"channels": {"nose": {
                "type": "nose_print_onnx",
                "detector_model_path": str(root / "detector.onnx"),
                "detector_manifest_path": str(detector_manifest_path),
                "embedding_model_path": str(root / "embedding.onnx"),
                "embedding_manifest_path": str(embedding_manifest_path),
                "roi_policy": {
                    "min_box_width": 2,
                    "min_box_height": 2,
                    "min_resolution_width": 8,
                    "min_resolution_height": 8,
                },
                "device": "cpu",
            }}}
            _ConfiguredNoseEvidence.calls.clear()
            with (
                patch(
                    "contracts.artifact_manifest.NoseDetectorManifest.from_dict",
                    return_value=detector_manifest,
                ),
                patch(
                    "contracts.artifact_manifest.NoseEmbeddingManifest.from_dict",
                    return_value=embedding_manifest,
                ),
                patch(
                    "identity_methods.nose.extractor.NosePrintExtractor",
                    _ConfiguredNoseEvidence,
                ),
            ):
                evidence = runtime._build_evidence()
            self.assertEqual(set(evidence), {"nose"})
            arguments, keywords = _ConfiguredNoseEvidence.calls[0]
            self.assertEqual(arguments[0], root / "detector.onnx")
            self.assertIs(arguments[1], detector_manifest)
            self.assertEqual(arguments[2], root / "embedding.onnx")
            self.assertIs(arguments[3], embedding_manifest)
            self.assertFalse(keywords["use_cuda"])

    def test_v3_migration_preserves_stored_vectors_and_requires_new_output(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "v3"
            output = root / "v4"
            source.mkdir()
            index = faiss.IndexFlatIP(2)
            vector = np.asarray([[0.6, 0.8]], dtype=np.float32)
            index.add(vector)
            index_path = source / "master-generation.idx"
            metadata_path = source / "metadata-generation.json"
            breeds_path = source / "breeds-generation.json"
            faiss.write_index(index, str(index_path))
            content_hash = "a" * 64
            template_id = hashlib.sha256(
                f"cvi.gallery_template.v1\0{content_hash}".encode("ascii")
            ).hexdigest()
            registered_dog_id = compute_registered_dog_id(
                "fixture:v1:migration:dog"
            )
            metadata_path.write_text(json.dumps({"0": {
                "registered_dog_id": registered_dog_id,
                "template_id": template_id,
                "content_sha256": content_hash,
                "idempotency_key": "request",
                "template_schema": "cvi.gallery_template.v1",
                "metadata": {},
            }}), encoding="utf-8")
            breeds_path.write_text(json.dumps({"0": "unknown"}), encoding="utf-8")
            files = {}
            for kind, path in (
                ("index", index_path), ("metadata", metadata_path),
                ("breeds", breeds_path),
            ):
                files[kind] = {
                    "name": path.name,
                    "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                }
            (source / "gallery_manifest.json").write_text(json.dumps({
                "schema_version": "cvi.gallery_manifest.v3",
                "dimension": 2,
                "embedding_contract": {
                    "schema_version": "cvi.gallery_embedding_contract.v1",
                    "kind": "opaque",
                    "dimension": 2,
                },
                "count": 1,
                "template_count": 1,
                "identity_count": 1,
                "identity_aggregation": "max",
                "files": files,
            }), encoding="utf-8")

            source_snapshot = {
                path.name: path.read_bytes() for path in source.iterdir()
            }
            migrate_gallery(source, output)
            self.assertEqual(
                {path.name: path.read_bytes() for path in source.iterdir()},
                source_snapshot,
            )
            self.assertEqual(output.stat().st_mode & 0o777, 0o700)
            migrated = IdentityGallery(output, dim=2, read_only=True)
            try:
                np.testing.assert_array_equal(migrated._index.reconstruct(0), vector[0])
            finally:
                migrated.close()
            with self.assertRaisesRegex(ValueError, "new, non-existing"):
                migrate_gallery(source, output)


if __name__ == "__main__":
    unittest.main()
