from __future__ import annotations

import json
import struct
import unittest
from contextlib import redirect_stdout
from dataclasses import replace
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data.acquisition import sha256_file
from data.crop_export import CropExportReceipt
from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    ArtifactSourceKind,
    ControlBlindScoreReceipt,
    ControlScorePolicy,
    ControlScoringInventory,
    EmbeddingCacheEntry,
    EmbeddingCacheManifest,
    EmbeddingCachePolicy,
    build_control_scoring_inventory,
    embedding_cache_key,
    score_control_requests_from_cache,
    verify_embedding_cache_files,
)
from evaluation.controls.control_transform import (
    ControlArtifactManifest,
    ControlTransformCost,
    ControlTransformReceipt,
    verify_control_artifact_files,
)
from evaluation.controls.policy import ControlScoringRequest
from evaluation.controls.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    verify_pair_artifact_files,
)
from foundation.provenance import content_sha256
from workflows.evaluate_visual_controls import main

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64
HASH_D = "d" * 64
HASH_E = "e" * 64


def _entry(root: Path, token: str, payload: bytes) -> PairArtifactEntry:
    path = root / f"{token}.png"
    path.write_bytes(payload)
    return PairArtifactEntry(
        artifact_token=token,
        relative_path=path.name,
        content_sha256=sha256_file(path),
        byte_size=path.stat().st_size,
        media_type="image/png",
    )


def _write_vector(path: Path, values: tuple[float, ...]) -> None:
    path.write_bytes(struct.pack(f"<{len(values)}f", *values))


class ControlScoringTests(unittest.TestCase):
    def _inventory_fixture(
        self,
        root: Path,
    ) -> tuple[
        tuple[ControlScoringRequest, ...],
        ControlScoringInventory,
        Path,
        PairArtifactManifest,
        object,
        Path,
        ControlTransformReceipt,
    ]:
        base_root = root / "base"
        control_root = root / "control"
        base_root.mkdir()
        control_root.mkdir()
        base_manifest = PairArtifactManifest(
            pair_set_sha256=HASH_A,
            artifact_bindings_sha256=HASH_B,
            entries=(
                _entry(base_root, "base-a", b"shared-pixels"),
                _entry(base_root, "base-b", b"base-b-pixels"),
            ),
        )
        base_verification = verify_pair_artifact_files(
            base_root,
            base_manifest,
        )
        control_entries = (
            _entry(control_root, "control-c", b"shared-pixels"),
            _entry(control_root, "control-d", b"control-d-pixels"),
        )
        control_manifest = ControlArtifactManifest(
            transform_tasks_sha256=HASH_C,
            transform_config_manifest_sha256=HASH_D,
            entries=control_entries,
        )
        control_verification = verify_control_artifact_files(
            control_root,
            control_manifest,
        )
        requests = (
            ControlScoringRequest("request-1", "base-a", "base-b"),
            ControlScoringRequest(
                "request-2",
                "control-c",
                "control-d",
            ),
        )
        scoring_requests_sha256 = content_sha256(
            {
                "schema_version": (
                    "cvi.visual_control_scoring_requests.v1"
                ),
                "plan_sha256": HASH_A,
                "requests": [request.to_dict() for request in requests],
            }
        )
        transform_receipt = ControlTransformReceipt(
            plan_sha256=HASH_A,
            scoring_requests_sha256=scoring_requests_sha256,
            transform_tasks_sha256=HASH_C,
            base_artifact_manifest_sha256=base_manifest.manifest_sha256,
            base_artifact_verification_sha256=content_sha256(
                base_verification.to_dict()
            ),
            mask_manifest_sha256=HASH_B,
            mask_verification_sha256=HASH_C,
            mask_semantic_verification_sha256=HASH_D,
            transform_config_manifest_sha256=HASH_D,
            execution_policy_sha256=HASH_E,
            ffmpeg_version="fixture",
            artifact_manifest=control_manifest,
            verification=control_verification,
            cost=ControlTransformCost(
                transform_tasks=2,
                unique_base_decodes=2,
                unique_mask_decodes=2,
                validation_blur_decodes=0,
                subprocess_calls=8,
                total_task_pixels=2,
                output_bytes=control_verification.verified_bytes,
                peak_validation_raw_bytes=2,
            ),
        )
        inventory = build_control_scoring_inventory(
            plan_sha256=HASH_A,
            requests=requests,
            base_root=base_root,
            base_manifest=base_manifest,
            base_verification=base_verification,
            control_root=control_root,
            transform_receipt=transform_receipt,
        )
        return (
            requests,
            inventory,
            base_root,
            base_manifest,
            base_verification,
            control_root,
            transform_receipt,
        )

    def _cache_fixture(
        self,
        root: Path,
        inventory: ControlScoringInventory,
    ) -> tuple[
        Path,
        EmbeddingCacheManifest,
        EmbeddingCachePolicy,
        object,
    ]:
        cache_root = root / "cache"
        cache_root.mkdir()
        vectors = {
            "shared-pixels": (1.0, 0.0),
            "base-b-pixels": (0.0, 1.0),
            "control-d-pixels": (-1.0, 0.0),
        }
        vector_by_content = {
            content_sha256: vectors[
                {
                    sha256_file(root / "base" / "base-a.png"): (
                        "shared-pixels"
                    ),
                    sha256_file(root / "base" / "base-b.png"): (
                        "base-b-pixels"
                    ),
                    sha256_file(root / "control" / "control-d.png"): (
                        "control-d-pixels"
                    ),
                }[content_sha256]
            ]
            for content_sha256 in {
                entry.content_sha256 for entry in inventory.entries
            }
        }
        bindings = []
        entries_by_key: dict[str, EmbeddingCacheEntry] = {}
        for artifact in inventory.entries:
            key = embedding_cache_key(
                artifact_content_sha256=artifact.content_sha256,
                model_sha256=HASH_A,
                inference_config_sha256=HASH_B,
                dependency_lock_sha256=HASH_C,
                code_revision="fixture-revision",
                precision="fp32",
                vector_dimension=2,
            )
            bindings.append(
                ArtifactCacheBinding(
                    artifact.artifact_token,
                    artifact.content_sha256,
                    key,
                )
            )
            if key in entries_by_key:
                continue
            path = cache_root / f"{key}.f32le"
            _write_vector(path, vector_by_content[artifact.content_sha256])
            entries_by_key[key] = EmbeddingCacheEntry(
                cache_key=key,
                relative_path=path.name,
                content_sha256=sha256_file(path),
                byte_size=path.stat().st_size,
            )
        manifest = EmbeddingCacheManifest(
            scoring_inventory_sha256=inventory.inventory_sha256,
            model_sha256=HASH_A,
            inference_config_sha256=HASH_B,
            dependency_lock_sha256=HASH_C,
            code_revision="fixture-revision",
            precision="fp32",
            vector_dimension=2,
            normalization_tolerance=1e-6,
            bindings=tuple(bindings),
            entries=tuple(
                entries_by_key[key] for key in sorted(entries_by_key)
            ),
        )
        policy = EmbeddingCachePolicy(
            maximum_artifacts=4,
            maximum_unique_vectors=3,
            maximum_vector_dimension=2,
            maximum_vector_bytes=8,
            maximum_total_cache_bytes=24,
            scan_chunk_floats=1,
        )
        verification = verify_embedding_cache_files(
            root=cache_root,
            inventory=inventory,
            manifest=manifest,
            policy=policy,
        )
        return cache_root, manifest, policy, verification

    def test_inventory_cache_dedup_and_blind_scores_are_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                requests,
                inventory,
                _,
                _,
                _,
                _,
                _,
            ) = self._inventory_fixture(root)
            self.assertEqual(
                ControlScoringInventory.from_dict(inventory.to_dict()),
                inventory,
            )
            self.assertEqual(len(inventory.entries), 4)
            self.assertEqual(
                {entry.source_kind for entry in inventory.entries},
                {ArtifactSourceKind.BASE, ArtifactSourceKind.CONTROL},
            )
            cache_root, manifest, cache_policy, verification = (
                self._cache_fixture(root, inventory)
            )
            self.assertEqual(
                EmbeddingCacheManifest.from_dict(manifest.to_dict()),
                manifest,
            )
            self.assertEqual(len(manifest.bindings), 4)
            self.assertEqual(len(manifest.entries), 3)
            receipt = score_control_requests_from_cache(
                requests=requests,
                inventory=inventory,
                cache_root=cache_root,
                cache_manifest=manifest,
                cache_verification=verification,
                cache_policy=cache_policy,
                score_policy=ControlScorePolicy(
                    maximum_requests=2,
                    maximum_scalar_products=4,
                    maximum_embedding_bytes_read=32,
                    dot_chunk_floats=1,
                ),
                gallery_sha256=HASH_D,
            )
            self.assertEqual(
                tuple(score.score for score in receipt.scores),
                (0.0, -1.0),
            )
            self.assertEqual(
                receipt.cost.dot_product_scalar_products,
                4,
            )
            self.assertEqual(
                receipt.cost.cache_verification_square_terms,
                12,
            )
            self.assertEqual(receipt.cost.dot_product_bytes_read, 32)
            self.assertEqual(
                receipt.cost.cache_verification_bytes_read,
                48,
            )
            self.assertEqual(receipt.cost.total_file_bytes_read, 80)
            self.assertEqual(receipt.cost.unique_artifacts, 4)
            self.assertEqual(receipt.cost.unique_embedding_vectors, 3)
            self.assertEqual(receipt.cost.neural_embedding_calls_saved, 1)
            self.assertEqual(receipt.cost.peak_raw_chunk_bytes, 8)
            self.assertEqual(
                ControlBlindScoreReceipt.from_dict(receipt.to_dict()),
                receipt,
            )
            receipt_text = str(receipt.to_dict())
            self.assertNotIn("dog", receipt_text)
            self.assertNotIn("session", receipt_text)

    def test_non_normalized_vector_is_rejected_even_with_fresh_hash(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                _,
                inventory,
                _,
                _,
                _,
                _,
                _,
            ) = self._inventory_fixture(root)
            cache_root, manifest, policy, _ = self._cache_fixture(
                root,
                inventory,
            )
            bad_entry = manifest.entries[0]
            bad_path = cache_root / bad_entry.relative_path
            _write_vector(bad_path, (2.0, 0.0))
            changed = replace(
                bad_entry,
                content_sha256=sha256_file(bad_path),
            )
            bad_manifest = replace(
                manifest,
                entries=tuple(
                    changed if entry.cache_key == changed.cache_key else entry
                    for entry in manifest.entries
                ),
            )
            with self.assertRaisesRegex(ValueError, "L2-normalized"):
                verify_embedding_cache_files(
                    root=cache_root,
                    inventory=inventory,
                    manifest=bad_manifest,
                    policy=policy,
                )

    def test_cache_provenance_and_resource_caps_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                _,
                inventory,
                _,
                _,
                _,
                _,
                _,
            ) = self._inventory_fixture(root)
            cache_root, manifest, policy, _ = self._cache_fixture(
                root,
                inventory,
            )
            first = manifest.bindings[0]
            with self.assertRaisesRegex(ValueError, "provenance"):
                replace(
                    manifest,
                    bindings=(
                        replace(first, artifact_content_sha256=HASH_E),
                    )
                    + manifest.bindings[1:],
                )
            with self.assertRaisesRegex(ValueError, "dimension"):
                verify_embedding_cache_files(
                    root=cache_root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=replace(policy, maximum_vector_dimension=1),
                )

    def test_request_recombination_breaks_transform_receipt_binding(
        self,
    ) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                requests,
                _,
                base_root,
                base_manifest,
                base_verification,
                control_root,
                transform_receipt,
            ) = self._inventory_fixture(root)
            tampered = (
                replace(
                    requests[0],
                    reference_artifact_token="control-d",
                ),
                requests[1],
            )
            with self.assertRaisesRegex(
                ValueError,
                "transform receipt binding",
            ):
                build_control_scoring_inventory(
                    plan_sha256=HASH_A,
                    requests=tampered,
                    base_root=base_root,
                    base_manifest=base_manifest,
                    base_verification=base_verification,
                    control_root=control_root,
                    transform_receipt=transform_receipt,
                )

    def test_cli_builds_label_blind_inventory_and_score_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                requests,
                inventory,
                base_root,
                base_manifest,
                base_verification,
                control_root,
                transform_receipt,
            ) = self._inventory_fixture(root)
            cache_root, cache_manifest, cache_policy, _ = (
                self._cache_fixture(root, inventory)
            )
            crop_receipt = CropExportReceipt(
                pair_set_sha256=base_manifest.pair_set_sha256,
                source_manifest_sha256=HASH_C,
                export_policy_sha256=HASH_D,
                ffmpeg_version="fixture",
                artifact_manifest=base_manifest,
                verification=base_verification,
            )
            score_policy = ControlScorePolicy(
                maximum_requests=2,
                maximum_scalar_products=4,
                maximum_embedding_bytes_read=32,
                dot_chunk_floats=1,
            )
            payloads = {
                "requests": {
                    "schema_version": (
                        "cvi.visual_control_scoring_requests.v1"
                    ),
                    "plan_sha256": HASH_A,
                    "requests": [request.to_dict() for request in requests],
                },
                "crop": crop_receipt.to_dict(),
                "transform": transform_receipt.to_dict(),
                "cache": cache_manifest.to_dict(),
                "cache-policy": cache_policy.to_dict(),
                "score-policy": score_policy.to_dict(),
            }
            paths: dict[str, Path] = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(json.dumps(payload), encoding="utf-8")
                paths[name] = path
            protected = root / "protected"
            protected.mkdir()
            outputs = tuple(
                protected / name
                for name in (
                    "inventory.json",
                    "cache-verification.json",
                    "scores.json",
                )
            )
            argv = [
                "evaluate_visual_controls.py",
                "score",
                "--scoring-requests",
                str(paths["requests"]),
                "--crop-export-receipt",
                str(paths["crop"]),
                "--base-artifact-directory",
                str(base_root),
                "--control-transform-receipt",
                str(paths["transform"]),
                "--control-artifact-directory",
                str(control_root),
                "--embedding-cache-manifest",
                str(paths["cache"]),
                "--embedding-cache-directory",
                str(cache_root),
                "--embedding-cache-policy",
                str(paths["cache-policy"]),
                "--score-policy",
                str(paths["score-policy"]),
                "--gallery-sha256",
                HASH_D,
                "--inventory-output",
                str(outputs[0]),
                "--cache-verification-output",
                str(outputs[1]),
                "--score-receipt-output",
                str(outputs[2]),
            ]
            stdout = StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                main()
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "CREATED")
            self.assertEqual(summary["unique_artifacts"], 4)
            self.assertEqual(summary["unique_embedding_vectors"], 3)
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertEqual(output.stat().st_mode & 0o777, 0o600)
            self.assertNotIn(
                "dog",
                outputs[2].read_text(encoding="utf-8"),
            )


if __name__ == "__main__":
    unittest.main()
