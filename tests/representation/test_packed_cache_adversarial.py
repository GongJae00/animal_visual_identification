from __future__ import annotations

import hashlib
import os
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any

from evaluation.controls.control_scoring import (
    ArtifactCacheBinding,
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
    embedding_cache_key,
)
from evaluation.integrity.packed_cache import (
    PACKED_VECTOR_FILE_NAME,
    PackedEmbeddingCacheEntry,
    PackedEmbeddingCacheManifest,
    PackedEmbeddingCacheStorage,
    PackedEmbeddingCacheVerification,
    verify_packed_embedding_cache_files,
)

_MAX_SIGNED_OFFSET = (1 << 63) - 1

def _digest(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()

class PackedCacheAdversarialTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
    ) -> tuple[
        ControlScoringInventory,
        PackedEmbeddingCacheManifest,
        EmbeddingCachePolicy,
    ]:
        content_by_token = {
            "artifact-a": _digest("artifact-a-content"),
            "artifact-b": _digest("artifact-b-content"),
        }
        inventory = ControlScoringInventory(
            plan_sha256=_digest("plan"),
            scoring_requests_sha256=_digest("requests"),
            base_artifact_manifest_sha256=_digest("base-manifest"),
            base_artifact_verification_sha256=_digest("base-verification"),
            control_transform_receipt_sha256=_digest("transform-receipt"),
            entries=tuple(
                ScoringArtifactEntry(
                    artifact_token=token,
                    content_sha256=content,
                    byte_size=64,
                    source_kind=ArtifactSourceKind.BASE,
                )
                for token, content in content_by_token.items()
            ),
        )
        provenance: dict[str, Any] = {
            "model_sha256": _digest("model"),
            "inference_config_sha256": _digest("inference-config"),
            "dependency_lock_sha256": _digest("dependency-lock"),
            "code_revision": "packed-adversarial-test",
            "precision": "FP32",
            "vector_dimension": 2,
        }
        key_by_content = {
            content: embedding_cache_key(
                artifact_content_sha256=content,
                **provenance,
            )
            for content in content_by_token.values()
        }
        vector_by_key = {
            key_by_content[content_by_token["artifact-a"]]: struct.pack(
                "<2f", 1.0, 0.0
            ),
            key_by_content[content_by_token["artifact-b"]]: struct.pack(
                "<2f", 0.0, -1.0
            ),
        }
        ordered_vectors = tuple(sorted(vector_by_key.items()))
        pack = b"".join(payload for _, payload in ordered_vectors)
        root.mkdir()
        (root / PACKED_VECTOR_FILE_NAME).write_bytes(pack)
        entries = tuple(
            PackedEmbeddingCacheEntry(
                cache_key=cache_key,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                byte_offset=ordinal * 8,
                byte_size=8,
            )
            for ordinal, (cache_key, payload) in enumerate(ordered_vectors)
        )
        bindings = tuple(
            ArtifactCacheBinding(
                artifact_token=entry.artifact_token,
                artifact_content_sha256=entry.content_sha256,
                cache_key=key_by_content[entry.content_sha256],
            )
            for entry in inventory.entries
        )
        manifest = PackedEmbeddingCacheManifest(
            scoring_inventory_sha256=inventory.inventory_sha256,
            normalization_tolerance=1e-6,
            storage=PackedEmbeddingCacheStorage(
                content_sha256=hashlib.sha256(pack).hexdigest(),
                byte_size=len(pack),
                vector_count=len(entries),
                vector_stride_bytes=8,
            ),
            bindings=bindings,
            entries=entries,
            **provenance,
        )
        return inventory, manifest, EmbeddingCachePolicy()

    def test_signed_offset_and_derived_arithmetic_overflow_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            _, manifest, _ = self._fixture(Path(temporary) / "cache")
            with self.assertRaisesRegex(ValueError, "signed offset range"):
                PackedEmbeddingCacheEntry(
                    cache_key=_digest("key"),
                    content_sha256=_digest("content"),
                    byte_offset=_MAX_SIGNED_OFFSET,
                    byte_size=1,
                )
            with self.assertRaisesRegex(ValueError, "signed offset range"):
                replace(
                    manifest.storage,
                    byte_size=_MAX_SIGNED_OFFSET + 1,
                )
            with self.assertRaisesRegex(ValueError, "stride exceeds offset range"):
                replace(
                    manifest,
                    vector_dimension=1 << 61,
                    storage=replace(
                        manifest.storage,
                        vector_stride_bytes=1 << 63,
                    ),
                )

    def test_changed_provenance_cannot_reuse_existing_cache_keys(self) -> None:
        with TemporaryDirectory() as temporary:
            _, manifest, _ = self._fixture(Path(temporary) / "cache")
            for field, value in (
                ("model_sha256", _digest("different-model")),
                ("inference_config_sha256", _digest("different-config")),
                ("dependency_lock_sha256", _digest("different-lock")),
                ("code_revision", "different-revision"),
                ("precision", "FP16"),
            ):
                with self.subTest(field=field), self.assertRaisesRegex(
                    ValueError,
                    "provenance mismatch",
                ):
                    replace(manifest, **{field: value})

    def test_whole_pack_hash_is_checked_after_valid_slice_hashes(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            wrong_whole_hash = replace(
                manifest,
                storage=replace(
                    manifest.storage,
                    content_sha256=_digest("forged-whole-pack"),
                ),
            )
            with self.assertRaisesRegex(ValueError, "whole-file hash mismatch"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=wrong_whole_hash,
                    policy=policy,
                )

    def test_inventory_identity_coverage_and_content_are_all_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)

            other_inventory = replace(
                inventory,
                plan_sha256=_digest("other-plan"),
            )
            with self.assertRaisesRegex(ValueError, "another inventory"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=other_inventory,
                    manifest=manifest,
                    policy=policy,
                )

            extra_inventory = replace(
                inventory,
                entries=inventory.entries
                + (
                    ScoringArtifactEntry(
                        artifact_token="artifact-c",
                        content_sha256=_digest("artifact-c-content"),
                        byte_size=64,
                        source_kind=ArtifactSourceKind.CONTROL,
                    ),
                ),
            )
            extra_inventory_manifest = replace(
                manifest,
                scoring_inventory_sha256=extra_inventory.inventory_sha256,
            )
            with self.assertRaisesRegex(ValueError, "do not cover inventory"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=extra_inventory,
                    manifest=extra_inventory_manifest,
                    policy=policy,
                )

            changed_entries = (
                replace(
                    inventory.entries[0],
                    content_sha256=_digest("substituted-artifact-content"),
                ),
                inventory.entries[1],
            )
            changed_inventory = replace(inventory, entries=changed_entries)
            changed_inventory_manifest = replace(
                manifest,
                scoring_inventory_sha256=changed_inventory.inventory_sha256,
            )
            with self.assertRaisesRegex(ValueError, "artifact content mismatch"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=changed_inventory,
                    manifest=changed_inventory_manifest,
                    policy=policy,
                )

    def test_each_resource_policy_limit_is_enforced(self) -> None:
        cases = (
            ("maximum_artifacts", 1, "bindings exceed policy"),
            ("maximum_unique_vectors", 1, "vectors exceed policy"),
            ("maximum_vector_dimension", 1, "dimension exceeds policy"),
            ("maximum_vector_bytes", 4, "vector bytes exceed policy"),
            ("maximum_total_cache_bytes", 8, "cache bytes exceed policy"),
            (
                "maximum_normalization_tolerance",
                5e-7,
                "normalization tolerance exceeds policy",
            ),
        )
        for field, limit, message in cases:
            with self.subTest(field=field), TemporaryDirectory() as temporary:
                root = Path(temporary) / "cache"
                inventory, manifest, policy = self._fixture(root)
                with self.assertRaisesRegex(ValueError, message):
                    verify_packed_embedding_cache_files(
                        root=root,
                        inventory=inventory,
                        manifest=manifest,
                        policy=replace(policy, **{field: limit}),
                    )

    def test_symlink_pack_is_never_followed(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "cache"
            inventory, manifest, policy = self._fixture(root)
            pack_path = root / PACKED_VECTOR_FILE_NAME
            external = base / "external.pack"
            external.write_bytes(pack_path.read_bytes())
            pack_path.unlink()
            pack_path.symlink_to(external)
            with self.assertRaises(OSError):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                )

    def test_path_replacement_truncation_and_open_directory_are_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            base = Path(temporary)
            root = base / "cache"
            inventory, manifest, policy = self._fixture(root)
            pack_path = root / PACKED_VECTOR_FILE_NAME

            def replace_path_after_open(phase: str) -> None:
                if phase == "PACK_OPENED":
                    held = base / "opened-original.pack"
                    pack_path.rename(held)
                    pack_path.write_bytes(held.read_bytes())

            with self.assertRaisesRegex(
                RuntimeError,
                "changed during|no longer names",
            ):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                    verification_phase_callback=replace_path_after_open,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            pack_path = root / PACKED_VECTOR_FILE_NAME

            def truncate_after_open(phase: str) -> None:
                if phase == "PACK_OPENED":
                    os.truncate(pack_path, 1)

            with self.assertRaisesRegex(ValueError, "slice is truncated"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                    verification_phase_callback=truncate_after_open,
                )

        for extra_name in ("manifest.json", ".staging", "extra-directory"):
            with (
                self.subTest(extra_name=extra_name),
                TemporaryDirectory() as temporary,
            ):
                root = Path(temporary) / "cache"
                inventory, manifest, policy = self._fixture(root)
                extra = root / extra_name
                if extra_name == "extra-directory":
                    extra.mkdir()
                else:
                    extra.write_bytes(b"undeclared")
                with self.assertRaisesRegex(ValueError, "not a closed set"):
                    verify_packed_embedding_cache_files(
                        root=root,
                        inventory=inventory,
                        manifest=manifest,
                        policy=policy,
                    )

    def test_malformed_or_downgraded_schemas_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            _, manifest, _ = self._fixture(Path(temporary) / "cache")
            valid = manifest.to_dict()

            malformed_manifests = (
                ({**valid, "schema_version": "operations.embedding_cache_manifest.v1"},
                 "unsupported packed embedding cache manifest schema"),
                ({**valid, "unexpected": True}, "fields differ"),
                ({key: value for key, value in valid.items() if key != "entries"},
                 "fields differ"),
                ({**valid, "storage": []}, "storage must be an object"),
                ({**valid, "bindings": {}}, "bindings and entries must be lists"),
                ({**valid, "entries": {}}, "bindings and entries must be lists"),
                ({
                    **valid,
                    "storage": {
                        **valid["storage"],
                        "relative_path": "../vectors.f32le.pack",
                    },
                }, "storage path is fixed"),
            )
            for payload, message in malformed_manifests:
                with self.subTest(message=message), self.assertRaisesRegex(
                    (TypeError, ValueError),
                    message,
                ):
                    PackedEmbeddingCacheManifest.from_dict(payload)

            entry = manifest.entries[0].to_dict()
            for payload in (
                {**entry, "byte_offset": True},
                {**entry, "content_sha256": entry["content_sha256"].upper()},
                {**entry, "extra": 0},
            ):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    PackedEmbeddingCacheEntry.from_dict(payload)

            verification = PackedEmbeddingCacheVerification(
                cache_manifest_sha256=manifest.manifest_sha256,
                logical_cache_sha256=manifest.logical_cache_sha256,
                cache_policy_sha256=EmbeddingCachePolicy().policy_sha256,
                observed_pack_sha256=manifest.storage.content_sha256,
                verified_files=1,
                verified_bytes=manifest.storage.byte_size,
                verified_vectors=len(manifest.entries),
                maximum_observed_norm_error=0.0,
            )
            for payload in (
                {
                    **verification.to_dict(),
                    "schema_version": "operations.embedding_cache_verification.v1",
                },
                {**verification.to_dict(), "unexpected": True},
            ):
                with self.subTest(payload=payload), self.assertRaises(ValueError):
                    PackedEmbeddingCacheVerification.from_dict(payload)

if __name__ == "__main__":
    unittest.main()
