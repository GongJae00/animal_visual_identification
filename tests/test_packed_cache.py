from __future__ import annotations

import hashlib
import os
import struct
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.control_scoring import (
    ArtifactCacheBinding,
    ArtifactSourceKind,
    ControlScoringInventory,
    EmbeddingCachePolicy,
    ScoringArtifactEntry,
    embedding_cache_key,
)
from cvi.packed_cache import (
    PACKED_VECTOR_FILE_NAME,
    PackedEmbeddingCacheEntry,
    PackedEmbeddingCacheManifest,
    PackedEmbeddingCacheStorage,
    PackedEmbeddingCacheVerification,
    verify_packed_embedding_cache_files,
)
from cvi.provenance import content_sha256


def _opaque(label: str) -> str:
    return hashlib.sha256(label.encode("utf-8")).hexdigest()


class PackedCacheTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
        *,
        vectors: dict[str, bytes] | None = None,
    ) -> tuple[
        ControlScoringInventory,
        PackedEmbeddingCacheManifest,
        EmbeddingCachePolicy,
    ]:
        content_a = _opaque("artifact-content-a")
        content_b = _opaque("artifact-content-b")
        inventory = ControlScoringInventory(
            plan_sha256=_opaque("plan"),
            scoring_requests_sha256=_opaque("requests"),
            base_artifact_manifest_sha256=_opaque("base-manifest"),
            base_artifact_verification_sha256=_opaque("base-verification"),
            control_transform_receipt_sha256=_opaque("transform"),
            entries=(
                ScoringArtifactEntry(
                    "artifact-a",
                    content_a,
                    10,
                    ArtifactSourceKind.BASE,
                ),
                ScoringArtifactEntry(
                    "artifact-b",
                    content_b,
                    11,
                    ArtifactSourceKind.BASE,
                ),
                ScoringArtifactEntry(
                    "artifact-c-alias",
                    content_a,
                    10,
                    ArtifactSourceKind.CONTROL,
                ),
            ),
        )
        model_sha256 = _opaque("model")
        inference_config_sha256 = _opaque("config")
        dependency_lock_sha256 = _opaque("lock")
        key_by_content = {
            content: embedding_cache_key(
                artifact_content_sha256=content,
                model_sha256=model_sha256,
                inference_config_sha256=inference_config_sha256,
                dependency_lock_sha256=dependency_lock_sha256,
                code_revision="packed-format-test",
                precision="FP32",
                vector_dimension=3,
            )
            for content in (content_a, content_b)
        }
        default_vectors = {
            key_by_content[content_a]: struct.pack("<3f", 1.0, -0.0, 0.0),
            key_by_content[content_b]: struct.pack("<3f", 0.0, 1.0, 0.0),
        }
        payloads = default_vectors if vectors is None else vectors
        ordered = tuple(sorted(payloads.items()))
        pack = b"".join(payload for _, payload in ordered)
        root.mkdir()
        (root / PACKED_VECTOR_FILE_NAME).write_bytes(pack)
        entries = tuple(
            PackedEmbeddingCacheEntry(
                cache_key=cache_key,
                content_sha256=hashlib.sha256(payload).hexdigest(),
                byte_offset=ordinal * 12,
                byte_size=12,
            )
            for ordinal, (cache_key, payload) in enumerate(ordered)
        )
        bindings = tuple(
            ArtifactCacheBinding(
                artifact_token=item.artifact_token,
                artifact_content_sha256=item.content_sha256,
                cache_key=key_by_content[item.content_sha256],
            )
            for item in inventory.entries
        )
        manifest = PackedEmbeddingCacheManifest(
            scoring_inventory_sha256=inventory.inventory_sha256,
            model_sha256=model_sha256,
            inference_config_sha256=inference_config_sha256,
            dependency_lock_sha256=dependency_lock_sha256,
            code_revision="packed-format-test",
            precision="FP32",
            vector_dimension=3,
            normalization_tolerance=1e-6,
            storage=PackedEmbeddingCacheStorage(
                content_sha256=hashlib.sha256(pack).hexdigest(),
                byte_size=len(pack),
                vector_count=len(entries),
                vector_stride_bytes=12,
            ),
            bindings=bindings,
            entries=entries,
        )
        return inventory, manifest, EmbeddingCachePolicy()

    def test_valid_pack_round_trip_and_alias_verification(self) -> None:
        with TemporaryDirectory() as temporary:
            inventory, manifest, policy = self._fixture(
                Path(temporary) / "cache"
            )
            parsed = PackedEmbeddingCacheManifest.from_dict(manifest.to_dict())
            self.assertEqual(parsed, manifest)
            verification = verify_packed_embedding_cache_files(
                root=Path(temporary) / "cache",
                inventory=inventory,
                manifest=parsed,
                policy=policy,
            )
            self.assertEqual(verification.verified_files, 1)
            self.assertEqual(verification.verified_vectors, 2)
            self.assertEqual(verification.verified_bytes, 24)
            self.assertEqual(
                PackedEmbeddingCacheVerification.from_dict(
                    verification.to_dict()
                ),
                verification,
            )

    def test_manifest_rejects_noncanonical_layout_and_boolean_offset(self) -> None:
        with TemporaryDirectory() as temporary:
            _, manifest, _ = self._fixture(Path(temporary) / "cache")
            with self.assertRaisesRegex(ValueError, "offset is not canonical"):
                replace(
                    manifest,
                    entries=(
                        replace(manifest.entries[0], byte_offset=12),
                        manifest.entries[1],
                    ),
                )
            with self.assertRaisesRegex(ValueError, "cache-key-sorted"):
                replace(manifest, entries=tuple(reversed(manifest.entries)))
            with self.assertRaisesRegex(ValueError, "non-negative integer"):
                PackedEmbeddingCacheEntry.from_dict(
                    {
                        **manifest.entries[0].to_dict(),
                        "byte_offset": False,
                    }
                )
            with self.assertRaisesRegex(ValueError, "storage byte size differs"):
                replace(
                    manifest,
                    storage=replace(
                        manifest.storage,
                        byte_size=manifest.storage.byte_size + 1,
                    ),
                )

    def test_verifier_rejects_extra_symlink_trailing_and_policy_overflow(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            (root / "extra").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "not a closed set"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                )
            (root / "extra").unlink()
            symlink_root = Path(temporary) / "cache-link"
            symlink_root.symlink_to(root, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                verify_packed_embedding_cache_files(
                    root=symlink_root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                )
            with self.assertRaisesRegex(ValueError, "vectors exceed policy"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=replace(policy, maximum_unique_vectors=1),
                )
            pack_path = root / PACKED_VECTOR_FILE_NAME
            pack_path.write_bytes(pack_path.read_bytes() + b"trailing")
            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                )

    def test_verifier_rejects_slice_hash_nonfinite_and_norm_failures(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            pack_path = root / PACKED_VECTOR_FILE_NAME
            payload = bytearray(pack_path.read_bytes())
            payload[0] ^= 1
            pack_path.write_bytes(payload)
            with self.assertRaisesRegex(ValueError, "slice hash mismatch"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                )

        for bad_vector, message in (
            (struct.pack("<3f", float("nan"), 0.0, 0.0), "malformed"),
            (struct.pack("<3f", 0.5, 0.0, 0.0), "not L2-normalized"),
        ):
            with self.subTest(message=message), TemporaryDirectory() as temporary:
                root = Path(temporary) / "cache"
                inventory, manifest, policy = self._fixture(root)
                pack_path = root / PACKED_VECTOR_FILE_NAME
                original = pack_path.read_bytes()
                changed = bad_vector + original[12:]
                pack_path.write_bytes(changed)
                entries = (
                    replace(
                        manifest.entries[0],
                        content_sha256=hashlib.sha256(bad_vector).hexdigest(),
                    ),
                    manifest.entries[1],
                )
                changed_manifest = replace(
                    manifest,
                    entries=entries,
                    storage=replace(
                        manifest.storage,
                        content_sha256=hashlib.sha256(changed).hexdigest(),
                    ),
                )
                with self.assertRaisesRegex(ValueError, message):
                    verify_packed_embedding_cache_files(
                        root=root,
                        inventory=inventory,
                        manifest=changed_manifest,
                        policy=policy,
                    )

    def test_open_file_descriptor_detects_path_replacement_and_mutation(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            original = root / PACKED_VECTOR_FILE_NAME

            def replace_after_open(phase: str) -> None:
                if phase != "PACK_OPENED":
                    return
                original.rename(root / "opened-original")
                original.write_bytes((root / "opened-original").read_bytes())

            with self.assertRaisesRegex(
                RuntimeError,
                "changed during|no longer names",
            ):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                    verification_phase_callback=replace_after_open,
                )

        with TemporaryDirectory() as temporary:
            root = Path(temporary) / "cache"
            inventory, manifest, policy = self._fixture(root)
            original = root / PACKED_VECTOR_FILE_NAME

            def mutate_after_scan(phase: str) -> None:
                if phase == "PACK_SCANNED":
                    with original.open("r+b") as stream:
                        stream.seek(0)
                        stream.write(b"\x00")
                        stream.flush()
                        os.fsync(stream.fileno())

            with self.assertRaisesRegex(RuntimeError, "changed during"):
                verify_packed_embedding_cache_files(
                    root=root,
                    inventory=inventory,
                    manifest=manifest,
                    policy=policy,
                    verification_phase_callback=mutate_after_scan,
                )

    def test_manifest_logical_digest_excludes_only_physical_pack_hash(self) -> None:
        with TemporaryDirectory() as temporary:
            _, manifest, _ = self._fixture(Path(temporary) / "cache")
            changed_storage = replace(
                manifest.storage,
                content_sha256=_opaque("another-whole-pack-hash"),
            )
            changed = replace(manifest, storage=changed_storage)
            self.assertNotEqual(changed.manifest_sha256, manifest.manifest_sha256)
            self.assertEqual(
                changed.logical_cache_sha256,
                manifest.logical_cache_sha256,
            )
            self.assertNotEqual(
                content_sha256(changed.to_dict()),
                content_sha256(manifest.to_dict()),
            )


if __name__ == "__main__":
    unittest.main()
