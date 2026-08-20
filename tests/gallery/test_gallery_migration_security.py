from __future__ import annotations

import hashlib
import json
import tempfile
import unittest
from contextlib import ExitStack
from pathlib import Path
from unittest.mock import patch

import faiss
import numpy as np

import gallery.migration.v3_to_v4 as migration
from enrollment.registry.identity_registry import compute_registered_dog_id

def _write_v3(source: Path) -> dict:
    source.mkdir()
    index = faiss.IndexFlatIP(2)
    index.add(np.asarray([[0.6, 0.8]], dtype=np.float32))
    paths = {
        "index": source / "master-generation.idx",
        "metadata": source / "metadata-generation.json",
        "breeds": source / "breeds-generation.json",
    }
    faiss.write_index(index, str(paths["index"]))
    content_hash = "a" * 64
    template_id = hashlib.sha256(
        f"cvi.gallery_template.v1\0{content_hash}".encode("ascii")
    ).hexdigest()
    paths["metadata"].write_text(
        json.dumps(
            {
                "0": {
                    "registered_dog_id": compute_registered_dog_id(
                        "fixture:v1:migration-security:dog"
                    ),
                    "template_id": template_id,
                    "content_sha256": content_hash,
                    "idempotency_key": "request",
                    "template_schema": "cvi.gallery_template.v1",
                    "metadata": {},
                }
            }
        ),
        encoding="utf-8",
    )
    paths["breeds"].write_text('{"0":"unknown"}', encoding="utf-8")
    manifest = {
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
        "files": {
            kind: {
                "name": path.name,
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            }
            for kind, path in paths.items()
        },
    }
    _write_manifest(source, manifest)
    return manifest

def _write_manifest(source: Path, manifest: dict) -> None:
    (source / "gallery_manifest.json").write_text(
        json.dumps(manifest, separators=(",", ":")), encoding="utf-8"
    )

def _replace_sidecar(source: Path, manifest: dict, kind: str, payload: bytes) -> None:
    path = source / manifest["files"][kind]["name"]
    path.write_bytes(payload)
    manifest["files"][kind]["sha256"] = hashlib.sha256(payload).hexdigest()
    _write_manifest(source, manifest)

def _snapshot(source: Path) -> dict[str, bytes]:
    return {path.name: path.read_bytes() for path in source.iterdir()}

class GalleryMigrationSecurityTests(unittest.TestCase):
    def test_rejects_symlink_source_output_parent_and_source_file(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            manifest = _write_v3(source)

            source_link = root / "source-link"
            source_link.symlink_to(source, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "source gallery root"):
                migration.migrate_gallery(source_link, root / "from-source-link")

            destination = root / "destination"
            destination.mkdir()
            destination_link = root / "destination-link"
            destination_link.symlink_to(destination, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "destination parent root"):
                migration.migrate_gallery(source, destination_link / "v4")
            self.assertFalse((destination / "v4").exists())

            metadata = source / manifest["files"]["metadata"]["name"]
            real_metadata = source / "real-metadata.json"
            metadata.rename(real_metadata)
            metadata.symlink_to(real_metadata.name)
            with self.assertRaisesRegex(ValueError, "non-symlink regular file"):
                migration.migrate_gallery(source, root / "from-file-link")
            self.assertFalse((root / "from-file-link").exists())

    def test_rejects_non_directory_roots_and_output_symlink(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source_file = root / "source-file"
            source_file.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "source gallery root"):
                migration.migrate_gallery(source_file, root / "v4")

            source = root / "source"
            _write_v3(source)
            parent_file = root / "parent-file"
            parent_file.write_text("not a directory", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "destination parent root"):
                migration.migrate_gallery(source, parent_file / "v4")

            symlink_target = root / "existing"
            symlink_target.mkdir()
            output = root / "output"
            output.symlink_to(symlink_target, target_is_directory=True)
            with self.assertRaisesRegex(ValueError, "new, non-existing"):
                migration.migrate_gallery(source, output)
            self.assertTrue(output.is_symlink())

    def test_rejects_destination_parent_writable_by_other_principals(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _write_v3(source)
            destination = root / "destination"
            destination.mkdir(mode=0o777)
            destination.chmod(0o777)
            with self.assertRaisesRegex(ValueError, "other principals"):
                migration.migrate_gallery(source, destination / "v4")
            self.assertFalse((destination / "v4").exists())

    def test_rejects_output_inside_source_to_preserve_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _write_v3(source)
            before = _snapshot(source)
            with self.assertRaisesRegex(ValueError, "inside the source"):
                migration.migrate_gallery(source, source / "v4")
            self.assertEqual(_snapshot(source), before)

    def test_rejects_duplicate_nonfinite_and_extra_json_fields(self) -> None:
        cases = ("duplicate", "nonfinite", "extra")
        for case in cases:
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                manifest = _write_v3(source)
                if case == "duplicate":
                    path = source / "gallery_manifest.json"
                    payload = path.read_text(encoding="utf-8")
                    path.write_text(payload[:-1] + ',"count":1}', encoding="utf-8")
                    message = "duplicate JSON object key"
                elif case == "nonfinite":
                    metadata = (
                        source / manifest["files"]["metadata"]["name"]
                    ).read_text(encoding="utf-8")
                    payload = metadata.replace('"metadata": {}', '"metadata":{"x":NaN}')
                    _replace_sidecar(
                        source, manifest, "metadata", payload.encode("utf-8")
                    )
                    message = "non-finite JSON number"
                else:
                    manifest["unexpected"] = True
                    _write_manifest(source, manifest)
                    message = "exact cvi.gallery_manifest.v3"
                output = root / "v4"
                with self.assertRaisesRegex(ValueError, message):
                    migration.migrate_gallery(source, output)
                self.assertFalse(output.exists())

    def test_rejects_invalid_metadata_schema_and_hash_mismatch(self) -> None:
        for case in ("extra-row-key", "hash-mismatch"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                manifest = _write_v3(source)
                metadata_path = source / manifest["files"]["metadata"]["name"]
                if case == "extra-row-key":
                    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                    metadata["0"]["unexpected"] = True
                    _replace_sidecar(
                        source,
                        manifest,
                        "metadata",
                        json.dumps(metadata).encode("utf-8"),
                    )
                    message = "invalid schema"
                else:
                    metadata_path.write_bytes(metadata_path.read_bytes() + b" ")
                    message = "missing or corrupted"
                output = root / "v4"
                with self.assertRaisesRegex(ValueError, message):
                    migration.migrate_gallery(source, output)
                self.assertFalse(output.exists())

    def test_rejects_manifest_sidecar_and_index_file_size_limits(self) -> None:
        for case in ("manifest", "sidecar", "index"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                manifest = _write_v3(source)
                patches = []
                if case == "manifest":
                    path = source / "gallery_manifest.json"
                    with path.open("r+b") as stream:
                        stream.truncate(migration._MAXIMUM_MANIFEST_BYTES + 1)
                elif case == "sidecar":
                    path = source / manifest["files"]["metadata"]["name"]
                    with path.open("r+b") as stream:
                        stream.truncate(migration._MAXIMUM_SIDECAR_JSON_BYTES + 1)
                else:
                    patches.append(
                        patch.object(
                            migration, "_MAXIMUM_INDEX_OVERHEAD_BYTES", 0
                        )
                    )
                output = root / "v4"
                with ExitStack() as stack:
                    for patcher in patches:
                        stack.enter_context(patcher)
                    with self.assertRaisesRegex(ValueError, "exceeds its byte limit"):
                        migration.migrate_gallery(source, output)
                self.assertFalse(output.exists())

    def test_rejects_cardinality_dimension_and_aggregate_before_faiss(self) -> None:
        for case in ("templates", "dimension", "aggregate"):
            with self.subTest(case=case), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                source = root / "source"
                manifest = _write_v3(source)
                if case == "templates":
                    value = migration._MAXIMUM_GALLERY_TEMPLATES + 1
                    manifest["count"] = value
                    manifest["template_count"] = value
                elif case == "dimension":
                    value = migration._MAXIMUM_DIMENSION + 1
                    manifest["dimension"] = value
                    manifest["embedding_contract"]["dimension"] = value
                else:
                    manifest["count"] = migration._MAXIMUM_GALLERY_TEMPLATES
                    manifest["template_count"] = migration._MAXIMUM_GALLERY_TEMPLATES
                    manifest["identity_count"] = 0
                    manifest["dimension"] = 2_000
                    manifest["embedding_contract"]["dimension"] = 2_000
                _write_manifest(source, manifest)
                output = root / "v4"
                with (
                    patch.object(
                        migration.faiss,
                        "read_index",
                        side_effect=AssertionError(
                            "FAISS must not receive invalid bounds"
                        ),
                    ),
                    self.assertRaises(ValueError),
                ):
                    migration.migrate_gallery(source, output)
                self.assertFalse(output.exists())

    def test_preexisting_output_is_untouched(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _write_v3(source)
            output = root / "v4"
            output.mkdir()
            marker = output / "marker"
            marker.write_text("keep", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "new, non-existing"):
                migration.migrate_gallery(source, output)
            self.assertEqual(marker.read_text(encoding="utf-8"), "keep")

    def test_partial_failure_removes_staging_and_preserves_source(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _write_v3(source)
            before = _snapshot(source)
            output = root / "v4"
            with (
                patch.object(
                    migration, "_build_gallery", side_effect=RuntimeError("injected")
                ),
                self.assertRaisesRegex(RuntimeError, "injected"),
            ):
                migration.migrate_gallery(source, output)
            self.assertFalse(output.exists())
            self.assertEqual(_snapshot(source), before)
            self.assertEqual(
                [path.name for path in root.iterdir() if ".migrate-" in path.name],
                [],
            )

    def test_publish_is_atomic_no_replace_under_race(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "source"
            _write_v3(source)
            output = root / "v4"
            original_build = migration._build_gallery

            def build_then_race(*args, **kwargs):
                original_build(*args, **kwargs)
                output.mkdir()
                (output / "marker").write_text("racer", encoding="utf-8")

            with (
                patch.object(migration, "_build_gallery", build_then_race),
                self.assertRaisesRegex(ValueError, "new, non-existing"),
            ):
                migration.migrate_gallery(source, output)
            self.assertEqual((output / "marker").read_text(encoding="utf-8"), "racer")
            self.assertEqual(
                [path.name for path in root.iterdir() if ".migrate-" in path.name],
                [],
            )

if __name__ == "__main__":
    unittest.main()
