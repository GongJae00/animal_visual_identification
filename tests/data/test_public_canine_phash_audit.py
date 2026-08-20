from __future__ import annotations

import binascii
import hashlib
import io
import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch
from tests.repo_root import REPO_ROOT

try:
    from PIL import Image
except ModuleNotFoundError:
    PILLOW_AVAILABLE = False
else:
    PILLOW_AVAILABLE = True

from data.public_sources.public_canine_manifest import (
    CanineRegion,
    IdentitySemantics,
    PublicCanineManifest,
    PublicCanineRecord,
)
from data.public_sources.public_image_content_audit import (
    ImageContentAuditPolicy,
    audit_public_canine_image_content,
)
from shared.foundation.protected_io import read_strict_json_object
from shared.foundation.provenance import content_sha256
from data.audit.phash_mih import (
    CandidateLimitExceeded,
    PHashFingerprint,
    opaque_sample_id,
)
from data.audit.public_canine_phash_audit import (
    PublicCaninePHashPolicy,
    PublicCaninePHashSource,
    _AuthenticatedSource,
    _bound_member_info,
    _fingerprint_member,
    _read_image_bundle,
    _semantic_manifest_sha256,
    _unique_info_index,
    _verify_image_receipt,
    read_public_canine_phash_policy,
    read_public_canine_phash_sources,
    run_public_canine_phash_audit,
)

_DATASETS = ("dogfacenet224", "mpdd", "sibetan", "yt-bb-dog")
_ARCHIVE_RECEIPT = hashlib.sha256(b"archive receipt").hexdigest()

def _jpeg() -> bytes:
    output = io.BytesIO()
    image = Image.new("RGB", (41, 29))
    image.putdata([
        ((x * 7 + y * 3) % 256, (x * 11 + y) % 256, (x + y * 13) % 256)
        for y in range(29)
        for x in range(41)
    ])
    image.save(output, format="JPEG", quality=97, subsampling=0)
    return output.getvalue()

def _record(dataset: str, *, archive_sha256: str, payload: bytes) -> PublicCanineRecord:
    return PublicCanineRecord(
        dataset_name=dataset,
        dataset_version="v1",
        source_variant="original",
        source_sample_id=f"{dataset}:v1:sample:1",
        dataset_identity_id=f"{dataset}:v1:identity:1",
        identity_semantics=IdentitySemantics.WEB_FOLDER,
        region=CanineRegion.DOG_CROP,
        original_split="train",
        sequence_id=None,
        camera_token=None,
        camera_token_verified=False,
        filename_identity_token=None,
        source_cluster_id=None,
        in_no_mono_subset=None,
        paired_source_sample_id=None,
        member_path="images/1.jpg",
        member_crc32=binascii.crc32(payload),
        member_uncompressed_bytes=len(payload),
        source_archive_sha256=archive_sha256,
        source_archive_receipt_sha256=_ARCHIVE_RECEIPT,
    )

def _source(dataset: str, root: Path) -> PublicCaninePHashSource:
    class_path = root / "class.txt" if dataset == "dogfacenet224" else None
    return PublicCaninePHashSource(
        dataset_name=dataset,
        archive_path=root / f"{dataset}.zip",
        archive_receipt_path=root / f"{dataset}-archive.json",
        semantic_receipt_path=root / f"{dataset}-semantic.json",
        image_content_receipt_path=root / f"{dataset}-image.json",
        dogface_classes_train_path=class_path,
        dogface_classes_test_path=class_path,
    )

def _provenance() -> dict[str, object]:
    files: list[object] = []
    runtime = {"python": "fixture"}
    return {
        "schema_version": "cvi.offline_tool_provenance.v1",
        "code_source_manifest_sha256": content_sha256(files),
        "code_source_files": files,
        "runtime": runtime,
        "runtime_sha256": content_sha256(runtime),
    }

class PublicCaninePHashConfigTests(unittest.TestCase):
    def test_cli_help_and_example_policy_round_trip(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                str(REPO_ROOT / "data" / "audit" / "phash_candidate_audit.py"),
                "--help",
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        self.assertIn("--binding-output", completed.stdout)
        policy = read_public_canine_phash_policy(
            REPO_ROOT
            / "archive"
            / "shared_helpers"
            / "configs"
            / "contracts"
            / "public_canine_phash_policy.example.json"
        )
        self.assertEqual(policy.radius, 10)
        self.assertEqual(policy.policy_sha256, policy.policy_sha256)

    def test_source_spec_requires_each_dataset_once_and_exact_fields(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            rows = []
            for dataset in _DATASETS:
                class_value = str(root / "class.txt") if dataset == "dogfacenet224" else None
                rows.append({
                    "schema_version": "cvi.public_canine_phash_source.v1",
                    "dataset_name": dataset,
                    "archive_path": str(root / f"{dataset}.zip"),
                    "archive_receipt_path": str(root / f"{dataset}-archive.json"),
                    "semantic_receipt_path": str(root / f"{dataset}-semantic.json"),
                    "image_content_receipt_path": str(root / f"{dataset}-image.json"),
                    "dogface_classes_train_path": class_value,
                    "dogface_classes_test_path": class_value,
                })
            path = root / "sources.json"
            path.write_text(json.dumps({
                "schema_version": "cvi.public_canine_phash_source_spec.v1",
                "sources": rows,
            }), encoding="utf-8")
            sources = read_public_canine_phash_sources(path)
            self.assertEqual({item.dataset_name for item in sources}, set(_DATASETS))

            rows[-1]["dataset_name"] = "mpdd"
            duplicate = root / "duplicate.json"
            duplicate.write_text(json.dumps({
                "schema_version": "cvi.public_canine_phash_source_spec.v1",
                "sources": rows,
            }), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "each audited dataset once"):
                read_public_canine_phash_sources(duplicate)

@unittest.skipUnless(PILLOW_AVAILABLE, "optional Pillow dependency is unavailable")
class PublicCaninePHashDecodeTests(unittest.TestCase):
    def test_authenticated_pixels_are_exif_rgb_lanczos_fingerprinted(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _jpeg()
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("images/1.jpg", payload)
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            record = _record("fixture", archive_sha256=archive_sha, payload=payload)
            manifest = PublicCanineManifest(
                "fixture", "v1", archive_sha, _ARCHIVE_RECEIPT, (record,)
            )
            image_policy = ImageContentAuditPolicy(
                maximum_archive_bytes=1_000_000,
                maximum_records=10,
                maximum_member_encoded_bytes=100_000,
                maximum_member_compressed_bytes=100_000,
                maximum_total_encoded_bytes=100_000,
                maximum_container_bytes=100_000,
                minimum_temporary_free_bytes_after_stage=1,
                maximum_width=100,
                maximum_height=100,
                maximum_image_pixels=10_000,
                maximum_total_pixels=10_000,
                maximum_hash_chunk_bytes=1_024,
                zip_read_chunk_bytes=31,
            )
            protected = audit_public_canine_image_content(
                archive_path=archive, manifest=manifest, policy=image_policy
            ).records[0].to_dict()
            phash_policy = PublicCaninePHashPolicy(
                maximum_fingerprints=10,
                maximum_pair_inspections=10,
                maximum_archive_bytes=1_000_000,
                maximum_member_encoded_bytes=100_000,
                maximum_member_compressed_bytes=100_000,
                maximum_container_bytes=100_000,
                minimum_temporary_free_bytes_after_stage=1,
                maximum_image_pixels=10_000,
                maximum_total_pixels=10_000,
                read_chunk_bytes=31,
            )
            from PIL import ImageFile, ImageOps, UnidentifiedImageError

            with zipfile.ZipFile(archive) as bundle:
                info = _bound_member_info(_unique_info_index(bundle), record, phash_policy)
                fingerprint, pixels = _fingerprint_member(
                    bundle,
                    info,
                    record,
                    protected,
                    phash_policy,
                    Image,
                    ImageOps,
                    UnidentifiedImageError,
                )
            self.assertEqual(fingerprint.opaque_sample_id, opaque_sample_id(record.source_sample_id))
            self.assertEqual(pixels, 41 * 29)
            self.assertFalse(ImageFile.LOAD_TRUNCATED_IMAGES)

    def test_protected_image_bundle_and_semantic_source_binding_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _jpeg()
            archive = root / "images.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("images/1.jpg", payload)
            archive_sha = hashlib.sha256(archive.read_bytes()).hexdigest()
            record = _record("fixture", archive_sha256=archive_sha, payload=payload)
            manifest = PublicCanineManifest(
                "fixture", "v1", archive_sha, _ARCHIVE_RECEIPT, (record,)
            )
            policy = ImageContentAuditPolicy(
                maximum_archive_bytes=1_000_000,
                maximum_records=10,
                maximum_member_encoded_bytes=100_000,
                maximum_member_compressed_bytes=100_000,
                maximum_total_encoded_bytes=100_000,
                maximum_container_bytes=100_000,
                minimum_temporary_free_bytes_after_stage=1,
                maximum_width=100,
                maximum_height=100,
                maximum_image_pixels=10_000,
                maximum_total_pixels=10_000,
                maximum_hash_chunk_bytes=1_024,
                zip_read_chunk_bytes=31,
            )
            receipt = audit_public_canine_image_content(
                archive_path=archive, manifest=manifest, policy=policy
            )
            provenance = _provenance()
            image_bundle = {
                "schema_version": "cvi.image_content_audit_bundle.v1",
                "semantic_receipt_sha256": "4" * 64,
                "policy": policy.to_dict(),
                "policy_sha256": policy.policy_sha256,
                "receipt": receipt.to_dict(),
                "receipt_sha256": receipt.receipt_sha256,
                "tool_provenance": provenance,
                "tool_provenance_sha256": content_sha256(provenance),
            }
            path = root / "image-content.json"
            path.write_text(json.dumps(image_bundle), encoding="utf-8")
            loaded = _read_image_bundle(path)
            _verify_image_receipt(loaded["receipt"], manifest, loaded)
            self.assertEqual(
                loaded["receipt"]["semantic_manifest_sha256"],
                _semantic_manifest_sha256(manifest),
            )

            forged_receipt = json.loads(json.dumps(loaded["receipt"]))
            forged_receipt["records"][0]["member_path"] = "images/other.jpg"
            with self.assertRaisesRegex(ValueError, "source binding differs"):
                _verify_image_receipt(forged_receipt, manifest, loaded)

            forged_bundle = json.loads(json.dumps(image_bundle))
            forged_bundle["receipt"]["total_encoded_bytes"] += 1
            forged_path = root / "forged.json"
            forged_path.write_text(json.dumps(forged_bundle), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "receipt digest differs"):
                _read_image_bundle(forged_path)

class PublicCaninePHashPublicationTests(unittest.TestCase):
    def _authenticated(
        self, root: Path, *, shared_pixel_digest: bool = False
    ) -> tuple[_AuthenticatedSource, ...]:
        payload = b"x"
        output = []
        for index, dataset in enumerate(_DATASETS):
            record = _record(dataset, archive_sha256="1" * 64, payload=payload)
            manifest = PublicCanineManifest(
                dataset, "v1", "1" * 64, _ARCHIVE_RECEIPT, (record,)
            )
            output.append(_AuthenticatedSource(
                source=_source(dataset, root),
                manifests=(manifest,),
                archive_receipt_sha256=_ARCHIVE_RECEIPT,
                semantic_receipt_sha256=hashlib.sha256(f"s:{dataset}".encode()).hexdigest(),
                image_receipt_sha256=hashlib.sha256(f"i:{dataset}".encode()).hexdigest(),
                image_policy_sha256="2" * 64,
                image_decoder_name="Pillow",
                image_decoder_version="test",
                image_records={record.source_sample_id: {
                    "pixel_sha256": (
                        "3" * 64
                        if shared_pixel_digest
                        else hashlib.sha256(f"pixel:{index}".encode()).hexdigest()
                    )
                }},
            ))
        return tuple(output)

    @unittest.skipUnless(PILLOW_AVAILABLE, "optional Pillow dependency is unavailable")
    def test_two_outputs_are_separated_no_overwrite_and_content_bound(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticated = self._authenticated(root)
            sources = tuple(item.source for item in authenticated)
            fingerprints = tuple(
                PHashFingerprint(
                    opaque_sample_id=opaque_sample_id(
                        f"{dataset}:v1:sample:1"
                    ),
                    original_hash=index << 16,
                    horizontal_flip_hash=index << 16,
                )
                for index, dataset in enumerate(_DATASETS)
            )
            side_effect = [
                ([fingerprint], [{
                    "opaque_sample_id": fingerprint.opaque_sample_id,
                    "dataset_name": dataset,
                    "source_sample_id": f"{dataset}:v1:sample:1",
                }], 1)
                for dataset, fingerprint in zip(_DATASETS, fingerprints, strict=True)
            ]
            evidence_path = root / "evidence.json"
            binding_path = root / "binding.json"
            policy = PublicCaninePHashPolicy(
                maximum_fingerprints=10,
                maximum_pair_inspections=100,
                minimum_temporary_free_bytes_after_stage=1,
            )
            with patch(
                "data.audit.public_canine_phash_audit._authenticate_source",
                side_effect=authenticated,
            ), patch(
                "data.audit.public_canine_phash_audit._fingerprint_source",
                side_effect=side_effect,
            ):
                evidence_sha, binding_sha = run_public_canine_phash_audit(
                    sources=sources,
                    policy=policy,
                    evidence_output=evidence_path,
                    binding_output=binding_path,
                    tool_provenance=_provenance(),
                )
            evidence = read_strict_json_object(evidence_path)
            binding = read_strict_json_object(binding_path)
            self.assertEqual(evidence["evidence_sha256"], evidence_sha)
            self.assertEqual(binding["binding_sha256"], binding_sha)
            evidence_text = evidence_path.read_text(encoding="utf-8")
            self.assertNotIn("dataset_name", evidence_text)
            self.assertNotIn(":v1:sample:", evidence_text)
            self.assertIn("dataset_name", binding_path.read_text(encoding="utf-8"))
            self.assertEqual(
                binding["binding"]["interpretation"],
                "SENSITIVE_OPAQUE_TO_SOURCE_PROVENANCE_JOIN_ONLY_"
                "MUST_NOT_ENTER_CANDIDATE_GENERATION_OR_SCORING",
            )
            with self.assertRaises(FileExistsError):
                with patch(
                    "data.audit.public_canine_phash_audit._authenticate_source",
                    side_effect=authenticated,
                ), patch(
                    "data.audit.public_canine_phash_audit._fingerprint_source",
                    side_effect=side_effect,
                ):
                    run_public_canine_phash_audit(
                        sources=sources,
                        policy=policy,
                        evidence_output=evidence_path,
                        binding_output=binding_path,
                        tool_provenance=_provenance(),
                    )

    @unittest.skipUnless(PILLOW_AVAILABLE, "optional Pillow dependency is unavailable")
    def test_candidate_cap_fails_before_publication(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            authenticated = self._authenticated(root, shared_pixel_digest=True)
            sources = tuple(item.source for item in authenticated)
            fingerprints = tuple(
                PHashFingerprint(
                    opaque_sample_id=opaque_sample_id(
                        f"{dataset}:v1:sample:1"
                    ),
                    original_hash=0,
                    horizontal_flip_hash=0,
                )
                for dataset in _DATASETS
            )
            side_effect = [
                ([fingerprint], [{
                    "opaque_sample_id": fingerprint.opaque_sample_id,
                    "dataset_name": dataset,
                    "source_sample_id": f"{dataset}:v1:sample:1",
                }], 1)
                for dataset, fingerprint in zip(_DATASETS, fingerprints, strict=True)
            ]
            evidence_path, binding_path = root / "evidence.json", root / "binding.json"
            with patch(
                "data.audit.public_canine_phash_audit._authenticate_source",
                side_effect=authenticated,
            ), patch(
                "data.audit.public_canine_phash_audit._fingerprint_source",
                side_effect=side_effect,
            ):
                with self.assertRaises(CandidateLimitExceeded):
                    run_public_canine_phash_audit(
                        sources=sources,
                        policy=PublicCaninePHashPolicy(
                            maximum_fingerprints=10,
                            maximum_pair_inspections=2,
                            maximum_expanded_candidates=2,
                            minimum_temporary_free_bytes_after_stage=1,
                        ),
                        evidence_output=evidence_path,
                        binding_output=binding_path,
                        tool_provenance=_provenance(),
                    )
            self.assertFalse(evidence_path.exists())
            self.assertFalse(binding_path.exists())

if __name__ == "__main__":
    unittest.main()
