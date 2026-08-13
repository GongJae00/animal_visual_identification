from __future__ import annotations

import binascii
import hashlib
import io
import unittest
import zipfile
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

try:
    import PIL
    from PIL import Image
except ModuleNotFoundError:
    PILLOW_AVAILABLE = False
else:
    PILLOW_AVAILABLE = True

from data.public_canine_manifest import (
    CanineRegion,
    IdentitySemantics,
    PublicCanineManifest,
    PublicCanineRecord,
)
from data.public_image_content_audit import (
    ImageContentAuditPolicy,
    audit_public_canine_image_content,
)

_RECEIPT_SHA256 = hashlib.sha256(b"synthetic archive receipt").hexdigest()


def _jpeg(color: tuple[int, int, int], *, size: tuple[int, int] = (4, 3)) -> bytes:
    output = io.BytesIO()
    Image.new("RGB", size, color).save(
        output,
        format="JPEG",
        quality=95,
        subsampling=0,
    )
    return output.getvalue()


def _jpeg_with_comment(payload: bytes) -> bytes:
    if not payload.startswith(b"\xff\xd8"):
        raise ValueError("fixture is not JPEG")
    comment = b"same decoded pixels, different encoded bytes"
    return payload[:2] + b"\xff\xfe" + (len(comment) + 2).to_bytes(2, "big") + comment + payload[2:]


def _png_rgba(alpha: int) -> bytes:
    output = io.BytesIO()
    Image.new("RGBA", (4, 3), (17, 53, 91, alpha)).save(output, format="PNG")
    return output.getvalue()


def _archive_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _record(
    *,
    archive_sha256: str,
    path: str,
    payload: bytes,
    sample: str,
    identity: str,
    variant: str = "original",
    paired: str | None = None,
    container_path: str | None = None,
    container_payload: bytes | None = None,
) -> PublicCanineRecord:
    return PublicCanineRecord(
        dataset_name="fixture",
        dataset_version="v1",
        source_variant=variant,
        source_sample_id=f"fixture:v1:{sample}",
        dataset_identity_id=f"fixture:v1:web-folder:{identity}",
        identity_semantics=IdentitySemantics.WEB_FOLDER,
        region=CanineRegion.FACE,
        original_split="train" if identity == "1" else "test",
        sequence_id=f"fixture:v1:sequence:{identity}",
        camera_token=None,
        camera_token_verified=False,
        filename_identity_token=identity,
        source_cluster_id=None,
        in_no_mono_subset=None,
        paired_source_sample_id=paired,
        member_path=path,
        member_crc32=binascii.crc32(payload),
        member_uncompressed_bytes=len(payload),
        source_archive_sha256=archive_sha256,
        source_archive_receipt_sha256=_RECEIPT_SHA256,
        container_member_path=container_path,
        container_member_crc32=(
            None if container_payload is None else binascii.crc32(container_payload)
        ),
        container_member_uncompressed_bytes=(
            None if container_payload is None else len(container_payload)
        ),
    )


def _manifest(
    archive: Path, records: tuple[PublicCanineRecord, ...]
) -> PublicCanineManifest:
    digest = _archive_sha256(archive)
    rebound = tuple(replace(item, source_archive_sha256=digest) for item in records)
    return PublicCanineManifest("fixture", "v1", digest, _RECEIPT_SHA256, rebound)


def _small_policy(**overrides: object) -> ImageContentAuditPolicy:
    values: dict[str, object] = {
        "maximum_archive_bytes": 2_000_000,
        "maximum_records": 20,
        "maximum_member_encoded_bytes": 100_000,
        "maximum_member_compressed_bytes": 100_000,
        "maximum_total_encoded_bytes": 500_000,
        "maximum_container_bytes": 500_000,
        "minimum_temporary_free_bytes_after_stage": 1,
        "maximum_container_compression_ratio": 4.0,
        "maximum_member_compression_ratio": 200.0,
        "maximum_width": 64,
        "maximum_height": 64,
        "maximum_image_pixels": 4_096,
        "maximum_total_pixels": 20_000,
        "maximum_hash_chunk_bytes": 1_024,
        "zip_read_chunk_bytes": 31,
    }
    values.update(overrides)
    return ImageContentAuditPolicy(**values)


@unittest.skipUnless(PILLOW_AVAILABLE, "optional Pillow dependency is unavailable")
class PublicImageContentAuditTests(unittest.TestCase):
    def test_byte_different_jpegs_with_identical_pixels_are_exact_duplicates(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base = _jpeg((17, 53, 91))
            commented = _jpeg_with_comment(base)
            self.assertNotEqual(hashlib.sha256(base).digest(), hashlib.sha256(commented).digest())
            archive = root / "direct.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("dogs/a.jpg", base)
                bundle.writestr("dogs/b.jpg", commented)
            records = (
                _record(
                    archive_sha256="0" * 64,
                    path="dogs/a.jpg",
                    payload=base,
                    sample="sample-z",
                    identity="1",
                ),
                _record(
                    archive_sha256="0" * 64,
                    path="dogs/b.jpg",
                    payload=commented,
                    sample="sample-a",
                    identity="999",
                ),
            )
            manifest = _manifest(archive, records)
            receipt = audit_public_canine_image_content(
                archive_path=archive,
                manifest=manifest,
                policy=_small_policy(),
            )
            self.assertEqual([item.source_sample_id for item in receipt.records], [
                "fixture:v1:sample-a",
                "fixture:v1:sample-z",
            ])
            self.assertNotEqual(receipt.records[0].encoded_sha256, receipt.records[1].encoded_sha256)
            self.assertEqual(receipt.records[0].pixel_sha256, receipt.records[1].pixel_sha256)
            self.assertEqual(len(receipt.exact_duplicate_groups), 1)
            self.assertEqual(receipt.records[0].pixel_hash_input_fields, (
                "canonical_width",
                "canonical_height",
                "canonical_mode",
                "canonical_pixels",
            ))

    def test_nested_original_and_random_background_remain_distinguishable(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            original = _jpeg((10, 20, 30))
            random = _jpeg((220, 210, 200))
            inner_original = io.BytesIO()
            with zipfile.ZipFile(inner_original, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("YT-BB-Dog/test/2000/2000_1.jpg", original)
            inner_random = io.BytesIO()
            with zipfile.ZipFile(inner_random, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr(
                    "YT-BB-Dog_random_bckg/YT-BB-Dog/test/2000/2000_1.jpg",
                    random,
                )
            original_zip, random_zip = inner_original.getvalue(), inner_random.getvalue()
            archive = root / "outer.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_STORED) as outer:
                outer.writestr("package/original.zip", original_zip)
                outer.writestr("package/random.zip", random_zip)
            paired_id = "fixture:v1:original"
            records = (
                _record(
                    archive_sha256="0" * 64,
                    path="YT-BB-Dog/test/2000/2000_1.jpg",
                    payload=original,
                    sample="original",
                    identity="1",
                    container_path="package/original.zip",
                    container_payload=original_zip,
                ),
                _record(
                    archive_sha256="0" * 64,
                    path="YT-BB-Dog_random_bckg/YT-BB-Dog/test/2000/2000_1.jpg",
                    payload=random,
                    sample="random",
                    identity="1",
                    variant="random_background",
                    paired=paired_id,
                    container_path="package/random.zip",
                    container_payload=random_zip,
                ),
            )
            receipt = audit_public_canine_image_content(
                archive_path=archive,
                manifest=_manifest(archive, records),
                policy=_small_policy(),
            )
            self.assertEqual(dict(receipt.source_variant_counts), {
                "original": 1,
                "random_background": 1,
            })
            self.assertEqual(receipt.paired_record_count, 1)
            self.assertNotEqual(receipt.records[0].pixel_sha256, receipt.records[1].pixel_sha256)
            random_record = next(item for item in receipt.records if item.source_variant == "random_background")
            self.assertEqual(random_record.paired_source_sample_id, paired_id)

    def test_archive_and_member_binding_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            payload = _jpeg((1, 2, 3))
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("dog/a.jpg", payload)
            manifest = _manifest(archive, (
                _record(
                    archive_sha256="0" * 64,
                    path="dog/a.jpg",
                    payload=payload,
                    sample="a",
                    identity="1",
                ),
            ))
            with archive.open("ab") as stream:
                stream.write(b"mutated")
            with self.assertRaisesRegex(ValueError, "archive bytes differ"):
                audit_public_canine_image_content(
                    archive_path=archive,
                    manifest=manifest,
                    policy=_small_policy(),
                )

            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("dog/a.jpg", payload)
            current = _manifest(archive, manifest.records)
            bad_record = replace(current.records[0], member_crc32=0)
            bad_manifest = PublicCanineManifest(
                current.dataset_name,
                current.dataset_version,
                current.source_archive_sha256,
                current.source_archive_receipt_sha256,
                (bad_record,),
            )
            with self.assertRaisesRegex(ValueError, "metadata differs"):
                audit_public_canine_image_content(
                    archive_path=archive,
                    manifest=bad_manifest,
                    policy=_small_policy(),
                )

    def test_invalid_decode_member_limit_and_pixel_limit_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            invalid = b"not a JPEG image"
            archive = root / "invalid.zip"
            with zipfile.ZipFile(archive, "w") as bundle:
                bundle.writestr("dog/a.jpg", invalid)
            manifest = _manifest(archive, (
                _record(
                    archive_sha256="0" * 64,
                    path="dog/a.jpg",
                    payload=invalid,
                    sample="invalid",
                    identity="1",
                ),
            ))
            with self.assertRaisesRegex(ValueError, "header decode failed"):
                audit_public_canine_image_content(
                    archive_path=archive,
                    manifest=manifest,
                    policy=_small_policy(),
                )
            with self.assertRaisesRegex(ValueError, "encoded-byte limit"):
                audit_public_canine_image_content(
                    archive_path=archive,
                    manifest=manifest,
                    policy=_small_policy(maximum_member_encoded_bytes=4),
                )

            compressible = b"x" * 20_000
            compressed_archive = root / "compressed-bomb.zip"
            with zipfile.ZipFile(
                compressed_archive, "w", zipfile.ZIP_DEFLATED
            ) as bundle:
                bundle.writestr("dog/bomb.jpg", compressible)
            compressed_manifest = _manifest(compressed_archive, (
                _record(
                    archive_sha256="0" * 64,
                    path="dog/bomb.jpg",
                    payload=compressible,
                    sample="compressed-bomb",
                    identity="1",
                ),
            ))
            with self.assertRaisesRegex(ValueError, "compression-ratio limit"):
                audit_public_canine_image_content(
                    archive_path=compressed_archive,
                    manifest=compressed_manifest,
                    policy=_small_policy(maximum_member_compression_ratio=2.0),
                )

            large = _jpeg((1, 2, 3), size=(9, 9))
            large_archive = root / "large.zip"
            with zipfile.ZipFile(large_archive, "w") as bundle:
                bundle.writestr("dog/large.jpg", large)
            large_manifest = _manifest(large_archive, (
                _record(
                    archive_sha256="0" * 64,
                    path="dog/large.jpg",
                    payload=large,
                    sample="large",
                    identity="1",
                ),
            ))
            with self.assertRaisesRegex(ValueError, "pixel limit"):
                audit_public_canine_image_content(
                    archive_path=large_archive,
                    manifest=large_manifest,
                    policy=_small_policy(maximum_image_pixels=64),
                )

    def test_misextended_opaque_png_is_audited_but_transparency_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            opaque = _png_rgba(255)
            transparent = _png_rgba(127)
            archive = root / "png-as-jpg.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("dog/opaque.jpg", opaque)
                bundle.writestr("dog/transparent.jpg", transparent)
            manifest = _manifest(
                archive,
                (
                    _record(
                        archive_sha256="0" * 64,
                        path="dog/opaque.jpg",
                        payload=opaque,
                        sample="opaque",
                        identity="1",
                    ),
                    _record(
                        archive_sha256="0" * 64,
                        path="dog/transparent.jpg",
                        payload=transparent,
                        sample="transparent",
                        identity="2",
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "non-opaque alpha"):
                audit_public_canine_image_content(
                    archive_path=archive,
                    manifest=manifest,
                    policy=_small_policy(),
                )

            opaque_only = PublicCanineManifest(
                manifest.dataset_name,
                manifest.dataset_version,
                manifest.source_archive_sha256,
                manifest.source_archive_receipt_sha256,
                (manifest.records[0],),
            )
            receipt = audit_public_canine_image_content(
                archive_path=archive,
                manifest=opaque_only,
                policy=_small_policy(),
            )
            self.assertEqual(receipt.records[0].decoded_format, "PNG")
            self.assertEqual(receipt.records[0].source_mode, "RGBA")

    def test_receipt_is_deterministic_under_manifest_record_order(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first, second = _jpeg((5, 6, 7)), _jpeg((8, 9, 10))
            archive = root / "order.zip"
            with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED) as bundle:
                bundle.writestr("dog/a.jpg", first)
                bundle.writestr("dog/b.jpg", second)
            initial = _manifest(archive, (
                _record(
                    archive_sha256="0" * 64,
                    path="dog/a.jpg",
                    payload=first,
                    sample="b",
                    identity="1",
                ),
                _record(
                    archive_sha256="0" * 64,
                    path="dog/b.jpg",
                    payload=second,
                    sample="a",
                    identity="2",
                ),
            ))
            reversed_manifest = PublicCanineManifest(
                initial.dataset_name,
                initial.dataset_version,
                initial.source_archive_sha256,
                initial.source_archive_receipt_sha256,
                tuple(reversed(initial.records)),
            )
            first_receipt = audit_public_canine_image_content(
                archive_path=archive,
                manifest=initial,
                policy=_small_policy(),
            )
            second_receipt = audit_public_canine_image_content(
                archive_path=archive,
                manifest=reversed_manifest,
                policy=_small_policy(),
            )
            self.assertEqual(first_receipt.receipt_sha256, second_receipt.receipt_sha256)
            self.assertEqual(first_receipt.to_dict(), second_receipt.to_dict())


if __name__ == "__main__":
    unittest.main()
