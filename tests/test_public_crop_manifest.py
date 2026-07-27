from __future__ import annotations

import hashlib
import os
import unittest
from dataclasses import FrozenInstanceError, replace
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from PIL import Image

from cvi.public_crop_manifest import (
    DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY,
    PublicCropArtifact,
    PublicCropManifest,
    PublicCropVerification,
    PublicCropVerificationPolicy,
    canonical_rgb_pixel_sha256,
    verify_public_crop_manifest,
)


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _artifact(
    root: Path,
    *,
    sample: str,
    subject: str,
    component: str,
    color: tuple[int, int, int],
    source_variant: str = "original",
) -> PublicCropArtifact:
    path = root / f"{sample}.png"
    with Image.new("RGB", (3, 2), color) as image:
        image.save(path, format="PNG")
        pixels = image.tobytes("raw", "RGB")
    payload = path.read_bytes()
    return PublicCropArtifact(
        sample_token=sample,
        public_subject_token=subject,
        component_token=component,
        source_variant=source_variant,
        relative_path=path.name,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        pixel_sha256=canonical_rgb_pixel_sha256(3, 2, pixels),
        width=3,
        height=2,
        mode="RGB",
        format="PNG",
    )


class PublicCropManifestTests(unittest.TestCase):
    def test_verification_policy_is_immutable_and_positive(self) -> None:
        policy = PublicCropVerificationPolicy()
        with self.assertRaises(FrozenInstanceError):
            policy.maximum_artifacts = 1  # type: ignore[misc]
        with self.assertRaisesRegex(ValueError, "maximum_width"):
            replace(policy, maximum_width=0)

    def test_strict_round_trip_and_manifest_content_digest(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            manifest = PublicCropManifest((artifact,))
            self.assertEqual(PublicCropArtifact.from_dict(artifact.to_dict()), artifact)
            self.assertEqual(PublicCropManifest.from_dict(manifest.to_dict()), manifest)
            self.assertEqual(manifest.content_digest, manifest.manifest_sha256)
            self.assertEqual(len(artifact.artifact_sha256), 64)
            payload = artifact.to_dict()
            payload["registered_dog_id"] = "forbidden"
            with self.assertRaisesRegex(ValueError, "unknown"):
                PublicCropArtifact.from_dict(payload)

    def test_canonical_one_artifact_per_sample_paths(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            with self.assertRaisesRegex(ValueError, "canonical root filename"):
                replace(artifact, relative_path=f"nested/{artifact.relative_path}")
            with self.assertRaisesRegex(ValueError, "must equal"):
                replace(artifact, relative_path="other.png")
            with self.assertRaisesRegex(ValueError, "duplicate sample tokens"):
                PublicCropManifest((artifact, artifact))

    def test_exact_bytes_metadata_and_decoded_pixels_are_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            manifest = PublicCropManifest((artifact,))
            verification = verify_public_crop_manifest(root, manifest)
            self.assertEqual(
                PublicCropVerification.from_dict(verification.to_dict()), verification
            )
            self.assertEqual(verification.state, "PASS")
            self.assertTrue(verification.decoded_rgb_pixels_verified)

            wrong_pixels = replace(artifact, pixel_sha256="f" * 64)
            with self.assertRaisesRegex(ValueError, "pixel hash mismatch"):
                verify_public_crop_manifest(root, PublicCropManifest((wrong_pixels,)))
            encoded_only = verify_public_crop_manifest(
                root,
                PublicCropManifest((wrong_pixels,)),
                verify_decoded_rgb_pixels=False,
            )
            self.assertFalse(encoded_only.decoded_rgb_pixels_verified)

    def test_extra_missing_symlink_and_changed_content_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            manifest = PublicCropManifest((artifact,))
            extra = root / "extra.png"
            extra.write_bytes(b"extra")
            with self.assertRaisesRegex(ValueError, "paths mismatch"):
                verify_public_crop_manifest(root, manifest)
            extra.unlink()

            path = root / artifact.relative_path
            original = path.read_bytes()
            path.write_bytes(original + b"changed")
            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                verify_public_crop_manifest(root, manifest)
            path.write_bytes(original)

            target = root / artifact.relative_path
            elsewhere = root.parent / f"{_token('elsewhere')}.png"
            elsewhere.write_bytes(original)
            target.unlink()
            target.symlink_to(elsewhere)
            try:
                with self.assertRaisesRegex(ValueError, "must not contain symlinks"):
                    verify_public_crop_manifest(root, manifest)
            finally:
                elsewhere.unlink()

            root_link = root.parent / f"{root.name}-link"
            root_link.symlink_to(root, target_is_directory=True)
            try:
                with self.assertRaisesRegex(ValueError, "root must not be a symlink"):
                    verify_public_crop_manifest(root_link, manifest)
            finally:
                root_link.unlink()

    def test_every_verification_cap_fails_before_file_read_or_decode(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            first = _artifact(
                root,
                sample=_token("first-sample"),
                subject=_token("first-subject"),
                component=_token("first-component"),
                color=(10, 20, 30),
            )
            second = _artifact(
                root,
                sample=_token("second-sample"),
                subject=_token("second-subject"),
                component=_token("second-component"),
                color=(30, 20, 10),
            )
            pair = PublicCropManifest(
                tuple(sorted((first, second), key=lambda item: item.sample_token))
            )
            single = PublicCropManifest((first,))
            default = DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY
            cases = (
                (
                    "artifact count",
                    pair,
                    replace(default, maximum_artifacts=1),
                    "maximum_artifacts",
                ),
                (
                    "per-file encoded bytes",
                    single,
                    replace(
                        default,
                        maximum_encoded_bytes_per_file=first.byte_size - 1,
                    ),
                    "maximum_encoded_bytes_per_file",
                ),
                (
                    "aggregate encoded bytes",
                    pair,
                    replace(
                        default,
                        maximum_total_encoded_bytes=(
                            first.byte_size + second.byte_size - 1
                        ),
                    ),
                    "maximum_total_encoded_bytes",
                ),
                (
                    "width",
                    single,
                    replace(default, maximum_width=first.width - 1),
                    "maximum_width",
                ),
                (
                    "height",
                    single,
                    replace(default, maximum_height=first.height - 1),
                    "maximum_height",
                ),
                (
                    "per-image decoded pixels",
                    single,
                    replace(
                        default,
                        maximum_decoded_pixels_per_image=(
                            first.width * first.height - 1
                        ),
                    ),
                    "maximum_decoded_pixels_per_image",
                ),
                (
                    "aggregate decoded pixels",
                    pair,
                    replace(
                        default,
                        maximum_total_decoded_pixels=(
                            first.width * first.height
                            + second.width * second.height
                            - 1
                        ),
                    ),
                    "maximum_total_decoded_pixels",
                ),
            )

            for label, manifest, policy, message in cases:
                with self.subTest(cap=label):
                    with (
                        patch(
                            "cvi.public_crop_manifest._read_exact_regular_file"
                        ) as read_file,
                        patch(
                            "cvi.public_crop_manifest._verify_decoded_image"
                        ) as decode_image,
                    ):
                        with self.assertRaisesRegex(ValueError, message):
                            verify_public_crop_manifest(
                                root, manifest, policy=policy
                            )
                    read_file.assert_not_called()
                    decode_image.assert_not_called()

    def test_encoded_stat_cap_and_decoded_header_cap_precede_expensive_work(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            path = root / artifact.relative_path
            original_byte_size = artifact.byte_size
            path.write_bytes(path.read_bytes() + b"hostile-growth")
            policy = replace(
                DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY,
                maximum_encoded_bytes_per_file=original_byte_size,
            )
            with patch("cvi.public_crop_manifest.os.read") as read_file:
                with self.assertRaisesRegex(
                    ValueError, "maximum_encoded_bytes_per_file"
                ):
                    verify_public_crop_manifest(
                        root, PublicCropManifest((artifact,)), policy=policy
                    )
            read_file.assert_not_called()

            with Image.new("RGB", (4, 2), (10, 20, 30)) as image:
                image.save(path, format="PNG")
            payload = path.read_bytes()
            hostile_header = replace(
                artifact,
                byte_size=len(payload),
                content_sha256=hashlib.sha256(payload).hexdigest(),
            )
            policy = replace(
                DEFAULT_PUBLIC_CROP_VERIFICATION_POLICY,
                maximum_width=artifact.width,
            )
            with patch.object(
                Image.Image,
                "load",
                side_effect=AssertionError("full decode must not run"),
            ) as load_image:
                with self.assertRaisesRegex(ValueError, "maximum_width"):
                    verify_public_crop_manifest(
                        root,
                        PublicCropManifest((hostile_header,)),
                        policy=policy,
                    )
            load_image.assert_not_called()

    def test_non_regular_artifact_is_rejected_without_opening_it(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            artifact = _artifact(
                root,
                sample=_token("sample"),
                subject=_token("subject"),
                component=_token("component"),
                color=(10, 20, 30),
            )
            path = root / artifact.relative_path
            path.unlink()
            os.mkfifo(path)
            with patch(
                "cvi.public_crop_manifest._read_exact_regular_file"
            ) as read_file:
                with self.assertRaisesRegex(ValueError, "regular files only"):
                    verify_public_crop_manifest(
                        root, PublicCropManifest((artifact,))
                    )
            read_file.assert_not_called()


if __name__ == "__main__":
    unittest.main()
