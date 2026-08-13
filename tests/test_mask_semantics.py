from __future__ import annotations

import shutil
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from data.acquisition import sha256_file
from evaluation.controls.policy import (
    ControlMaskEntry,
    ControlMaskManifest,
    MaskEvidence,
    MaskReviewStatus,
    MaskRole,
    verify_control_mask_files,
)
from evaluation.controls.mask_semantics import (
    MaskSemanticPolicy,
    _scan_binary_raw,
    build_mask_raw_decode_command,
    verify_mask_pixel_semantics,
)
from evaluation.controls.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    verify_pair_artifact_files,
)

HASH_A = "a" * 64
HASH_B = "b" * 64


def _render(path: Path, source: str) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-y",
            "-f",
            "lavfi",
            "-i",
            source,
            "-frames:v",
            "1",
            str(path),
        ),
        check=True,
        capture_output=True,
        text=True,
    )


def _mask_evidence(
    path: Path,
    role: MaskRole,
) -> MaskEvidence:
    return MaskEvidence(
        role,
        path.stem,
        path.name,
        sha256_file(path),
        path.stat().st_size,
        16,
        16,
        "v1",
        "manual-reviewed",
        HASH_B,
        MaskReviewStatus.VERIFIED,
    )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg tools are required",
)
class MaskSemanticIntegrationTests(unittest.TestCase):
    def test_binary_masks_and_accessory_containment_are_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_root = root / "base"
            mask_root = root / "masks"
            base_root.mkdir()
            mask_root.mkdir()
            base_path = base_root / "base.png"
            dog_path = mask_root / "dog.png"
            accessory_path = mask_root / "accessory.png"
            _render(base_path, "testsrc2=size=16x16:rate=1")
            _render(
                dog_path,
                (
                    "color=black:s=16x16,"
                    "drawbox=x=2:y=2:w=12:h=12:color=white:t=fill,"
                    "format=gray"
                ),
            )
            _render(
                accessory_path,
                (
                    "color=black:s=16x16,"
                    "drawbox=x=4:y=4:w=2:h=2:color=white:t=fill,"
                    "format=gray"
                ),
            )
            base_manifest = PairArtifactManifest(
                pair_set_sha256=HASH_A,
                artifact_bindings_sha256=HASH_B,
                entries=(
                    PairArtifactEntry(
                        "base",
                        base_path.name,
                        sha256_file(base_path),
                        base_path.stat().st_size,
                        "image/png",
                    ),
                ),
            )
            masks = ControlMaskManifest(
                base_manifest.manifest_sha256,
                (
                    ControlMaskEntry(
                        "base",
                        (
                            _mask_evidence(dog_path, MaskRole.DOG),
                            _mask_evidence(
                                accessory_path,
                                MaskRole.ACCESSORY,
                            ),
                        ),
                    ),
                ),
            )
            base_verification = verify_pair_artifact_files(
                base_root,
                base_manifest,
            )
            mask_verification = verify_control_mask_files(
                mask_root,
                masks,
            )
            receipt = verify_mask_pixel_semantics(
                base_root=base_root,
                base_manifest=base_manifest,
                base_verification=base_verification,
                mask_root=mask_root,
                mask_manifest=masks,
                mask_file_verification=mask_verification,
                policy=MaskSemanticPolicy(raw_scan_chunk_bytes=17),
            )
            self.assertEqual(len(receipt.entries), 1)
            stats = {item.role: item for item in receipt.entries[0].masks}
            self.assertEqual(stats[MaskRole.DOG].foreground_pixels, 144)
            self.assertEqual(
                stats[MaskRole.ACCESSORY].foreground_pixels,
                4,
            )
            self.assertEqual(
                receipt.entries[0].accessory_outside_dog_pixels,
                0,
            )
            self.assertEqual(
                type(receipt).from_dict(receipt.to_dict()),
                receipt,
            )
            command = build_mask_raw_decode_command(
                dog_path,
                root / "dog.raw",
            )
            self.assertIn("-frames:v", command)
            self.assertIn("rawvideo", command)
            self.assertIn("gray", command)

    def test_accessory_outside_dog_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            base_root = root / "base"
            mask_root = root / "masks"
            base_root.mkdir()
            mask_root.mkdir()
            base_path = base_root / "base.png"
            dog_path = mask_root / "dog.png"
            accessory_path = mask_root / "accessory.png"
            _render(base_path, "testsrc2=size=16x16:rate=1")
            _render(
                dog_path,
                (
                    "color=black:s=16x16,"
                    "drawbox=x=4:y=4:w=8:h=8:color=white:t=fill,"
                    "format=gray"
                ),
            )
            _render(
                accessory_path,
                (
                    "color=black:s=16x16,"
                    "drawbox=x=0:y=0:w=2:h=2:color=white:t=fill,"
                    "format=gray"
                ),
            )
            base_manifest = PairArtifactManifest(
                HASH_A,
                HASH_B,
                (
                    PairArtifactEntry(
                        "base",
                        base_path.name,
                        sha256_file(base_path),
                        base_path.stat().st_size,
                        "image/png",
                    ),
                ),
            )
            masks = ControlMaskManifest(
                base_manifest.manifest_sha256,
                (
                    ControlMaskEntry(
                        "base",
                        (
                            _mask_evidence(dog_path, MaskRole.DOG),
                            _mask_evidence(
                                accessory_path,
                                MaskRole.ACCESSORY,
                            ),
                        ),
                    ),
                ),
            )
            with self.assertRaisesRegex(ValueError, "outside-dog"):
                verify_mask_pixel_semantics(
                    base_root=base_root,
                    base_manifest=base_manifest,
                    base_verification=verify_pair_artifact_files(
                        base_root,
                        base_manifest,
                    ),
                    mask_root=mask_root,
                    mask_manifest=masks,
                    mask_file_verification=verify_control_mask_files(
                        mask_root,
                        masks,
                    ),
                    policy=MaskSemanticPolicy(),
                )


class MaskSemanticUnitTests(unittest.TestCase):
    def test_nonbinary_raw_mask_is_rejected_in_bounded_chunks(self) -> None:
        with TemporaryDirectory() as temporary:
            path = Path(temporary) / "mask.raw"
            path.write_bytes(bytes((0, 255, 128, 0)))
            with self.assertRaisesRegex(ValueError, "non-binary"):
                _scan_binary_raw(
                    path,
                    expected_bytes=4,
                    chunk_bytes=2,
                )

    def test_policy_is_strict_and_content_addressed(self) -> None:
        policy = MaskSemanticPolicy()
        self.assertEqual(
            MaskSemanticPolicy.from_dict(policy.to_dict()),
            policy,
        )
        self.assertEqual(len(policy.policy_sha256), 64)
        invalid = policy.to_dict()
        invalid["allow_soft_masks"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            MaskSemanticPolicy.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
