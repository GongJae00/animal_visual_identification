from __future__ import annotations

import shutil
import stat
import subprocess
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from cvi.acquisition import sha256_file
from cvi.contracts import Modality
from cvi.crop_export import (
    CropBox,
    CropExportPolicy,
    OracleCropSource,
    build_crop_command,
    export_oracle_crops,
    probe_still_image,
)
from cvi.pairing import (
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairScoringRequest,
    PairStratum,
)

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def construction() -> PairConstructionResult:
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=HASH_B,
        attributes_sha256=HASH_C,
        eligible_query_count=1,
        selected_query_count=1,
        dropped_query_count=0,
        scoring_requests=(
            PairScoringRequest("pair-1", "token-rgb", "token-ir"),
        ),
        artifact_bindings=(
            PairArtifactBinding("token-rgb", "sample-rgb"),
            PairArtifactBinding("token-ir", "sample-ir"),
        ),
        ground_truth=(
            PairGroundTruth(
                "pair-1",
                "dog-1",
                "dog-2",
                "query-session",
                "reference-session",
                PairStratum.RANDOM,
            ),
        ),
        quotas=(),
    )


@unittest.skipUnless(
    shutil.which("ffmpeg") and shutil.which("ffprobe"),
    "FFmpeg tools are required",
)
class CropExportTests(unittest.TestCase):
    def test_rgb_and_ir_exports_are_cropped_and_sanitized(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source_rgb = root / "source-rgb.png"
            source_ir = root / "source-ir.png"
            output = root / "output"
            output.mkdir()
            for path in (source_rgb, source_ir):
                subprocess.run(
                    (
                        "ffmpeg",
                        "-nostdin",
                        "-hide_banner",
                        "-loglevel",
                        "error",
                        "-f",
                        "lavfi",
                        "-i",
                        "testsrc2=size=320x240:rate=1",
                        "-frames:v",
                        "1",
                        str(path),
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                )
            receipt = export_oracle_crops(
                construction(),
                sources=(
                    OracleCropSource(
                        "sample-rgb",
                        str(source_rgb),
                        sha256_file(source_rgb),
                        Modality.RGB,
                        CropBox(12, 14, 100, 80),
                    ),
                    OracleCropSource(
                        "sample-ir",
                        str(source_ir),
                        sha256_file(source_ir),
                        Modality.IR,
                        CropBox(20, 30, 96, 72),
                    ),
                ),
                policy=CropExportPolicy(),
                output_directory=output,
            )
            self.assertEqual(
                type(receipt).from_dict(receipt.to_dict()),
                receipt,
            )
            self.assertEqual(receipt.verification.verified_files, 2)
            rgb_path = output / "token-rgb.png"
            ir_path = output / "token-ir.png"
            rgb = probe_still_image(rgb_path)
            ir = probe_still_image(ir_path)
            self.assertEqual((rgb.width, rgb.height, rgb.pixel_format), (
                100, 80, "rgb24"
            ))
            self.assertEqual((ir.width, ir.height, ir.pixel_format), (
                96, 72, "gray"
            ))
            self.assertFalse(rgb.stream_tags or rgb.format_tags)
            self.assertFalse(ir.stream_tags or ir.format_tags)
            self.assertEqual(stat.S_IMODE(rgb_path.stat().st_mode), 0o600)
            self.assertEqual(stat.S_IMODE(ir_path.stat().st_mode), 0o600)

    def test_resource_limits_and_symlinks_fail_closed(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            source_link = root / "source-link.png"
            output = root / "output"
            output.mkdir()
            subprocess.run(
                (
                    "ffmpeg",
                    "-nostdin",
                    "-hide_banner",
                    "-loglevel",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc2=size=32x24:rate=1",
                    "-frames:v",
                    "1",
                    str(source),
                ),
                check=True,
                capture_output=True,
                text=True,
            )
            source_link.symlink_to(source)
            common = (
                OracleCropSource(
                    "sample-rgb",
                    str(source_link),
                    sha256_file(source),
                    Modality.RGB,
                    CropBox(0, 0, 16, 16),
                ),
                OracleCropSource(
                    "sample-ir",
                    str(source),
                    sha256_file(source),
                    Modality.IR,
                    CropBox(0, 0, 16, 16),
                ),
            )
            with self.assertRaisesRegex(ValueError, "must not be a symlink"):
                export_oracle_crops(
                    construction(),
                    sources=common,
                    policy=CropExportPolicy(),
                    output_directory=output,
                )
            with self.assertRaisesRegex(ValueError, "maximum_artifacts"):
                export_oracle_crops(
                    construction(),
                    sources=(
                        OracleCropSource(
                            "sample-rgb",
                            str(source),
                            sha256_file(source),
                            Modality.RGB,
                            CropBox(0, 0, 16, 16),
                        ),
                        common[1],
                    ),
                    policy=CropExportPolicy(maximum_artifacts=1),
                    output_directory=output,
                )

    def test_crop_bounds_and_no_overwrite_are_enforced(self) -> None:
        command = build_crop_command(
            Path("/protected/source.png"),
            Path("/protected/token.png"),
            crop=CropBox(1, 2, 3, 4),
            pixel_format="rgb24",
        )
        self.assertIn("-n", command)
        self.assertIn("-map_metadata", command)
        self.assertIn("crop=w=3:h=4:x=1:y=2:exact=1,setsar=1", command)
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            source = root / "source.png"
            output = root / "output"
            output.mkdir()
            (output / "existing.png").write_bytes(b"x")
            source.write_bytes(b"not-used")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                export_oracle_crops(
                    construction(),
                    sources=(
                        OracleCropSource(
                            "sample-rgb",
                            str(source),
                            sha256_file(source),
                            Modality.RGB,
                            CropBox(0, 0, 1, 1),
                        ),
                        OracleCropSource(
                            "sample-ir",
                            str(source),
                            sha256_file(source),
                            Modality.IR,
                            CropBox(0, 0, 1, 1),
                        ),
                    ),
                    policy=CropExportPolicy(),
                    output_directory=output,
                )


class CropContractTests(unittest.TestCase):
    def test_mixed_modality_and_invalid_limits_are_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "must be RGB or IR"):
            OracleCropSource(
                "sample",
                "/protected/source.png",
                HASH_A,
                Modality.MIXED,
                CropBox(0, 0, 1, 1),
            )
        with self.assertRaisesRegex(ValueError, "positive integer"):
            CropExportPolicy(maximum_crop_pixels=0)
        policy = CropExportPolicy()
        self.assertEqual(CropExportPolicy.from_dict(policy.to_dict()), policy)
        invalid = policy.to_dict()
        invalid["allow_resize"] = True
        with self.assertRaisesRegex(ValueError, "unknown"):
            CropExportPolicy.from_dict(invalid)


if __name__ == "__main__":
    unittest.main()
