from __future__ import annotations

import json
import shutil
import stat
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data_pipeline.acquisition import sha256_file
from evaluation.controls import (
    ControlMaskEntry,
    ControlMaskManifest,
    MaskEvidence,
    MaskReviewStatus,
    MaskRole,
    VisualControlKind,
    VisualControlPanel,
    VisualControlPolicy,
    VisualControlRecipe,
)
from data_pipeline.crop_export import CropExportReceipt
from evaluation.control_transform import (
    ControlTransformConfig,
    ControlTransformConfigManifest,
)
from evaluation.mask_semantics import MaskSemanticPolicy
from evaluation.pairing import (
    PairArtifactBinding,
    PairConstructionResult,
    PairGroundTruth,
    PairScoringRequest,
    PairStratum,
)
from foundation.provenance import content_sha256
from evaluation.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    PairArtifactVerification,
)
from workflows.plan_visual_shortcut_controls import main

HASH_A = "a" * 64
HASH_B = "b" * 64
HASH_C = "c" * 64


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def _construction() -> PairConstructionResult:
    return PairConstructionResult(
        split_manifest_sha256=HASH_A,
        pairing_policy_sha256=HASH_B,
        attributes_sha256=HASH_C,
        eligible_query_count=1,
        selected_query_count=1,
        dropped_query_count=0,
        scoring_requests=(
            PairScoringRequest("pair-1", "query", "reference"),
        ),
        artifact_bindings=(
            PairArtifactBinding("query", "sample-query"),
            PairArtifactBinding("reference", "sample-reference"),
        ),
        ground_truth=(
            PairGroundTruth(
                "pair-1",
                "dog-1",
                "dog-2",
                "session-query",
                "session-reference",
                PairStratum.RANDOM,
            ),
        ),
        quotas=(),
    )


class VisualControlToolTests(unittest.TestCase):
    @unittest.skipUnless(
        shutil.which("ffmpeg") and shutil.which("ffprobe"),
        "FFmpeg tools are required",
    )
    def test_main_rehashes_inputs_and_writes_separated_bundle(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            pairs = _construction()
            pair_payloads = {
                "scoring": pairs.scoring_payload(),
                "bindings": pairs.artifact_binding_payload(),
                "truth": pairs.ground_truth_payload(),
                "summary": pairs.summary_payload(),
            }
            pair_paths: dict[str, Path] = {}
            for name, payload in pair_payloads.items():
                path = root / f"{name}.json"
                _write_json(path, payload)
                pair_paths[name] = path

            base_directory = root / "base"
            base_directory.mkdir()
            base_entries = []
            for token in ("query", "reference"):
                path = base_directory / f"{token}.png"
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
                        "testsrc2=size=10x10:rate=1",
                        "-frames:v",
                        "1",
                        str(path),
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                base_entries.append(
                    PairArtifactEntry(
                        token,
                        path.name,
                        sha256_file(path),
                        path.stat().st_size,
                        "image/png",
                    )
                )
            base_manifest = PairArtifactManifest(
                pair_set_sha256=pairs.result_sha256,
                artifact_bindings_sha256=content_sha256(
                    pairs.artifact_binding_payload()
                ),
                entries=tuple(base_entries),
            )
            base_verification = PairArtifactVerification(
                base_manifest.manifest_sha256,
                len(base_entries),
                sum(entry.byte_size for entry in base_entries),
            )
            crop_receipt = CropExportReceipt(
                pair_set_sha256=pairs.result_sha256,
                source_manifest_sha256=HASH_A,
                export_policy_sha256=HASH_B,
                ffmpeg_version="synthetic-test",
                artifact_manifest=base_manifest,
                verification=base_verification,
            )
            crop_receipt_path = root / "crop-receipt.json"
            _write_json(crop_receipt_path, crop_receipt.to_dict())

            mask_directory = root / "masks"
            mask_directory.mkdir()
            mask_entries = []
            for token in ("query", "reference"):
                mask_path = mask_directory / f"mask-{token}-dog.png"
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
                        (
                            "color=black:s=10x10,"
                            "drawbox=x=1:y=1:w=8:h=8:"
                            "color=white:t=fill,format=gray"
                        ),
                        "-frames:v",
                        "1",
                        str(mask_path),
                    ),
                    check=True,
                    capture_output=True,
                    text=True,
                )
                mask_entries.append(
                    ControlMaskEntry(
                        token,
                        (
                            MaskEvidence(
                                MaskRole.DOG,
                                f"mask-{token}-dog",
                                mask_path.name,
                                sha256_file(mask_path),
                                mask_path.stat().st_size,
                                10,
                                10,
                                "v1",
                                "manual-reviewed",
                                HASH_C,
                                MaskReviewStatus.VERIFIED,
                            ),
                        ),
                    )
                )
            mask_manifest = ControlMaskManifest(
                base_manifest.manifest_sha256,
                tuple(mask_entries),
            )
            mask_manifest_path = root / "mask-manifest.json"
            _write_json(mask_manifest_path, mask_manifest.to_dict())
            semantic_policy_path = root / "mask-semantic-policy.json"
            _write_json(
                semantic_policy_path,
                MaskSemanticPolicy().to_dict(),
            )
            transform_config = ControlTransformConfig(
                VisualControlKind.BACKGROUND_ONLY
            )
            transform_configs = ControlTransformConfigManifest(
                (transform_config,)
            )
            transform_config_path = root / "transform-configs.json"
            _write_json(
                transform_config_path,
                transform_configs.to_dict(),
            )
            policy = VisualControlPolicy(
                name="background-audit",
                recipes=(
                    VisualControlRecipe(
                        VisualControlKind.ORIGINAL,
                        content_sha256({"recipe": "ORIGINAL"}),
                        "original-v1",
                    ),
                    VisualControlRecipe(
                        VisualControlKind.BACKGROUND_ONLY,
                        transform_config.transform_config_sha256,
                        transform_config.semantics_version,
                    ),
                ),
                panels=(
                    VisualControlPanel(
                        "background",
                        (
                            VisualControlKind.ORIGINAL,
                            VisualControlKind.BACKGROUND_ONLY,
                        ),
                        1,
                        1,
                    ),
                ),
                seed=7,
            )
            policy_path = root / "policy.json"
            _write_json(policy_path, policy.to_dict())
            output_directory = root / "outputs"
            output_directory.mkdir()
            outputs = tuple(
                output_directory / name
                for name in (
                    "control-scoring.json",
                    "control-transform.json",
                    "control-evaluation.json",
                    "mask-verification.json",
                    "mask-semantic-verification.json",
                    "control-summary.json",
                )
            )
            argv = (
                "plan_visual_shortcut_controls.py",
                "--scoring-requests",
                str(pair_paths["scoring"]),
                "--artifact-bindings",
                str(pair_paths["bindings"]),
                "--ground-truth",
                str(pair_paths["truth"]),
                "--pair-summary",
                str(pair_paths["summary"]),
                "--crop-export-receipt",
                str(crop_receipt_path),
                "--base-artifact-directory",
                str(base_directory),
                "--mask-manifest",
                str(mask_manifest_path),
                "--mask-directory",
                str(mask_directory),
                "--mask-semantic-policy",
                str(semantic_policy_path),
                "--control-policy",
                str(policy_path),
                "--transform-config-manifest",
                str(transform_config_path),
                "--scoring-output",
                str(outputs[0]),
                "--transform-output",
                str(outputs[1]),
                "--evaluation-output",
                str(outputs[2]),
                "--mask-verification-output",
                str(outputs[3]),
                "--mask-semantic-verification-output",
                str(outputs[4]),
                "--summary-output",
                str(outputs[5]),
            )
            with patch("sys.argv", argv), redirect_stdout(StringIO()) as stdout:
                main()
            result = json.loads(stdout.getvalue())
            self.assertEqual(result["status"], "CREATED")
            for output in outputs:
                self.assertTrue(output.is_file())
                self.assertEqual(
                    stat.S_IMODE(output.stat().st_mode),
                    0o600,
                )
            scoring_text = outputs[0].read_text(encoding="utf-8")
            self.assertNotIn("dog-1", scoring_text)
            self.assertNotIn("base_artifact_token", scoring_text)
            self.assertIn(
                "base_artifact_token",
                outputs[1].read_text(encoding="utf-8"),
            )

            blocked_policy = VisualControlPolicy(
                name=policy.name,
                recipes=policy.recipes,
                panels=(
                    VisualControlPanel(
                        "background",
                        (
                            VisualControlKind.ORIGINAL,
                            VisualControlKind.BACKGROUND_ONLY,
                        ),
                        2,
                        2,
                    ),
                ),
                seed=policy.seed,
            )
            _write_json(policy_path, blocked_policy.to_dict())
            blocked_directory = root / "blocked-outputs"
            blocked_directory.mkdir()
            blocked_outputs = tuple(
                blocked_directory / path.name for path in outputs
            )
            blocked_argv = list(argv)
            for flag, output in zip(
                (
                    "--scoring-output",
                    "--transform-output",
                    "--evaluation-output",
                    "--mask-verification-output",
                    "--mask-semantic-verification-output",
                    "--summary-output",
                ),
                blocked_outputs,
                strict=True,
            ):
                blocked_argv[blocked_argv.index(flag) + 1] = str(output)
            with (
                patch("sys.argv", tuple(blocked_argv)),
                self.assertRaisesRegex(RuntimeError, "blocked"),
            ):
                main()
            self.assertEqual(tuple(blocked_directory.iterdir()), ())


if __name__ == "__main__":
    unittest.main()
