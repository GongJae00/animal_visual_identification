from __future__ import annotations

import json
import subprocess
import unittest
from contextlib import redirect_stdout
from io import StringIO
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

from data.acquisition import sha256_file
from data.crop_export import CropExportReceipt
from evaluation.controls.control_transform import (
    SUPPORTED_SEMANTICS_VERSION,
    ControlTransformConfig,
    ControlTransformConfigManifest,
    ControlTransformExecutionPolicy,
    build_control_transform_command,
    execute_control_transforms,
)
from evaluation.controls.policy import (
    ControlMaskEntry,
    ControlMaskManifest,
    ControlTransformTask,
    MaskEvidence,
    MaskReviewStatus,
    MaskRole,
    VisualControlKind,
    control_artifact_token,
    verify_control_mask_files,
)
from evaluation.controls.mask_semantics import (
    MaskSemanticPolicy,
    verify_mask_pixel_semantics,
)
from evaluation.controls.scoring import (
    PairArtifactEntry,
    PairArtifactManifest,
    verify_pair_artifact_files,
)
from workflows.evaluate_visual_controls import main

HASH_A = "a" * 64
HASH_B = "b" * 64


def _write_png(
    path: Path,
    *,
    pixels: bytes,
    width: int,
    height: int,
    pixel_format: str,
) -> None:
    subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-f",
            "rawvideo",
            "-pixel_format",
            pixel_format,
            "-video_size",
            f"{width}x{height}",
            "-i",
            "pipe:0",
            "-map_metadata",
            "-1",
            "-frames:v",
            "1",
            "-pix_fmt",
            pixel_format,
            "-pred",
            "mixed",
            "-f",
            "image2",
            str(path),
        ),
        input=pixels,
        check=True,
        capture_output=True,
    )


def _decode_raw(path: Path, pixel_format: str) -> bytes:
    completed = subprocess.run(
        (
            "ffmpeg",
            "-nostdin",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(path),
            "-frames:v",
            "1",
            "-pix_fmt",
            pixel_format,
            "-f",
            "rawvideo",
            "pipe:1",
        ),
        check=True,
        capture_output=True,
    )
    return completed.stdout


class ControlTransformTests(unittest.TestCase):
    def _fixture(
        self,
        root: Path,
    ) -> tuple[
        Path,
        PairArtifactManifest,
        object,
        Path,
        ControlMaskManifest,
        object,
        object,
        bytes,
        bytes,
        bytes,
    ]:
        base_root = root / "base"
        mask_root = root / "mask"
        base_root.mkdir()
        mask_root.mkdir()
        width = 4
        height = 4
        rgb = bytes(
            value
            for pixel in range(width * height)
            for value in (
                pixel * 11 % 256,
                pixel * 17 % 256,
                pixel * 23 % 256,
            )
        )
        ir = bytes(pixel * 13 % 256 for pixel in range(width * height))
        dog = bytes(
            255 if pixel in {5, 6, 9, 10} else 0
            for pixel in range(width * height)
        )
        accessory = bytes(
            255 if pixel in {5, 9} else 0
            for pixel in range(width * height)
        )
        _write_png(
            base_root / "rgb.png",
            pixels=rgb,
            width=width,
            height=height,
            pixel_format="rgb24",
        )
        _write_png(
            base_root / "ir.png",
            pixels=ir,
            width=width,
            height=height,
            pixel_format="gray",
        )
        entries = tuple(
            PairArtifactEntry(
                artifact_token=token,
                relative_path=f"{token}.png",
                content_sha256=sha256_file(base_root / f"{token}.png"),
                byte_size=(base_root / f"{token}.png").stat().st_size,
                media_type="image/png",
            )
            for token in ("rgb", "ir")
        )
        base_manifest = PairArtifactManifest(
            pair_set_sha256=HASH_A,
            artifact_bindings_sha256=HASH_B,
            entries=entries,
        )
        base_verification = verify_pair_artifact_files(
            base_root,
            base_manifest,
        )
        mask_entries = []
        for token in ("rgb", "ir"):
            masks = []
            for role, raw in (
                (MaskRole.DOG, dog),
                (MaskRole.ACCESSORY, accessory),
            ):
                mask_token = f"mask-{token}-{role.value.casefold()}"
                path = mask_root / f"{mask_token}.png"
                _write_png(
                    path,
                    pixels=raw,
                    width=width,
                    height=height,
                    pixel_format="gray",
                )
                masks.append(
                    MaskEvidence(
                        role=role,
                        artifact_token=mask_token,
                        relative_path=path.name,
                        content_sha256=sha256_file(path),
                        byte_size=path.stat().st_size,
                        width=width,
                        height=height,
                        annotation_version="test-v1",
                        provenance_kind="synthetic-reviewed",
                        provenance_reference_sha256=HASH_A,
                        review_status=MaskReviewStatus.VERIFIED,
                    )
                )
            mask_entries.append(ControlMaskEntry(token, tuple(masks)))
        mask_manifest = ControlMaskManifest(
            base_artifact_manifest_sha256=base_manifest.manifest_sha256,
            entries=tuple(mask_entries),
        )
        mask_verification = verify_control_mask_files(
            mask_root,
            mask_manifest,
        )
        semantic = verify_mask_pixel_semantics(
            base_root=base_root,
            base_manifest=base_manifest,
            base_verification=base_verification,
            mask_root=mask_root,
            mask_manifest=mask_manifest,
            mask_file_verification=mask_verification,
            policy=MaskSemanticPolicy(maximum_mask_pixels=16),
        )
        return (
            base_root,
            base_manifest,
            base_verification,
            mask_root,
            mask_manifest,
            mask_verification,
            semantic,
            rgb,
            ir,
            dog,
        )

    def test_all_control_equations_and_ir_format_are_verified(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                base_root,
                base_manifest,
                base_verification,
                mask_root,
                mask_manifest,
                mask_verification,
                semantic,
                rgb,
                ir,
                dog,
            ) = self._fixture(root)
            output = root / "output"
            output.mkdir()
            kinds = (
                VisualControlKind.DOG_ONLY,
                VisualControlKind.BACKGROUND_ONLY,
                VisualControlKind.BODY_BLURRED,
                VisualControlKind.MASK_ONLY,
                VisualControlKind.ACCESSORY_ONLY,
                VisualControlKind.ACCESSORY_MASKED,
            )
            configs = tuple(
                ControlTransformConfig(
                    kind=kind,
                    blur_sigma_fraction_of_min_edge=(
                        0.25
                        if kind is VisualControlKind.BODY_BLURRED
                        else None
                    ),
                    minimum_blur_sigma_pixels=(
                        0.5
                        if kind is VisualControlKind.BODY_BLURRED
                        else None
                    ),
                    maximum_blur_sigma_pixels=(
                        16.0
                        if kind is VisualControlKind.BODY_BLURRED
                        else None
                    ),
                    blur_steps=2 if kind is VisualControlKind.BODY_BLURRED else None,
                )
                for kind in kinds
            )
            config_manifest = ControlTransformConfigManifest(configs)
            configs_by_kind = {config.kind: config for config in configs}
            base_by_token = {
                entry.artifact_token: entry
                for entry in base_manifest.entries
            }
            mask_by_base = {
                entry.base_artifact_token: entry
                for entry in mask_manifest.entries
            }

            def task(
                base_token: str,
                kind: VisualControlKind,
            ) -> ControlTransformTask:
                config = configs_by_kind[kind]
                evidence = mask_by_base[base_token]
                masks = tuple(
                    (
                        role,
                        evidence.mask_for(role).artifact_token,
                        evidence.mask_for(role).content_sha256,
                    )
                    for role in kind.required_mask_roles
                )
                token = control_artifact_token(
                    base_content_sha256=(
                        base_by_token[base_token].content_sha256
                    ),
                    kind=kind,
                    transform_config_sha256=(
                        config.transform_config_sha256
                    ),
                    semantics_version=SUPPORTED_SEMANTICS_VERSION,
                    mask_artifacts=masks,
                )
                return ControlTransformTask(
                    control_artifact_token=token,
                    base_artifact_token=base_token,
                    control_kind=kind,
                    transform_config_sha256=(
                        config.transform_config_sha256
                    ),
                    semantics_version=SUPPORTED_SEMANTICS_VERSION,
                    mask_artifacts=masks,
                )

            tasks = tuple(task("rgb", kind) for kind in kinds) + (
                task("ir", VisualControlKind.DOG_ONLY),
            )
            receipt = execute_control_transforms(
                plan_sha256=HASH_A,
                scoring_requests_sha256=HASH_B,
                tasks=tasks,
                base_root=base_root,
                base_manifest=base_manifest,
                base_verification=base_verification,
                mask_root=mask_root,
                mask_manifest=mask_manifest,
                mask_verification=mask_verification,
                mask_semantic_verification=semantic,
                config_manifest=config_manifest,
                policy=ControlTransformExecutionPolicy(
                    maximum_tasks=7,
                    maximum_source_pixels=16,
                    maximum_total_task_pixels=112,
                    maximum_validation_raw_bytes_per_group=176,
                    validation_chunk_pixels=3,
                ),
                output_directory=output,
            )
            self.assertEqual(receipt.verification.verified_files, 7)
            self.assertEqual(
                type(receipt).from_dict(receipt.to_dict()),
                receipt,
            )
            self.assertEqual(receipt.cost.unique_base_decodes, 2)
            self.assertEqual(receipt.cost.unique_mask_decodes, 3)
            self.assertEqual(receipt.cost.validation_blur_decodes, 1)
            self.assertEqual(receipt.cost.peak_validation_raw_bytes, 176)
            self.assertEqual(receipt.cost.subprocess_calls, 20)
            self.assertEqual(
                len(tuple(output.iterdir())),
                len(tasks),
            )

            by_kind = {
                task_item.control_kind: task_item
                for task_item in tasks
                if task_item.base_artifact_token == "rgb"
            }
            dog_only = _decode_raw(
                output
                / f"{by_kind[VisualControlKind.DOG_ONLY].control_artifact_token}.png",
                "rgb24",
            )
            background_only = _decode_raw(
                output
                / (
                    f"{by_kind[VisualControlKind.BACKGROUND_ONLY].control_artifact_token}"
                    ".png"
                ),
                "rgb24",
            )
            mask_only = _decode_raw(
                output
                / f"{by_kind[VisualControlKind.MASK_ONLY].control_artifact_token}.png",
                "rgb24",
            )
            for pixel, value in enumerate(dog):
                start = pixel * 3
                end = start + 3
                self.assertEqual(
                    dog_only[start:end],
                    rgb[start:end] if value else b"\x00\x00\x00",
                )
                self.assertEqual(
                    background_only[start:end],
                    b"\x00\x00\x00" if value else rgb[start:end],
                )
                self.assertEqual(mask_only[start:end], bytes((value,)) * 3)

            blurred = _decode_raw(
                output
                / (
                    f"{by_kind[VisualControlKind.BODY_BLURRED].control_artifact_token}"
                    ".png"
                ),
                "rgb24",
            )
            self.assertTrue(
                any(
                    blurred[pixel * 3 : pixel * 3 + 3]
                    != rgb[pixel * 3 : pixel * 3 + 3]
                    for pixel, value in enumerate(dog)
                    if value
                )
            )
            self.assertTrue(
                all(
                    blurred[pixel * 3 : pixel * 3 + 3]
                    == rgb[pixel * 3 : pixel * 3 + 3]
                    for pixel, value in enumerate(dog)
                    if not value
                )
            )
            ir_task = tasks[-1]
            ir_output = _decode_raw(
                output / f"{ir_task.control_artifact_token}.png",
                "gray",
            )
            self.assertEqual(
                ir_output,
                bytes(
                    ir[pixel] if value else 0
                    for pixel, value in enumerate(dog)
                ),
            )

    def test_hash_mismatch_fails_before_writing_output(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                base_root,
                base_manifest,
                base_verification,
                mask_root,
                mask_manifest,
                mask_verification,
                semantic,
                _,
                _,
                _,
            ) = self._fixture(root)
            output = root / "output"
            output.mkdir()
            config = ControlTransformConfig(
                VisualControlKind.DOG_ONLY,
            )
            evidence = mask_manifest.entries[0].mask_for(MaskRole.DOG)
            masks = (
                (
                    MaskRole.DOG,
                    evidence.artifact_token,
                    evidence.content_sha256,
                ),
            )
            task = ControlTransformTask(
                control_artifact_token="control-" + "1" * 24,
                base_artifact_token="rgb",
                control_kind=VisualControlKind.DOG_ONLY,
                transform_config_sha256=config.transform_config_sha256,
                semantics_version=SUPPORTED_SEMANTICS_VERSION,
                mask_artifacts=masks,
            )
            self.assertEqual(
                ControlTransformTask.from_dict(task.to_dict()),
                task,
            )
            with self.assertRaisesRegex(
                ValueError,
                "not content-addressed",
            ):
                execute_control_transforms(
                    plan_sha256=HASH_A,
                    scoring_requests_sha256=HASH_B,
                    tasks=(task,),
                    base_root=base_root,
                    base_manifest=base_manifest,
                    base_verification=base_verification,
                    mask_root=mask_root,
                    mask_manifest=mask_manifest,
                    mask_verification=mask_verification,
                    mask_semantic_verification=semantic,
                    config_manifest=ControlTransformConfigManifest(
                        (config,)
                    ),
                    policy=ControlTransformExecutionPolicy(
                        maximum_source_pixels=16,
                    ),
                    output_directory=output,
                )
            self.assertEqual(tuple(output.iterdir()), ())

    def test_nonempty_output_directory_is_rejected(self) -> None:
        with TemporaryDirectory() as temporary:
            output = Path(temporary)
            (output / "existing").write_bytes(b"x")
            with self.assertRaisesRegex(ValueError, "must be empty"):
                execute_control_transforms(
                    plan_sha256=HASH_A,
                    scoring_requests_sha256=HASH_B,
                    tasks=(),
                    base_root=output,
                    base_manifest=object(),
                    base_verification=object(),
                    mask_root=output,
                    mask_manifest=object(),
                    mask_verification=object(),
                    mask_semantic_verification=object(),
                    config_manifest=object(),
                    policy=ControlTransformExecutionPolicy(),
                    output_directory=output,
                )

    def test_config_schema_and_semantics_are_strict(self) -> None:
        config = ControlTransformConfig(
            VisualControlKind.BODY_BLURRED,
            blur_sigma_fraction_of_min_edge=0.1,
            minimum_blur_sigma_pixels=1.0,
            maximum_blur_sigma_pixels=64.0,
            blur_steps=3,
        )
        self.assertEqual(
            ControlTransformConfig.from_dict(config.to_dict()),
            config,
        )
        with self.assertRaisesRegex(ValueError, "fixed to zero"):
            ControlTransformConfig(
                VisualControlKind.DOG_ONLY,
                neutral_value=128,
            )
        with self.assertRaisesRegex(ValueError, "only for BODY_BLURRED"):
            ControlTransformConfig(
                VisualControlKind.DOG_ONLY,
                blur_sigma_fraction_of_min_edge=0.1,
                minimum_blur_sigma_pixels=1.0,
                maximum_blur_sigma_pixels=64.0,
                blur_steps=1,
            )
        proportional = build_control_transform_command(
            base=Path("base.png"),
            mask=Path("mask.png"),
            destination=Path("output.png"),
            config=config,
            pixel_format="rgb24",
            width=200,
            height=100,
        )
        self.assertIn("sigma=10", " ".join(proportional))
        clamped = build_control_transform_command(
            base=Path("base.png"),
            mask=Path("mask.png"),
            destination=Path("output.png"),
            config=config,
            pixel_format="rgb24",
            width=1000,
            height=1000,
        )
        self.assertIn("sigma=64", " ".join(clamped))

    def test_cli_rehashes_inputs_and_writes_private_receipt(self) -> None:
        with TemporaryDirectory() as temporary:
            root = Path(temporary)
            (
                base_root,
                base_manifest,
                base_verification,
                mask_root,
                mask_manifest,
                mask_verification,
                semantic,
                _,
                _,
                _,
            ) = self._fixture(root)
            config = ControlTransformConfig(
                VisualControlKind.DOG_ONLY,
            )
            base = next(
                entry
                for entry in base_manifest.entries
                if entry.artifact_token == "rgb"
            )
            evidence = mask_manifest.entries[0].mask_for(MaskRole.DOG)
            masks = (
                (
                    MaskRole.DOG,
                    evidence.artifact_token,
                    evidence.content_sha256,
                ),
            )
            token = control_artifact_token(
                base_content_sha256=base.content_sha256,
                kind=VisualControlKind.DOG_ONLY,
                transform_config_sha256=config.transform_config_sha256,
                semantics_version=SUPPORTED_SEMANTICS_VERSION,
                mask_artifacts=masks,
            )
            task = ControlTransformTask(
                control_artifact_token=token,
                base_artifact_token="rgb",
                control_kind=VisualControlKind.DOG_ONLY,
                transform_config_sha256=config.transform_config_sha256,
                semantics_version=SUPPORTED_SEMANTICS_VERSION,
                mask_artifacts=masks,
            )
            crop_receipt = CropExportReceipt(
                pair_set_sha256=base_manifest.pair_set_sha256,
                source_manifest_sha256=HASH_A,
                export_policy_sha256=HASH_B,
                ffmpeg_version="fixture",
                artifact_manifest=base_manifest,
                verification=base_verification,
            )
            config_manifest = ControlTransformConfigManifest((config,))
            execution_policy = ControlTransformExecutionPolicy(
                maximum_tasks=1,
                maximum_source_pixels=16,
                maximum_total_task_pixels=16,
                maximum_validation_raw_bytes_per_group=112,
                validation_chunk_pixels=5,
            )
            payloads = {
                "tasks": {
                    "schema_version": (
                        "cvi.visual_control_transform_tasks.v1"
                    ),
                    "plan_sha256": HASH_A,
                    "scoring_requests_sha256": HASH_B,
                    "tasks": [task.to_dict()],
                },
                "crop": crop_receipt.to_dict(),
                "masks": mask_manifest.to_dict(),
                "mask-verification": mask_verification.to_dict(),
                "semantic": semantic.to_dict(),
                "configs": config_manifest.to_dict(),
                "execution": execution_policy.to_dict(),
            }
            paths: dict[str, Path] = {}
            for name, payload in payloads.items():
                path = root / f"{name}.json"
                path.write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )
                paths[name] = path
            output = root / "output"
            output.mkdir()
            receipt_path = root / "receipt.json"
            argv = [
                "evaluate_visual_controls.py",
                "execute",
                "--transform-tasks",
                str(paths["tasks"]),
                "--crop-export-receipt",
                str(paths["crop"]),
                "--base-artifact-directory",
                str(base_root),
                "--mask-manifest",
                str(paths["masks"]),
                "--mask-directory",
                str(mask_root),
                "--mask-verification",
                str(paths["mask-verification"]),
                "--mask-semantic-verification",
                str(paths["semantic"]),
                "--transform-config-manifest",
                str(paths["configs"]),
                "--execution-policy",
                str(paths["execution"]),
                "--output-directory",
                str(output),
                "--receipt-output",
                str(receipt_path),
            ]
            stdout = StringIO()
            with patch("sys.argv", argv), redirect_stdout(stdout):
                main()
            summary = json.loads(stdout.getvalue())
            self.assertEqual(summary["status"], "CREATED")
            self.assertEqual(summary["artifact_count"], 1)
            self.assertEqual(receipt_path.stat().st_mode & 0o777, 0o600)
            self.assertTrue((output / f"{token}.png").is_file())


if __name__ == "__main__":
    unittest.main()
