from __future__ import annotations

import hashlib
import tempfile
import subprocess
import sys
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import torch

from representation_learning.trainer import (
    AdmittedCropDataset,
    ArcFaceHead,
    ArcFaceModel,
    AppearanceBoundedResidual,
    ConvNeXtEmbedding,
    Dinov2Embedding,
    IdentityBalancedSampler,
    TrainConfig,
    _build_label_index,
    _build_dataloader,
    _checkpoint_payload,
    compute_embeddings,
    _count_parameters,
    _a4_metric_learning_loss,
    _evaluate_development_retrieval,
    evaluate_pretrained_development,
    _prepare_training_images,
    _prepare_a4_training_images,
    _metric_learning_loss,
    _selection_improves,
    _warmup_cosine_schedule,
    train_model,
)
from representation_learning.train.augment import RandAugment
from data_pipeline.public_crop_manifest import (
    PublicCropArtifact,
    PublicCropManifest,
    canonical_rgb_pixel_sha256,
    verify_public_crop_manifest,
)
from identity_governance.training_admission import (
    TrainingAdmissionManifest,
    TrainingCropRow,
    admit_training,
)
from identity_governance.role_exposure import (
    ExposureDeclarationKind,
    ExposureStage,
    RoleExposureDeclaration,
    RoleExposureDeclarationRecord,
    create_role_exposure_receipt,
    merge_role_exposure_declarations,
)


def _token(label: str) -> str:
    return hashlib.sha256(label.encode("ascii")).hexdigest()


def _crop_fixture(
    root: Path,
    *,
    sample_label: str = "sample",
    subject_label: str = "subject",
) -> tuple[PublicCropManifest, tuple[TrainingCropRow, ...]]:
    from PIL import Image

    sample = _token(sample_label)
    path = root / f"{sample}.png"
    with Image.new("RGB", (8, 8), color=(100, 120, 140)) as image:
        image.save(path, format="PNG")
        pixel_sha256 = canonical_rgb_pixel_sha256(
            8, 8, image.tobytes("raw", "RGB")
        )
    payload = path.read_bytes()
    artifact = PublicCropArtifact(
        sample_token=sample,
        public_subject_token=_token(subject_label),
        component_token=_token(f"{sample_label}-component"),
        source_variant="original",
        relative_path=path.name,
        content_sha256=hashlib.sha256(payload).hexdigest(),
        byte_size=len(payload),
        pixel_sha256=pixel_sha256,
        width=8,
        height=8,
        mode="RGB",
        format="PNG",
    )
    row = TrainingCropRow(
        sample_token=artifact.sample_token,
        identity_token=_token(f"{subject_label}-identity"),
        public_subject_token=artifact.public_subject_token,
        component_token=artifact.component_token,
        lane="MODEL_TRAINING",
        role="YT_FIT",
        crop_relative_path=artifact.relative_path,
        crop_artifact_sha256=artifact.artifact_sha256,
    )
    return PublicCropManifest((artifact,)), (row,)


def _exposure(rows: tuple[TrainingCropRow, ...]):
    declaration = RoleExposureDeclaration(
        source_artifact_sha256=_token("training-test-exposure"),
        kind=ExposureDeclarationKind.PRIOR_ASSIGNMENT,
        revoked=False,
        records=tuple(
            RoleExposureDeclarationRecord(
                sample_token=row.sample_token,
                identity_token=row.identity_token,
                public_subject_token=row.public_subject_token,
                stage=ExposureStage.BYTES_EXPORTED,
            )
            for row in rows
        ),
    )
    ledger = merge_role_exposure_declarations((declaration,))
    return ledger, create_role_exposure_receipt(ledger)


class TrainConfigTests(unittest.TestCase):
    def test_defaults(self) -> None:
        c = TrainConfig()
        self.assertEqual(c.model_name, "dinov2-small")
        self.assertEqual(c.embedding_dim, 384)
        self.assertEqual(c.arcface_margin, 0.50)
        self.assertIsInstance(c.to_dict(), dict)

    def test_round_trip(self) -> None:
        c = TrainConfig(epochs=100, batch_size=128, lr=1e-3)
        d = c.to_dict()
        restored = TrainConfig.from_dict(d)
        self.assertEqual(restored.epochs, 100)
        self.assertEqual(restored.batch_size, 128)
        self.assertEqual(restored.lr, 1e-3)

    def test_a4_requires_explicit_positive_regularizers(self) -> None:
        with self.assertRaises(ValueError):
            TrainConfig(architecture="appearance_bounded_residual_v4")
        config = TrainConfig(
            architecture="appearance_bounded_residual_v4",
            border_consistency_weight=0.5,
            baseline_anchor_weight=1.0,
        )
        self.assertEqual(config.residual_scale, 0.1)


class LabelIndexTests(unittest.TestCase):
    def test_build_label_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = _crop_fixture(root, sample_label="a", subject_label="subject-a")
            _, second = _crop_fixture(root, sample_label="b", subject_label="subject-b")
            _, third = _crop_fixture(root, sample_label="c", subject_label="subject-a")
        index = _build_label_index(first + second + third)
        self.assertEqual(len(index), 2)
        self.assertEqual(set(index), {_token("subject-a"), _token("subject-b")})

    def test_sorted_order(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _, first = _crop_fixture(
                root, sample_label="z", subject_label="z-subject"
            )
            _, second = _crop_fixture(
                root, sample_label="a", subject_label="a-subject"
            )
        index = _build_label_index(first + second)
        labels = sorted((_token("z-subject"), _token("a-subject")))
        self.assertEqual(index[labels[0]], 0)
        self.assertEqual(index[labels[1]], 1)


class WarmupCosineScheduleTests(unittest.TestCase):
    def test_warmup_linear(self) -> None:
        lr = _warmup_cosine_schedule(0, 50, 5, 1e-3, 1e-6)
        self.assertAlmostEqual(lr, 1e-3 / 5)

    def test_warmup_peak(self) -> None:
        lr = _warmup_cosine_schedule(4, 50, 5, 1e-3, 1e-6)
        self.assertAlmostEqual(lr, 1e-3, places=6)

    def test_cosine_decay_below_max(self) -> None:
        lr = _warmup_cosine_schedule(30, 50, 5, 1e-3, 1e-6)
        self.assertLess(lr, 1e-3)
        self.assertGreaterEqual(lr, 1e-6)

    def test_final_lr_near_min(self) -> None:
        lr = _warmup_cosine_schedule(49, 50, 5, 1e-3, 1e-6)
        self.assertLessEqual(lr, 1e-3)
        self.assertGreaterEqual(lr, 1e-7)


class IdentityBalancedSamplerTests(unittest.TestCase):
    def test_batch_structure(self) -> None:
        labels = [0, 0, 0, 1, 1, 1, 2, 2, 2]
        sampler = IdentityBalancedSampler(
            labels, batch_size=3,
            generator=torch.Generator().manual_seed(42),
        )
        batches = list(sampler)
        self.assertTrue(len(batches) > 0)
        flattened = [index for batch in batches for index in batch]
        self.assertEqual(sorted(flattened), list(range(len(labels))))
        self.assertEqual(len(batches), len(sampler))
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertTrue(all(isinstance(i, int) for i in batch))
            self.assertEqual(len({labels[i] for i in batch}), len(batch))

    def test_identity_coverage_on_small_data(self) -> None:
        labels = [0, 1]
        sampler = IdentityBalancedSampler(labels, batch_size=2)
        batches = list(sampler)
        all_indices = set()
        for batch in batches:
            all_indices.update(batch)
        self.assertEqual(all_indices, {0, 1})

    def test_skewed_identity_batches_remain_unique(self) -> None:
        labels = [0, 0, 0, 0, 0, 1, 2]
        sampler = IdentityBalancedSampler(
            labels,
            batch_size=3,
            generator=torch.Generator().manual_seed(7),
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(
            sorted(index for batch in batches for index in batch),
            list(range(len(labels))),
        )
        for batch in batches:
            self.assertEqual(len({labels[i] for i in batch}), len(batch))

    def test_seeded_sampler_is_reproducible(self) -> None:
        labels = [0, 0, 1, 1, 2, 2]
        first = list(IdentityBalancedSampler(
            labels, 2, torch.Generator().manual_seed(42)
        ))
        second = list(IdentityBalancedSampler(
            labels, 2, torch.Generator().manual_seed(42)
        ))
        self.assertEqual(first, second)

    def test_adversarial_imbalance_length_is_exact(self) -> None:
        labels = [0] * 7 + [1] * 2 + [2] * 2 + [3]
        sampler = IdentityBalancedSampler(
            labels, 3, torch.Generator().manual_seed(1)
        )
        batches = list(sampler)
        self.assertEqual(len(batches), len(sampler))
        self.assertEqual(len(batches), 7)
        for batch in batches:
            self.assertLessEqual(len(batch), 3)
            self.assertEqual(len({labels[i] for i in batch}), len(batch))


class AdmittedCropDatasetTests(unittest.TestCase):
    def test_uncached_preprocessing_matches_cached_uint8_contract(self) -> None:
        import numpy as np

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = _crop_fixture(root)
            dataset = AdmittedCropDataset(
                root,
                rows,
                manifest,
                {rows[0].public_subject_token: 0},
                use_cache=False,
            )
            tensor, label = dataset[0]
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertEqual(label, 0)
            self.assertTrue(np.isfinite(tensor.numpy()).all())

    def test_nested_duplicate_is_not_discovered(self) -> None:
        from PIL import Image

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = _crop_fixture(root)
            nested = root / "nested"
            nested.mkdir()
            Image.new("RGB", (8, 8), color=(1, 2, 3)).save(
                nested / rows[0].crop_relative_path
            )
            dataset = AdmittedCropDataset(
                root,
                rows,
                manifest,
                {rows[0].public_subject_token: 0},
                use_cache=False,
            )
            self.assertEqual(len(dataset), 1)

    def test_uncached_reads_reverify_bytes_but_cache_retains_verified_bytes(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = _crop_fixture(root)
            labels = {rows[0].public_subject_token: 0}
            uncached = AdmittedCropDataset(
                root, rows, manifest, labels, use_cache=False
            )
            cached = AdmittedCropDataset(root, rows, manifest, labels, use_cache=True)
            path = root / rows[0].crop_relative_path
            path.write_bytes(path.read_bytes() + b"tampered")
            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                uncached[0]
            tensor, label = cached[0]
            self.assertEqual(tensor.dtype, torch.uint8)
            self.assertEqual(label, 0)

    def test_sampler_labels_do_not_decode_uncached_images(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            manifest, rows = _crop_fixture(root)
            dataset = AdmittedCropDataset(
                root,
                rows,
                manifest,
                {rows[0].public_subject_token: 7},
                use_cache=False,
            )
            with patch(
                "representation_learning.trainer.read_verified_crop_artifact",
                side_effect=AssertionError("label enumeration decoded a crop"),
            ):
                self.assertEqual(dataset.labels, (7,))

    def test_cuda_loader_pins_memory_with_workers(self) -> None:
        from torch.utils.data import TensorDataset

        loader = _build_dataloader(
            TensorDataset(torch.zeros(2, 1), torch.zeros(2, dtype=torch.long)),
            None,
            TrainConfig(batch_size=1, num_workers=1),
            torch.device("cuda"),
        )
        self.assertTrue(loader.pin_memory)

    def test_uncached_training_images_are_augmented(self) -> None:
        calls = 0

        def augment(image: torch.Tensor) -> torch.Tensor:
            nonlocal calls
            calls += 1
            return image

        images = torch.full((2, 3, 4, 4), 128, dtype=torch.uint8)
        mean = torch.zeros((1, 3, 1, 1))
        std = torch.ones((1, 3, 1, 1))
        prepared = _prepare_training_images(images, augment, mean, std)  # type: ignore[arg-type]
        self.assertEqual(calls, 2)
        self.assertEqual(prepared.dtype, torch.float32)

    def test_a4_border_challenge_preserves_the_central_source(self) -> None:
        images = torch.full((2, 3, 224, 224), 128, dtype=torch.uint8)
        mean = torch.zeros((1, 3, 1, 1))
        std = torch.ones((1, 3, 1, 1))
        clean, challenged = _prepare_a4_training_images(
            images, mean, std
        )
        self.assertEqual(clean.shape, challenged.shape)
        self.assertTrue(torch.equal(
            clean[:, :, 40:-40, 40:-40],
            challenged[:, :, 40:-40, 40:-40],
        ))
        self.assertFalse(torch.equal(clean, challenged))


class AdmittedTrainingBoundaryTests(unittest.TestCase):
    def test_crop_tampering_fails_before_model_artifact_or_backbone_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest, train_rows = _crop_fixture(
                root, sample_label="train", subject_label="train-subject"
            )
            development_manifest, development_rows = _crop_fixture(
                root,
                sample_label="development",
                subject_label="development-subject",
            )
            development_row = replace(
                development_rows[0],
                lane="MODEL_SELECTION",
                role="YT_DEVELOPMENT",
            )
            crop_manifest = PublicCropManifest(tuple(sorted(
                train_manifest.artifacts + development_manifest.artifacts,
                key=lambda artifact: artifact.sample_token,
            )))
            rows = tuple(sorted(
                train_rows + (development_row,), key=lambda row: row.sample_token
            ))
            crop_receipt_sha256 = verify_public_crop_manifest(
                root, crop_manifest
            ).receipt_sha256
            exposure_ledger, exposure_receipt = _exposure(rows)
            admission = TrainingAdmissionManifest(
                split_receipt_sha256=_token("split-receipt"),
                crop_manifest_sha256=crop_manifest.manifest_sha256,
                crop_receipt_sha256=crop_receipt_sha256,
                exposure_receipt_sha256=exposure_receipt.receipt_sha256,
                model_receipt_sha256=_token("model-receipt"),
                rows=rows,
            )
            receipt = admit_training(
                admission,
                crop_manifest,
                crop_root=root,
                exposure_ledger=exposure_ledger,
                exposure_receipt=exposure_receipt,
                expected_sample_tokens=tuple(row.sample_token for row in rows),
                expected_split_receipt_sha256=admission.split_receipt_sha256,
                expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                expected_model_receipt_sha256=admission.model_receipt_sha256,
            )
            crop_path = root / rows[0].crop_relative_path
            crop_path.write_bytes(crop_path.read_bytes() + b"tampered")
            model_accessed = False

            def verify_model() -> None:
                nonlocal model_accessed
                model_accessed = True

            output = root / "output"
            config = TrainConfig(
                checkpoint_dir=str(output / "checkpoints"), preload_images=False
            )
            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                train_model(
                    config,
                    root,
                    admission,
                    crop_manifest,
                    receipt,
                    exposure_ledger=exposure_ledger,
                    exposure_receipt=exposure_receipt,
                    output_directory=output,
                    expected_admission_manifest_sha256=admission.manifest_sha256,
                    expected_admission_receipt_sha256=receipt.receipt_sha256,
                    expected_split_receipt_sha256=admission.split_receipt_sha256,
                    expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=(
                        admission.exposure_receipt_sha256
                    ),
                    expected_model_receipt_sha256=admission.model_receipt_sha256,
                    model_artifact_verifier=verify_model,
                )
            self.assertFalse(model_accessed)
            self.assertFalse(output.exists())

    def test_pretrained_evaluation_rejects_tampering_before_model_access(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            train_manifest, train_rows = _crop_fixture(
                root, sample_label="train", subject_label="train-subject"
            )
            development_manifest, development_rows = _crop_fixture(
                root,
                sample_label="development",
                subject_label="development-subject",
            )
            development_row = replace(
                development_rows[0],
                lane="MODEL_SELECTION",
                role="YT_DEVELOPMENT",
            )
            crop_manifest = PublicCropManifest(tuple(sorted(
                train_manifest.artifacts + development_manifest.artifacts,
                key=lambda artifact: artifact.sample_token,
            )))
            rows = tuple(sorted(
                train_rows + (development_row,), key=lambda row: row.sample_token
            ))
            crop_receipt_sha256 = verify_public_crop_manifest(
                root, crop_manifest
            ).receipt_sha256
            exposure_ledger, exposure_receipt = _exposure(rows)
            admission = TrainingAdmissionManifest(
                split_receipt_sha256=_token("split-receipt"),
                crop_manifest_sha256=crop_manifest.manifest_sha256,
                crop_receipt_sha256=crop_receipt_sha256,
                exposure_receipt_sha256=exposure_receipt.receipt_sha256,
                model_receipt_sha256=_token("model-receipt"),
                rows=rows,
            )
            receipt = admit_training(
                admission,
                crop_manifest,
                crop_root=root,
                exposure_ledger=exposure_ledger,
                exposure_receipt=exposure_receipt,
                expected_sample_tokens=tuple(row.sample_token for row in rows),
                expected_split_receipt_sha256=admission.split_receipt_sha256,
                expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                expected_model_receipt_sha256=admission.model_receipt_sha256,
            )
            (root / rows[0].crop_relative_path).write_bytes(b"tampered")
            model_accessed = False

            def verify_model() -> None:
                nonlocal model_accessed
                model_accessed = True

            with self.assertRaisesRegex(ValueError, "byte size mismatch"):
                evaluate_pretrained_development(
                    TrainConfig(preload_images=False),
                    root,
                    admission,
                    crop_manifest,
                    receipt,
                    exposure_ledger=exposure_ledger,
                    exposure_receipt=exposure_receipt,
                    expected_admission_manifest_sha256=admission.manifest_sha256,
                    expected_admission_receipt_sha256=receipt.receipt_sha256,
                    expected_split_receipt_sha256=admission.split_receipt_sha256,
                    expected_crop_receipt_sha256=admission.crop_receipt_sha256,
                    expected_exposure_receipt_sha256=admission.exposure_receipt_sha256,
                    expected_model_receipt_sha256=admission.model_receipt_sha256,
                    model_artifact_verifier=verify_model,
                )
            self.assertFalse(model_accessed)


class AugmentationContractTests(unittest.TestCase):
    def test_each_randaugment_operation_is_clamped_to_image_range(self) -> None:
        augment = RandAugment(n=2, m=9)
        operations = [augment._adjust_brightness, augment._solarize]
        with patch(
            "representation_learning.train.augment.random.choice", side_effect=operations
        ), patch("representation_learning.train.augment.random.uniform", return_value=0.5):
            result = augment(torch.ones((3, 4, 4)))
        self.assertGreaterEqual(float(result.min()), 0.0)
        self.assertLessEqual(float(result.max()), 1.0)


class MultiHeadTrainingContractTests(unittest.TestCase):
    def test_embedding_trainer_requires_immutable_admission_inputs(self) -> None:
        completed = subprocess.run(
            [
                sys.executable,
                "workflows/train_embedding_model.py",
                "--crop-root", "crops",
                "--output-dir", "output",
            ],
            cwd=Path(__file__).resolve().parents[1],
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--admission-manifest", completed.stderr)
        self.assertIn("--model-artifact", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

    def test_embedding_trainer_refuses_existing_output_before_input_reads(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "existing"
            output.mkdir()
            command = [sys.executable, "workflows/train_embedding_model.py"]
            for argument in (
                "admission-manifest",
                "admission-receipt",
                "crop-manifest",
                "crop-root",
                "exposure-ledger",
                "exposure-receipt",
                "model-artifact",
                "model-receipt",
            ):
                command.extend((f"--{argument}", "missing"))
            command.extend(("--output-dir", str(output)))
            for argument in (
                "admission-manifest",
                "admission-receipt",
                "crop-manifest",
                "split-receipt",
                "crop-receipt",
                "exposure-receipt",
                "model-receipt",
            ):
                command.extend((f"--expected-{argument}-sha256", "0" * 64))
            completed = subprocess.run(
                command,
                cwd=Path(__file__).resolve().parents[1],
                capture_output=True,
                text=True,
                check=False,
            )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("--output-dir must not exist", completed.stderr)
        self.assertNotIn("Traceback", completed.stderr)

class CountParametersTests(unittest.TestCase):
    def test_arcface_head_count(self) -> None:
        head = ArcFaceHead(384, 100, 30.0, 0.5)
        counts = _count_parameters(head)
        self.assertIn("total", counts)
        self.assertIn("trainable", counts)
        self.assertGreater(counts["total"], 0)
        self.assertEqual(counts["total"], counts["trainable"])


class ArcFaceHeadForwardTests(unittest.TestCase):
    def test_output_shape(self) -> None:
        import torch
        head = ArcFaceHead(64, 10, 30.0, 0.5)
        features = torch.randn(4, 64)
        features = features / features.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 1, 2, 3])
        logits = head(features, labels)
        self.assertEqual(logits.shape, (4, 10))

    def test_gradient_flows_through_arcface(self) -> None:
        head = ArcFaceHead(64, 10, 30.0, 0.5)
        features = torch.randn(4, 64)
        features = features / features.norm(dim=1, keepdim=True)
        labels = torch.tensor([0, 1, 2, 3])
        logits = head(features, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        self.assertIsNotNone(head._w.grad)
        self.assertTrue(torch.isfinite(head._w.grad).all())


def _requires_dinov2():
    try:
        from transformers import AutoModel
        AutoModel.from_pretrained("facebook/dinov2-small", attn_implementation="sdpa")
        return False
    except Exception:
        return True


class Dinov2EmbeddingTests(unittest.TestCase):
    @unittest.skipIf(_requires_dinov2(), "DINOv2 model or transformers not available")
    def test_forward_shape_with_dummy_backbone(self) -> None:
        import torch
        model = Dinov2Embedding(embedding_dim=384)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 384))

    @unittest.skipIf(_requires_dinov2(), "DINOv2 model or transformers not available")
    def test_l2_normalized(self) -> None:
        import torch
        model = Dinov2Embedding(embedding_dim=384)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))


class ConvNeXtEmbeddingTests(unittest.TestCase):
    def test_output_dim(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 768))

    def test_l2_normalized(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        norms = emb.norm(p=2, dim=1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-5))

    def test_projection_dim(self) -> None:
        model = ConvNeXtEmbedding(embedding_dim=256)
        dummy = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model(dummy)
        self.assertEqual(emb.shape, (1, 256))


class _DummyBackbone(torch.nn.Module):
    def __init__(self, embedding_dim: int = 384,
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self._linear = torch.nn.Linear(224 * 224 * 3, embedding_dim)
        self._embedding_dim = embedding_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b = x.shape[0]
        flat = x.reshape(b, -1)
        return torch.nn.functional.normalize(self._linear(flat), p=2, dim=1)


class _VectorBackbone(torch.nn.Module):
    def __init__(self, embedding_dim: int = 2,
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.nn.functional.normalize(x[:, :2], p=2, dim=1)


class _RecordingImageBackbone(torch.nn.Module):
    last_input: torch.Tensor | None = None

    def __init__(self, embedding_dim: int = 2,
                 use_gradient_checkpointing: bool = False) -> None:
        super().__init__()
        self._anchor = torch.nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        type(self).last_input = x.detach().cpu()
        values = x.mean(dim=(2, 3))[:, :2]
        return torch.nn.functional.normalize(values, p=2, dim=1)


class TrainingArtifactTests(unittest.TestCase):
    def test_checkpoint_selection_requires_non_regressing_rank1_and_map(self) -> None:
        baseline = {"rank1": 0.8, "map": 0.7}
        self.assertTrue(_selection_improves({"rank1": 0.81, "map": 0.7}, baseline))
        self.assertTrue(_selection_improves({"rank1": 0.8, "map": 0.71}, baseline))
        self.assertFalse(_selection_improves({"rank1": 0.82, "map": 0.69}, baseline))
        self.assertFalse(_selection_improves({"rank1": 0.79, "map": 0.72}, baseline))
        self.assertFalse(_selection_improves(dict(baseline), baseline))

    def test_training_config_validates_regularization_and_early_stopping(self) -> None:
        with self.assertRaisesRegex(ValueError, "label_smoothing"):
            TrainConfig(label_smoothing=1.0)
        with self.assertRaisesRegex(ValueError, "early_stop_patience"):
            TrainConfig(early_stop_patience=0)
        with self.assertRaisesRegex(ValueError, "backbone_lr_scale"):
            TrainConfig(backbone_lr_scale=0.0)
        with self.assertRaisesRegex(ValueError, "freeze_backbone_epochs"):
            TrainConfig(epochs=1, freeze_backbone_epochs=2)
        with self.assertRaisesRegex(ValueError, "embedding_consistency_weight"):
            TrainConfig(embedding_consistency_weight=-1.0)

    def test_metric_learning_consistency_penalizes_embedding_drift(self) -> None:
        config = TrainConfig(
            embedding_dim=2,
            num_classes=2,
            epochs=1,
            freeze_backbone_epochs=0,
        )
        model = ArcFaceModel(config, backbone_factory=_VectorBackbone)
        images = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
        labels = torch.tensor([0, 1])
        matching_teacher = _VectorBackbone(embedding_dim=2)
        _, _, matching = _metric_learning_loss(
            model,
            model,
            images,
            labels,
            label_smoothing=0.0,
            frozen_backbone=matching_teacher,
            consistency_weight=2.0,
        )
        rotated_teacher = _VectorBackbone(embedding_dim=2)
        rotated_teacher.forward = lambda value: torch.nn.functional.normalize(
            value[:, [1, 0]], p=2, dim=1
        )
        _, _, drifted = _metric_learning_loss(
            model,
            model,
            images,
            labels,
            label_smoothing=0.0,
            frozen_backbone=rotated_teacher,
            consistency_weight=2.0,
        )
        self.assertAlmostEqual(float(matching), 0.0)
        self.assertGreater(float(drifted), float(matching))

    def test_compute_embeddings_normalizes_uint8_dataset_images(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        model = ArcFaceModel(
            TrainConfig(embedding_dim=2, num_classes=2),
            backbone_factory=_RecordingImageBackbone,
        )
        images = torch.full((2, 3, 4, 4), 255, dtype=torch.uint8)
        labels = torch.tensor([0, 1])
        embeddings, observed_labels = compute_embeddings(
            model,
            DataLoader(TensorDataset(images, labels), batch_size=2),
            torch.device("cpu"),
        )
        self.assertEqual(embeddings.shape, (2, 2))
        self.assertEqual(observed_labels.tolist(), [0, 1])
        self.assertEqual(_RecordingImageBackbone.last_input.dtype, torch.float32)
        self.assertAlmostEqual(
            float(_RecordingImageBackbone.last_input[0, 0, 0, 0]),
            (1.0 - 0.485) / 0.229,
            places=5,
        )

    def test_checkpoint_is_reconstructable_with_weights_only_load(self) -> None:
        cfg = TrainConfig(
            model_name="test-vector",
            embedding_dim=2,
            num_classes=2,
            epochs=1,
        )
        model = ArcFaceModel(cfg, backbone_factory=_VectorBackbone)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=None,
            config=cfg,
            label_to_index={"dog-a": 0, "dog-b": 1},
            epoch=1,
            global_step=3,
            selection_metric={"rank1": 1.0},
        )
        with tempfile.NamedTemporaryFile(suffix=".pt") as file:
            torch.save(payload, file.name)
            loaded = torch.load(file.name, weights_only=True)
        self.assertEqual(loaded["schema_version"], "cvi.training_checkpoint.v1")
        self.assertEqual(loaded["architecture"]["model_name"], "test-vector")
        self.assertEqual(loaded["label_to_index"], {"dog-a": 0, "dog-b": 1})

    def test_default_checkpoint_reconstructs_with_matching_backbone(self) -> None:
        from workflows import export_onnx

        cfg = TrainConfig(num_classes=2)
        model = ArcFaceModel(cfg, backbone_factory=_VectorBackbone)
        optimizer = torch.optim.AdamW(model.parameters())
        payload = _checkpoint_payload(
            model=model,
            optimizer=optimizer,
            scaler=None,
            config=cfg,
            label_to_index={"dog-a": 0, "dog-b": 1},
            epoch=1,
            global_step=1,
            selection_metric={"rank1": 1.0},
        )
        with patch.dict(
            export_onnx._BACKBONES, {"dinov2-small": _VectorBackbone}
        ):
            reconstructed = export_onnx.reconstruct_model(payload)
        self.assertEqual(
            set(reconstructed.state_dict()), set(model.state_dict())
        )

    def test_development_retrieval_uses_unseen_identity_labels(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        cfg = TrainConfig(embedding_dim=2, num_classes=2, epochs=1)
        model = ArcFaceModel(cfg, backbone_factory=_VectorBackbone)
        vectors = torch.tensor([
            [1.0, 0.0],
            [0.99, 0.01],
            [0.0, 1.0],
            [0.01, 0.99],
        ])
        labels = torch.tensor([10, 10, 11, 11])
        metrics = _evaluate_development_retrieval(
            model,
            DataLoader(TensorDataset(vectors, labels), batch_size=2),
            torch.device("cpu"),
            None,
            None,
        )
        self.assertEqual(metrics["rank1"], 1.0)
        self.assertEqual(metrics["map"], 1.0)
        self.assertEqual(metrics["identities"], 2.0)

    def test_development_retrieval_rejects_singletons(self) -> None:
        from torch.utils.data import DataLoader, TensorDataset

        cfg = TrainConfig(embedding_dim=2, num_classes=2, epochs=1)
        model = ArcFaceModel(cfg, backbone_factory=_VectorBackbone)
        loader = DataLoader(TensorDataset(
            torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
            torch.tensor([10, 11]),
        ))
        with self.assertRaisesRegex(RuntimeError, "two samples"):
            _evaluate_development_retrieval(
                model, loader, torch.device("cpu"), None, None
            )


class ArcFaceModelTests(unittest.TestCase):
    def test_a4_starts_at_frozen_baseline_and_respects_hard_bound(self) -> None:
        baseline = _DummyBackbone(embedding_dim=64)
        model = AppearanceBoundedResidual(baseline, 64, scale=0.1)
        images = torch.randn(3, 3, 224, 224)
        with torch.no_grad():
            baseline_embeddings = baseline(images)
            initial_embeddings = model(images)
        self.assertTrue(torch.allclose(
            initial_embeddings, baseline_embeddings, atol=1e-7, rtol=0.0
        ))
        self.assertTrue(all(not parameter.requires_grad for parameter in model.baseline.parameters()))
        model.train()
        self.assertFalse(model.baseline.training)
        self.assertTrue(model.adapter.training)

        with torch.no_grad():
            model.adapter[-1].weight.fill_(1.0)
            model.adapter[-1].bias.fill_(1.0)
            adapted_embeddings = model(images)
            raw_residual = model.adapter(baseline_embeddings)
            bounded_residual = raw_residual / raw_residual.norm(
                dim=1, keepdim=True
            ).clamp_min(1.0)
            expected = torch.nn.functional.normalize(
                baseline_embeddings + 0.1 * bounded_residual, p=2, dim=1
            )
        self.assertTrue(torch.all(
            (0.1 * bounded_residual).norm(dim=1) <= 0.100001
        ))
        self.assertTrue(torch.allclose(
            adapted_embeddings, expected, atol=1e-7, rtol=0.0
        ))

    def test_a4_arcface_model_only_trains_adapter_and_head(self) -> None:
        cfg = TrainConfig(
            architecture="appearance_bounded_residual_v4",
            embedding_dim=64,
            num_classes=5,
            border_consistency_weight=0.5,
            baseline_anchor_weight=1.0,
        )
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        self.assertIsInstance(model._backbone, AppearanceBoundedResidual)
        self.assertTrue(all(
            not parameter.requires_grad
            for parameter in model._backbone.baseline.parameters()
        ))
        self.assertTrue(all(
            parameter.requires_grad for parameter in model._backbone.adapter.parameters()
        ))

    def test_a4_losses_are_finite_and_update_the_adapter(self) -> None:
        cfg = TrainConfig(
            architecture="appearance_bounded_residual_v4",
            embedding_dim=64,
            num_classes=5,
            border_consistency_weight=0.5,
            baseline_anchor_weight=1.0,
        )
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        with torch.no_grad():
            torch.nn.init.normal_(model._backbone.adapter[-1].weight, std=0.01)
        images = torch.randn(4, 3, 224, 224)
        challenged = images + 0.1 * torch.randn_like(images)
        labels = torch.tensor([0, 1, 2, 3])
        with patch.object(
            model._backbone.baseline,
            "forward",
            wraps=model._backbone.baseline.forward,
        ) as baseline_forward:
            total, classification, consistency, anchor = _a4_metric_learning_loss(
                model,
                model,
                images,
                challenged,
                labels,
                label_smoothing=0.1,
                consistency_weight=0.5,
                anchor_weight=1.0,
            )
        self.assertEqual(baseline_forward.call_count, 2)
        self.assertTrue(all(torch.isfinite(value) for value in (
            total, classification, consistency, anchor
        )))
        self.assertGreater(float(consistency.detach()), 0.0)
        self.assertGreater(float(anchor.detach()), 0.0)
        total.backward()
        self.assertTrue(any(
            parameter.grad is not None and parameter.grad.abs().sum() > 0
            for parameter in model._backbone.adapter.parameters()
        ))

    def test_forward_training(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model.train()
        logits = model(images, labels)
        self.assertEqual(logits.shape, (4, 5))

    def test_forward_eval(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        model.eval()
        images = torch.randn(4, 3, 224, 224)
        with torch.no_grad():
            emb = model(images)
        self.assertEqual(emb.shape, (4, 64))

    def test_extract_embedding(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(2, 3, 224, 224)
        emb_np = model.extract_embedding(images)
        self.assertEqual(emb_np.shape, (2, 64))

    def test_forward_train_returns_logits_in_eval_mode(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        model.eval()
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        logits = model.forward_train(images, labels)
        embeddings = model.encode(images)
        self.assertEqual(logits.shape, (4, 5))
        self.assertEqual(embeddings.shape, (4, 64))

    def test_export_to_onnx(self) -> None:
        import numpy as np
        import onnx
        import onnxruntime as ort

        import tempfile
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            path = Path(f.name)
            model.export_to_onnx(path)
            self.assertTrue(path.exists())
            exported = onnx.load(path, load_external_data=False)
            self.assertEqual(exported.graph.input[0].name, "images")
            self.assertEqual(exported.graph.output[0].name, "embedding")
            self.assertEqual(
                exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param,
                "batch",
            )
            self.assertEqual(
                exported.graph.output[0].type.tensor_type.shape.dim[0].dim_param,
                "batch",
            )
            self.assertEqual(
                {opset.domain: opset.version for opset in exported.opset_import}[""],
                18,
            )
            self.assertTrue(all(
                initializer.data_location != onnx.TensorProto.EXTERNAL
                and not initializer.external_data
                for initializer in exported.graph.initializer
            ))

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(session.get_inputs()[0].shape, ["batch", 3, 224, 224])
            self.assertEqual(session.get_outputs()[0].shape, ["batch", 64])
            for batch_size in (1, 3):
                embedding = session.run(
                    ["embedding"],
                    {
                        "images": np.random.default_rng(batch_size).standard_normal(
                            (batch_size, 3, 224, 224), dtype=np.float32
                        )
                    },
                )[0]
                self.assertEqual(embedding.shape, (batch_size, 64))
                self.assertTrue(np.isfinite(embedding).all())

    def test_dinov2_compat_subclass(self) -> None:
        from representation_learning.trainer import Dinov2ArcFaceModel
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = Dinov2ArcFaceModel(cfg)
        self.assertIsInstance(model, ArcFaceModel)

    def test_gradient_flow(self) -> None:
        cfg = TrainConfig(embedding_dim=64, num_classes=5)
        model = ArcFaceModel(cfg, backbone_factory=_DummyBackbone)
        images = torch.randn(4, 3, 224, 224)
        labels = torch.tensor([0, 1, 2, 3])
        model.train()
        logits = model(images, labels)
        loss = torch.nn.functional.cross_entropy(logits, labels)
        loss.backward()
        has_grad = any(p.grad is not None and p.grad.abs().sum() > 0
                       for p in model.parameters() if p.requires_grad)
        self.assertTrue(has_grad)


class ConvNeXtOnnxExportTests(unittest.TestCase):
    def test_export_and_reload(self) -> None:
        import tempfile
        import numpy as np
        import onnx
        import onnxruntime as ort

        model = ConvNeXtEmbedding(embedding_dim=768)
        dummy = torch.randn(3, 3, 224, 224)
        with tempfile.NamedTemporaryFile(suffix=".onnx") as f:
            path = Path(f.name)
            model.eval()
            batch = torch.export.Dim("batch")
            torch.onnx.export(
                model, (dummy,), str(path),
                input_names=["images"], output_names=["embedding"],
                dynamo=True,
                dynamic_shapes=({0: batch},),
                opset_version=18,
                external_data=False,
            )
            exported = onnx.load(path, load_external_data=False)
            self.assertEqual(exported.graph.input[0].name, "images")
            self.assertEqual(exported.graph.output[0].name, "embedding")
            self.assertEqual(
                exported.graph.input[0].type.tensor_type.shape.dim[0].dim_param,
                "batch",
            )
            self.assertEqual(
                exported.graph.output[0].type.tensor_type.shape.dim[0].dim_param,
                "batch",
            )
            self.assertEqual(
                {opset.domain: opset.version for opset in exported.opset_import}[""],
                18,
            )
            self.assertTrue(all(
                initializer.data_location != onnx.TensorProto.EXTERNAL
                and not initializer.external_data
                for initializer in exported.graph.initializer
            ))

            session = ort.InferenceSession(
                str(path), providers=["CPUExecutionProvider"]
            )
            self.assertEqual(session.get_inputs()[0].shape, ["batch", 3, 224, 224])
            self.assertEqual(session.get_outputs()[0].shape, ["batch", 768])
            for batch_size in (1, 3):
                images = torch.randn(batch_size, 3, 224, 224).numpy()
                embedding = session.run(["embedding"], {"images": images})[0]
                self.assertEqual(embedding.shape, (batch_size, 768))
                self.assertTrue(np.isfinite(embedding).all())

            from identity_methods.backbones.extractors import OnnxExtractor
            from artifact_contracts.model_contract import (
                OnnxEvidenceModelManifest,
                OnnxPreprocessingContract,
            )
            manifest = OnnxEvidenceModelManifest(
                model_id="convnext-export-test",
                model_sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
                input_name="images",
                input_shape=("batch", 3, 224, 224),
                output_name="embedding",
                output_dim=768,
                preprocessing=OnnxPreprocessingContract(
                    color_mode="RGB",
                    layout="NCHW",
                    dtype="float32",
                    resize="bilinear",
                    scale=1.0 / 255.0,
                    mean=(0.485, 0.456, 0.406),
                    std=(0.229, 0.224, 0.225),
                ),
            )
            ext = OnnxExtractor(path, manifest)
            from PIL import Image
            img = Image.fromarray(
                (dummy[0].permute(1, 2, 0).numpy() * 255).astype(np.uint8)
            )
            emb_after = ext.extract(img)
        self.assertEqual(emb_after.shape, (768,))
        diff = float(abs(np.linalg.norm(emb_after) - 1.0))
        self.assertLess(diff, 1e-4)


if __name__ == "__main__":
    import torch
    unittest.main()
