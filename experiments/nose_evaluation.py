"""Common appearance, nose, and late-fusion closed-set evaluation."""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

from evaluation.retrieval import (
    compute_cosine_score_matrix,
    evaluate_multi_template_closed_set,
)
from identity_methods.nose.dataset import NoseIDDataset
from identity_methods.nose.protocol import NoseIDProtocolFold


def evaluate_noseid_ablation(
    *,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_template_ids: np.ndarray,
    gallery_template_ids: np.ndarray,
    query_capture_ids: np.ndarray,
    gallery_capture_ids: np.ndarray,
    query_appearance: np.ndarray,
    gallery_appearance: np.ndarray,
    query_nose: np.ndarray,
    gallery_nose: np.ndarray,
    query_nose_utility: np.ndarray | None = None,
    nose_base_weight: float = 0.30,
) -> dict[str, dict[str, Any]]:
    query_captures = np.asarray(query_capture_ids)
    gallery_captures = np.asarray(gallery_capture_ids)
    if query_captures.shape != (len(query_identity_ids),) or gallery_captures.shape != (len(gallery_identity_ids),):
        raise ValueError("capture IDs must align with query and gallery templates")
    capture_values = query_captures.tolist() + gallery_captures.tolist()
    for value in capture_values:
        if value is None:
            raise ValueError("capture IDs must be non-null")
        if isinstance(value, float) and not np.isfinite(value):
            raise ValueError("capture IDs must be finite")
        try:
            hash(value)
        except TypeError as exc:
            raise ValueError("capture IDs must be hashable") from exc
    overlap = set(query_captures.tolist()) & set(gallery_captures.tolist())
    if overlap:
        raise ValueError("query and gallery captures must be disjoint")
    appearance_scores = compute_cosine_score_matrix(query_appearance, gallery_appearance)
    nose_scores = compute_cosine_score_matrix(query_nose, gallery_nose)
    if query_nose_utility is None:
        nose_weight = np.full((len(query_identity_ids), 1), nose_base_weight, dtype=np.float64)
    else:
        utility = np.asarray(query_nose_utility, dtype=np.float64)
        if utility.shape != (len(query_identity_ids),) or not np.isfinite(utility).all():
            raise ValueError("query nose utility must be a finite query vector")
        nose_weight = (nose_base_weight * np.clip(utility, 0.0, 1.0))[:, None]
    fused_scores = (1.0 - nose_weight) * appearance_scores + nose_weight * nose_scores
    results = {}
    for name, scores in (
        ("appearance", appearance_scores),
        ("nose", nose_scores),
        ("fused", fused_scores),
    ):
        results[name] = evaluate_multi_template_closed_set(
            scores,
            query_identity_ids,
            gallery_identity_ids,
            self_match_policy="exclude",
            query_template_ids=query_template_ids,
            gallery_template_ids=gallery_template_ids,
            rank_ks=(1, 5),
        )
    return results


def deterministic_nose_quality(quality_vector: np.ndarray) -> np.ndarray:
    quality = np.asarray(quality_vector, dtype=np.float64)
    if quality.ndim != 2 or quality.shape[1] != 14 or not np.isfinite(quality).all():
        raise ValueError("quality vector must be finite [N,14]")
    selected = np.clip(quality[:, [1, 3, 4, 6, 8, 11, 13]], 1e-4, 1.0)
    return np.exp(np.mean(np.log(selected), axis=1))


def _evaluate_scores(
    scores: np.ndarray,
    *,
    query_indices: np.ndarray,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_template_ids: np.ndarray,
    gallery_template_ids: np.ndarray,
) -> dict[str, Any]:
    if len(query_indices) == 0:
        return {"query_count": 0, "status": "NO_ELIGIBLE_QUERIES"}
    return evaluate_multi_template_closed_set(
        scores[query_indices],
        query_identity_ids[query_indices],
        gallery_identity_ids,
        self_match_policy="exclude",
        query_template_ids=query_template_ids[query_indices],
        gallery_template_ids=gallery_template_ids,
        rank_ks=(1, 5),
    )


def evaluate_oracle_representations(
    *,
    query_identity_ids: np.ndarray,
    gallery_identity_ids: np.ndarray,
    query_template_ids: np.ndarray,
    gallery_template_ids: np.ndarray,
    query_capture_ids: np.ndarray,
    gallery_capture_ids: np.ndarray,
    query_appearance: np.ndarray,
    gallery_appearance: np.ndarray,
    query_rgb: np.ndarray,
    gallery_rgb: np.ndarray,
    query_texture: np.ndarray,
    gallery_texture: np.ndarray,
    query_nose: np.ndarray,
    gallery_nose: np.ndarray,
    query_quality_vector: np.ndarray,
    query_native_short_side: np.ndarray,
) -> dict[str, Any]:
    query_ids = np.asarray(query_identity_ids)
    gallery_ids = np.asarray(gallery_identity_ids)
    query_templates = np.asarray(query_template_ids)
    gallery_templates = np.asarray(gallery_template_ids)
    query_captures = np.asarray(query_capture_ids)
    gallery_captures = np.asarray(gallery_capture_ids)
    if set(query_captures.tolist()) & set(gallery_captures.tolist()):
        raise ValueError("query and gallery captures must be disjoint")
    quality = deterministic_nose_quality(query_quality_vector)
    native = np.asarray(query_native_short_side, dtype=np.float64)
    if native.shape != (len(query_ids),) or not np.isfinite(native).all():
        raise ValueError("native nose size must be a finite query vector")
    appearance_scores = compute_cosine_score_matrix(query_appearance, gallery_appearance)
    rgb_scores = compute_cosine_score_matrix(query_rgb, gallery_rgb)
    texture_scores = compute_cosine_score_matrix(query_texture, gallery_texture)
    nose_scores = compute_cosine_score_matrix(query_nose, gallery_nose)
    fixed_scores = 0.70 * appearance_scores + 0.30 * nose_scores
    nose_weight = (0.30 * quality)[:, None]
    quality_scores = (1.0 - nose_weight) * appearance_scores + nose_weight * nose_scores
    all_queries = np.arange(len(query_ids))
    score_sets = {
        "A0": appearance_scores,
        "N0": rgb_scores,
        "NT": texture_scores,
        "N3": nose_scores,
        "F0_FIXED": fixed_scores,
        "F0_QUALITY": quality_scores,
    }
    metrics = {
        name: _evaluate_scores(
            scores,
            query_indices=all_queries,
            query_identity_ids=query_ids,
            gallery_identity_ids=gallery_ids,
            query_template_ids=query_templates,
            gallery_template_ids=gallery_templates,
        )
        for name, scores in score_sets.items()
    }
    subsets = {
        "quality_ge_0_65": quality >= 0.65,
        "quality_lt_0_65": quality < 0.65,
        "native_ge_224": native >= 224.0,
        "native_160_223": (native >= 160.0) & (native < 224.0),
        "native_96_159": (native >= 96.0) & (native < 160.0),
    }
    subset_metrics = {
        subset: {
            name: _evaluate_scores(
                scores,
                query_indices=np.flatnonzero(mask),
                query_identity_ids=query_ids,
                gallery_identity_ids=gallery_ids,
                query_template_ids=query_templates,
                gallery_template_ids=gallery_templates,
            )
            for name, scores in score_sets.items()
        }
        for subset, mask in subsets.items()
    }
    return {
        "metrics": metrics,
        "subsets": subset_metrics,
        "query_count": len(query_ids),
        "identity_count": len(set(query_ids.tolist())),
        "capture_count": len(set(query_captures.tolist()) | set(gallery_captures.tolist())),
        "deterministic_quality": quality.tolist(),
    }


def _preprocess_appearance(
    images: list[Image.Image],
    preprocessor: dict[str, Any],
    device: torch.device,
) -> torch.Tensor:
    arrays: list[np.ndarray] = []
    shortest_edge = preprocessor["size"]["shortest_edge"]
    crop_height = preprocessor["crop_size"]["height"]
    crop_width = preprocessor["crop_size"]["width"]
    for image in images:
        width, height = image.size
        if width <= height:
            resized_width = shortest_edge
            resized_height = int(shortest_edge * height / width)
        else:
            resized_height = shortest_edge
            resized_width = int(shortest_edge * width / height)
        resized = image.resize(
            (resized_width, resized_height), Image.Resampling.BICUBIC
        )
        left = (resized_width - crop_width) // 2
        top = (resized_height - crop_height) // 2
        arrays.append(
            np.asarray(
                resized.crop((left, top, left + crop_width, top + crop_height)),
                dtype=np.uint8,
            )
        )
    tensor = torch.from_numpy(np.stack(arrays).transpose(0, 3, 1, 2)).to(
        device=device, dtype=torch.float32
    )
    tensor *= float(preprocessor["rescale_factor"])
    mean = tensor.new_tensor(preprocessor["image_mean"]).view(1, 3, 1, 1)
    std = tensor.new_tensor(preprocessor["image_std"]).view(1, 3, 1, 1)
    return (tensor - mean) / std


@torch.no_grad()
def extract_oracle_representations(
    model: torch.nn.Module,
    dataset: NoseIDDataset,
    *,
    preprocessor: dict[str, Any],
    device: torch.device,
    batch_size: int = 8,
    num_workers: int = 0,
    include_appearance: bool = True,
) -> dict[str, np.ndarray]:
    if batch_size <= 0:
        raise ValueError("evaluation batch size must be positive")
    model.eval()
    appearance_rows: list[torch.Tensor] = []
    if include_appearance:
        for start in range(0, len(dataset), batch_size):
            images = [
                dataset.load_source_image(index)
                for index in range(start, min(start + batch_size, len(dataset)))
            ]
            pixels = _preprocess_appearance(images, preprocessor, device)
            output = model.dino(pixel_values=pixels)
            embeddings = getattr(output, "pooler_output", None)
            if not isinstance(embeddings, torch.Tensor):
                raise RuntimeError("DINO appearance output is missing pooler_output")
            appearance_rows.append(F.normalize(embeddings.float(), dim=1).cpu())

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
    )
    collected: dict[str, list[torch.Tensor]] = {
        "N0": [],
        "NT": [],
        "N3": [],
        "quality": [],
        "gates": [],
    }
    native: list[torch.Tensor] = []
    for batch in loader:
        rgb = batch["aligned_rgb"].to(device=device, dtype=torch.float32)
        keypoints = batch["aligned_kp"].to(device=device, dtype=torch.float32)
        semantic = F.one_hot(
            batch["semantic_mask"].to(device=device, dtype=torch.long), num_classes=3
        ).permute(0, 3, 1, 2).float()
        invalid = batch["invalid_mask"].to(device=device, dtype=torch.float32)
        source_valid = batch["source_valid_mask"].to(
            device=device, dtype=torch.float32
        )
        native_side = batch["native_short_side"].to(device=device, dtype=torch.float32)
        runtime_quality = torch.stack(
            [
                torch.ones_like(native_side),
                (native_side / 448.0).clamp(0, 1),
                keypoints[:, :, 2].mean(dim=1),
                batch["alignment_rms"].to(device=device, dtype=torch.float32),
            ],
            dim=1,
        )
        output = model(
            rgb,
            keypoints,
            runtime_quality,
            semantic_probability=semantic,
            invalid_probability=invalid,
            source_valid_probability=source_valid,
        )
        for target, source in (
            ("N0", "z_rgb"),
            ("NT", "z_texture"),
            ("N3", "embedding"),
            ("quality", "quality_vector"),
            ("gates", "branch_gates"),
        ):
            collected[target].append(output[source].float().cpu())
        native.append(native_side.cpu())
    result = {
        **{
            name: torch.cat(values).numpy().astype(np.float32, copy=False)
            for name, values in collected.items()
        },
        "native_short_side": torch.cat(native).numpy().astype(np.float32, copy=False),
    }
    if include_appearance:
        result["A0"] = torch.cat(appearance_rows).numpy().astype(
            np.float32, copy=False
        )
    if any(len(value) != len(dataset) for value in result.values()):
        raise RuntimeError("oracle representation extraction count differs")
    return result


def evaluate_dev_folds(
    representations: dict[str, np.ndarray],
    dataset: NoseIDDataset,
    folds: tuple[NoseIDProtocolFold, ...],
) -> dict[str, Any]:
    sample_index = {row.sample_id: index for index, row in enumerate(dataset.rows)}
    fold_reports: list[dict[str, Any]] = []
    for fold in folds:
        gallery_samples = [
            sample
            for template in fold.gallery
            for sample in template.samples
        ]
        query_samples = [
            sample
            for template in fold.queries
            for sample in template.samples
        ]
        gallery_indices = np.asarray(
            [sample_index[sample.sample_id] for sample in gallery_samples], dtype=np.int64
        )
        query_indices = np.asarray(
            [sample_index[sample.sample_id] for sample in query_samples], dtype=np.int64
        )
        report = evaluate_oracle_representations(
            query_identity_ids=np.asarray(
                [sample.registered_dog_id for sample in query_samples]
            ),
            gallery_identity_ids=np.asarray(
                [sample.registered_dog_id for sample in gallery_samples]
            ),
            query_template_ids=np.asarray([sample.sample_id for sample in query_samples]),
            gallery_template_ids=np.asarray(
                [sample.sample_id for sample in gallery_samples]
            ),
            query_capture_ids=np.asarray(
                [template.capture_id for template in fold.queries for _ in template.samples]
            ),
            gallery_capture_ids=np.asarray(
                [template.capture_id for template in fold.gallery for _ in template.samples]
            ),
            query_appearance=representations["A0"][query_indices],
            gallery_appearance=representations["A0"][gallery_indices],
            query_rgb=representations["N0"][query_indices],
            gallery_rgb=representations["N0"][gallery_indices],
            query_texture=representations["NT"][query_indices],
            gallery_texture=representations["NT"][gallery_indices],
            query_nose=representations["N3"][query_indices],
            gallery_nose=representations["N3"][gallery_indices],
            query_quality_vector=representations["quality"][query_indices],
            query_native_short_side=representations["native_short_side"][query_indices],
        )
        report["fold_index"] = fold.fold_index
        fold_reports.append(report)
    aggregate: dict[str, dict[str, float]] = {}
    for representation in ("A0", "N0", "NT", "N3", "F0_FIXED", "F0_QUALITY"):
        count = sum(report["metrics"][representation]["num_queries"] for report in fold_reports)
        aggregate[representation] = {
            metric: sum(
                report["metrics"][representation][metric]
                * report["metrics"][representation]["num_queries"]
                for report in fold_reports
            )
            / count
            for metric in ("Rank-1", "Rank-5", "mAP", "mINP")
        }
        aggregate[representation]["query_count"] = count
    return {"folds": fold_reports, "aggregate": aggregate}


__all__ = [
    "deterministic_nose_quality",
    "evaluate_dev_folds",
    "evaluate_noseid_ablation",
    "evaluate_oracle_representations",
    "extract_oracle_representations",
]
