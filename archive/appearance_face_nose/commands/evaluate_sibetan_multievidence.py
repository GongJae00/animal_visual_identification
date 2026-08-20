"""Apply the frozen YT A/F/N policy to immutable SiBeTan K1/K3/K5 panels."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import hashlib
import json
import os
from collections.abc import Sequence
from pathlib import Path

import numpy as np
import torch

from shared.contracts.artifact_manifest import (
    ExactOnnxRuntime,
    NoseEmbeddingManifest,
    UsageLane,
    preprocess_image,
)
from shared.contracts.source_provenance import build_source_provenance
from evaluation.search_metrics.metrics import identity_clustered_bootstrap_ci
from archive.shared_helpers.experiments.sibetan_evidence import validate_evidence_bundle_v2
from archive.appearance_face_nose.experiments.sibetan_multievidence import (
    BRANCHES,
    evaluate_effective_k_panel,
    evaluate_n4_substitution,
    face_reliability,
    frozen_transfer_weights,
    nose_reliability,
)
from archive.appearance_face_nose.experiments.unified_multievidence import _extract_dino_embeddings, _read_bound_rgb
from shared.foundation.protected_io import (
    read_strict_json_document,
    read_strict_json_object,
    write_private_json_bundle,
)
from shared.foundation.provenance import content_sha256
from enrollment.registry.identity_registry import compute_registered_dog_id
from identification.export.appearance import ReceiptBoundDinov2Small

if __package__:
    from archive.shared_helpers.commands.evaluate_external_appearance import (
        _bind_image_locations,
        _build_populations,
        _derive_manifest_records,
        _Dinov2Pooler,
        _normalize_model_output,
        _preprocess_images,
        _source_spec_from_payload,
        _validate_split_documents,
    )
else:
    from evaluate_external_appearance import (
        _bind_image_locations,
        _build_populations,
        _derive_manifest_records,
        _Dinov2Pooler,
        _normalize_model_output,
        _preprocess_images,
        _source_spec_from_payload,
        _validate_split_documents,
    )


REPORT_SCHEMA = "cvi.sibetan_multievidence_evaluation.v2"
BUNDLE_SCHEMA = "cvi.sibetan_multievidence_evaluation_bundle.v2"
N4_REPORT_SCHEMA = "cvi.sibetan_n4_metric_adapter_evaluation.v1"
N4_BUNDLE_SCHEMA = "cvi.sibetan_n4_metric_adapter_evaluation_bundle.v1"
INTERPRETATION = (
    "EXPOSED_SIBETAN_CROSS_SEQUENCE_FROZEN_TRANSFER_DIAGNOSTIC_"
    "NOT_FINAL_OR_BIOMETRIC_VALIDATION"
)


def _sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    return digest.hexdigest()


def _normalize(value: np.ndarray) -> np.ndarray:
    result = np.asarray(value, dtype=np.float32)
    norm = float(np.linalg.norm(result))
    if result.ndim != 1 or not np.isfinite(result).all() or norm <= 1e-8:
        raise ValueError("SiBeTan embedding is non-finite or zero-norm")
    return result / norm


def _extract_external_control(
    images, dino: ReceiptBoundDinov2Small, *, device: str, batch_size: int
) -> list[np.ndarray]:
    """Reproduce the established external Appearance stretch preprocessing."""

    dino._ensure_loaded()
    model = _Dinov2Pooler(dino._backbone).to(torch.device(device)).eval()
    vectors: list[np.ndarray] = []
    for offset in range(0, len(images), batch_size):
        batch = images[offset : offset + batch_size]
        tensor = _preprocess_images(batch, torch.device(device))
        with torch.inference_mode():
            values = _normalize_model_output(model(tensor), len(batch), "external control")
        vectors.extend(values.cpu().numpy().astype(np.float32, copy=False))
    return vectors


def _row_metrics(rows):
    return {
        "query_count": len(rows),
        **{
            metric: float(np.mean([row[metric] for row in rows]))
            for metric in ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
        },
    }


def _validate_n4_runtime_bindings(
    checkpoint, *, runtime_manifest_sha256: str, onnx_sha256: str
) -> None:
    if checkpoint["bindings"]["n3"]["onnx_sha256"] != onnx_sha256:
        raise ValueError("N4 adapter and SiBeTan N3 runtime differ")
    if (
        checkpoint["bindings"]["n3"]["runtime_manifest_payload_sha256"]
        != runtime_manifest_sha256
    ):
        raise ValueError("N4 adapter and SiBeTan N3 preprocessing differ")


def _adapt_nose_embeddings(checkpoint, embeddings):
    if not embeddings:
        return {}
    from archive.nose_metric.experiments.n4_metric_adapter import apply_adapter

    tokens = sorted(embeddings)
    adapted = apply_adapter(
        checkpoint, np.stack([embeddings[token] for token in tokens])
    )
    return dict(zip(tokens, adapted, strict=True))


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--assignment", type=Path, required=True)
    parser.add_argument("--labels", type=Path, required=True)
    parser.add_argument("--source-bundle", type=Path, required=True)
    parser.add_argument("--split-receipt", type=Path, required=True)
    parser.add_argument("--split-receipt-sha256", required=True)
    parser.add_argument("--source-spec", type=Path, required=True)
    parser.add_argument("--evidence-bundle", type=Path, required=True)
    parser.add_argument("--evidence-manifest-sha256", required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--yt-policy-report", type=Path, required=True)
    parser.add_argument("--yt-policy-report-sha256", required=True)
    parser.add_argument("--model-dir", type=Path, required=True)
    parser.add_argument("--weight-intake-bundle", type=Path, required=True)
    parser.add_argument("--preprocessor-intake-bundle", type=Path, required=True)
    parser.add_argument("--frozen-model-sha256", required=True)
    parser.add_argument("--nose-manifest", type=Path, required=True)
    parser.add_argument("--nose-manifest-sha256", required=True)
    parser.add_argument("--nose-onnx", type=Path, required=True)
    parser.add_argument("--nose-onnx-sha256", required=True)
    parser.add_argument("--n4-checkpoint", type=Path)
    parser.add_argument("--n4-checkpoint-sha256")
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--nose-device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--bootstrap-resamples", type=int, default=10_000)
    parser.add_argument("--bootstrap-seed", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.batch_size < 1 or args.bootstrap_resamples < 1 or args.bootstrap_seed < 0:
        parser.error("batch size/resamples must be positive and seed non-negative")
    if (args.n4_checkpoint is None) != (args.n4_checkpoint_sha256 is None):
        parser.error("N4 checkpoint and SHA-256 must be provided together")
    return args


def main(argv: Sequence[str] | None = None) -> None:
    args = _parse_args(argv)
    output = Path(os.path.abspath(os.fspath(args.output)))
    repository = find_repo_root(__file__)
    if output.is_relative_to(repository):
        raise ValueError("SiBeTan evaluation report must remain outside Git")
    if output.exists() or output.is_symlink():
        raise FileExistsError(f"refusing to overwrite SiBeTan report: {output}")

    assignment = read_strict_json_object(args.assignment)
    labels = read_strict_json_object(args.labels)
    source_payload = read_strict_json_object(args.source_bundle)
    receipt = read_strict_json_object(args.split_receipt)
    source, source_by_token = _validate_split_documents(
        assignment,
        labels,
        receipt,
        source_payload,
        expected_receipt_sha256=args.split_receipt_sha256,
    )
    sources = _source_spec_from_payload(read_strict_json_object(args.source_spec))
    populations = tuple(
        population
        for population in _build_populations(assignment)
        if population.key.protocol == "SIBETAN_CROSS_SEQUENCE"
    )
    if [population.key.shot for population in populations] != [1, 3, 5]:
        raise ValueError("protected SiBeTan panel must contain nested K1/K3/K5")
    manifest_records, publisher_provenance = _derive_manifest_records(sources)
    locations = _bind_image_locations(
        populations, source_by_token, manifest_records, sources
    )

    evidence_document = read_strict_json_document(args.evidence_bundle)
    evidence_root = args.evidence_bundle.parent.resolve(strict=True)
    evidence_manifest = validate_evidence_bundle_v2(
        evidence_document.payload, root=evidence_root
    )
    if evidence_document.payload["manifest_sha256"] != args.evidence_manifest_sha256:
        raise ValueError("SiBeTan evidence manifest differs from the external pin")
    evidence_by_path = {row["image_path"]: row for row in evidence_manifest["records"]}
    if len(evidence_by_path) != len(evidence_manifest["records"]):
        raise ValueError("SiBeTan evidence repeats a publisher image path")

    yt_document = read_strict_json_document(args.yt_policy_report)
    if yt_document.payload.get("report_sha256") != args.yt_policy_report_sha256:
        raise ValueError("YT fusion policy report differs from the external pin")
    if yt_document.payload.get("schema_version") != "cvi.yt_masked_multievidence_policy_bundle.v2":
        raise ValueError("SiBeTan quality-aware transfer requires YT masked policy v2")
    transfer_weights = frozen_transfer_weights(yt_document.payload)

    dino = ReceiptBoundDinov2Small(
        model_directory=args.model_dir,
        weight_intake_bundle=args.weight_intake_bundle,
        preprocessor_intake_bundle=args.preprocessor_intake_bundle,
        device=args.device,
        max_batch_size=args.batch_size,
    )
    if dino.model_sha256 != args.frozen_model_sha256:
        raise ValueError("frozen DINOv2 differs from the external pin")
    nose_document = read_strict_json_document(args.nose_manifest)
    if nose_document.canonical_payload_sha256 != args.nose_manifest_sha256:
        raise ValueError("Nose runtime manifest differs from the external pin")
    nose_manifest = NoseEmbeddingManifest.from_dict(nose_document.payload)
    if nose_manifest.license.usage_lane != UsageLane.RESEARCH_ONLY:
        raise ValueError("Nose runtime must remain research-only")
    if _sha(args.nose_onnx) != args.nose_onnx_sha256 or nose_manifest.artifact_sha256 != args.nose_onnx_sha256:
        raise ValueError("Nose ONNX differs from the manifest or external pin")
    nose_runtime = ExactOnnxRuntime(
        args.nose_onnx, nose_manifest, use_cuda=args.nose_device == "cuda"
    )
    n4_checkpoint = None
    if args.n4_checkpoint is not None:
        from archive.nose_metric.experiments.n4_metric_adapter import load_adapter_checkpoint

        n4_checkpoint = load_adapter_checkpoint(
            args.n4_checkpoint,
            expected_file_sha256=args.n4_checkpoint_sha256,
        )
        _validate_n4_runtime_bindings(
            n4_checkpoint,
            runtime_manifest_sha256=nose_document.canonical_payload_sha256,
            onnx_sha256=args.nose_onnx_sha256,
        )

    selected_tokens = sorted(locations)
    row_by_token = {}
    for token in selected_tokens:
        source_sample = source_by_token[token]
        location = locations[token]
        evidence = evidence_by_path.get(location.record.member_path)
        if evidence is None:
            raise ValueError("protected SiBeTan panel source lacks an evidence outcome")
        if evidence["registered_identity_id"] != compute_registered_dog_id(
            source_sample.dataset_identity_id
        ):
            raise ValueError("SiBeTan evidence identity differs from protected labels")
        row_by_token[token] = evidence

    data_root = args.data_root.resolve(strict=True)
    face_root = evidence_root
    nose_root = evidence_root
    appearance_images = [
        _read_bound_rgb(
            data_root, row_by_token[token]["image_path"], row_by_token[token]["image_sha256"]
        )
        for token in selected_tokens
    ]
    appearance_vectors = _extract_dino_embeddings(
        appearance_images, dino, batch_size=args.batch_size
    )
    external_control_vectors = _extract_external_control(
        appearance_images, dino, device=args.device, batch_size=args.batch_size
    )
    embeddings: dict[str, dict[str, np.ndarray]] = {
        BRANCHES[0]: dict(zip(selected_tokens, appearance_vectors, strict=True)),
        BRANCHES[1]: {},
        BRANCHES[2]: {},
    }
    quality_maps: dict[str, dict[str, float]] = {
        BRANCHES[0]: {token: 1.0 for token in selected_tokens},
        BRANCHES[1]: {}, BRANCHES[2]: {},
    }
    face_tokens = [
        token for token in selected_tokens if row_by_token[token]["face"]["state"] == "AVAILABLE"
    ]
    face_images = [
        _read_bound_rgb(
            face_root,
            row_by_token[token]["face"]["crop_path"],
            row_by_token[token]["face"]["crop_sha256"],
        )
        for token in face_tokens
    ]
    face_vectors = _extract_dino_embeddings(face_images, dino, batch_size=args.batch_size)
    embeddings[BRANCHES[1]] = dict(zip(face_tokens, face_vectors, strict=True))
    quality_maps[BRANCHES[1]] = {
        token: face_reliability(
            upstream_overall=float(row_by_token[token]["face"]["upstream_quality"]["overall"]),
            native_short_side=int(row_by_token[token]["face"]["quality"]["native_short_side"]),
        )
        for token in face_tokens
    }
    for token in selected_tokens:
        nose = row_by_token[token]["nose"]
        if nose["state"] != "AVAILABLE":
            continue
        image = _read_bound_rgb(nose_root, nose["crop_path"], nose["crop_sha256"])
        embeddings[BRANCHES[2]][token] = _normalize(
            nose_runtime.run(preprocess_image(image, nose_manifest))[0]
        )
        quality = nose["quality"]
        quality_maps[BRANCHES[2]][token] = nose_reliability(
            detector_confidence=float(nose["localizer_confidence"]),
            frontality=float(nose["frontality"]),
            native_short_side=int(nose["native_short_side"]),
            blur_score=float(quality["blur_score"]),
            contrast_score=float(quality["contrast_score"]),
        )
    adapted_nose_embeddings = None
    if n4_checkpoint is not None:
        adapted_nose_embeddings = _adapt_nose_embeddings(
            n4_checkpoint, embeddings[BRANCHES[2]]
        )

    panel_results = []
    for population in populations:
        gallery_rows = [
            {"sample_token": member.sample_token, "identity_token": member.identity_token}
            for member in population.gallery
        ]
        query_rows = [
            {"sample_token": member.sample_token, "identity_token": member.identity_token}
            for member in population.queries
        ]
        panel_tokens = {
            row["sample_token"] for row in (*gallery_rows, *query_rows)
        }
        panel_embeddings = {
            branch: {
                token: vector for token, vector in branch_embeddings.items()
                if token in panel_tokens
            }
            for branch, branch_embeddings in embeddings.items()
        }
        panel_quality = {
            branch: {
                token: value for token, value in branch_quality.items()
                if token in panel_tokens
            }
            for branch, branch_quality in quality_maps.items()
        }
        result = evaluate_effective_k_panel(
            gallery=gallery_rows,
            queries=query_rows,
            embeddings=panel_embeddings,
            transfer_weights=transfer_weights, quality=panel_quality,
        )
        n4_substitution = None
        if adapted_nose_embeddings is not None:
            n4_substitution = evaluate_n4_substitution(
                gallery=gallery_rows,
                queries=query_rows,
                embeddings=panel_embeddings,
                adapted_nose_embeddings={
                    token: vector
                    for token, vector in adapted_nose_embeddings.items()
                    if token in panel_tokens
                },
                transfer_weights=transfer_weights,
                quality=panel_quality,
                bootstrap_resamples=args.bootstrap_resamples,
                bootstrap_seed=args.bootstrap_seed + population.key.shot * 1_000,
            )
        appearance_rows = {
            row["sample_token"]: row
            for row in result["methods"][BRANCHES[0]]["query_rows"]
        }
        for method_index, outcome in enumerate(result["methods"].values()):
            outcome["identity_clustered_bootstrap_cis"] = (
                {
                    metric: identity_clustered_bootstrap_ci(
                        outcome["query_rows"],
                        metric=metric,
                        resamples=args.bootstrap_resamples,
                        seed=args.bootstrap_seed + population.key.shot * 10 + method_index,
                    )
                    for metric in ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
                }
                if outcome["query_rows"]
                else None
            )
            if outcome["query_rows"] and outcome is not result["methods"][BRANCHES[0]]:
                paired_appearance = [appearance_rows[row["sample_token"]] for row in outcome["query_rows"]]
                outcome["paired_appearance_baseline_metrics"] = _row_metrics(paired_appearance)
                outcome["paired_delta_bootstrap_cis"] = {
                    metric: identity_clustered_bootstrap_ci(
                        [
                            {
                                "bootstrap_cluster_id": row["bootstrap_cluster_id"],
                                "delta": row[metric] - baseline[metric],
                            }
                            for row, baseline in zip(outcome["query_rows"], paired_appearance, strict=True)
                        ],
                        metric="delta",
                        resamples=args.bootstrap_resamples,
                        seed=args.bootstrap_seed + population.key.shot * 100 + method_index,
                    )
                    for metric in ("Rank-1", "Rank-5", "Rank-10", "reciprocal_rank")
                }
            else:
                outcome["paired_appearance_baseline_metrics"] = None
                outcome["paired_delta_bootstrap_cis"] = None
        control_embeddings = {
            BRANCHES[0]: {
                token: vector
                for token, vector in zip(selected_tokens, external_control_vectors, strict=True)
                if token in panel_tokens
            },
            BRANCHES[1]: panel_embeddings[BRANCHES[1]],
            BRANCHES[2]: panel_embeddings[BRANCHES[2]],
        }
        external_control = evaluate_effective_k_panel(
            gallery=gallery_rows,
            queries=query_rows,
            embeddings=control_embeddings,
            transfer_weights=transfer_weights, quality=panel_quality,
        )["methods"][BRANCHES[0]]
        panel_results.append(
            {
                "protocol": population.key.protocol,
                "episode": population.key.episode,
                "shot": population.key.shot,
                "population_sha256": content_sha256(
                    {
                        "gallery": [member.event_token for member in population.gallery],
                        "queries": [member.event_token for member in population.queries],
                    }
                ),
                "gallery_order_sha256": content_sha256(
                    [member.sample_token for member in population.gallery]
                ),
                "query_order_sha256": content_sha256(
                    [member.sample_token for member in population.queries]
                ),
                "external_appearance_control": {
                    "preprocessing": "BILINEAR_STRETCH_224X224",
                    "purpose": "REPRODUCE_ESTABLISHED_PROTECTED_APPEARANCE_BASELINE_ONLY",
                    "metrics": external_control["metrics"],
                },
                **(
                    {"n4_metric_adapter_substitution": n4_substitution}
                    if n4_substitution is not None
                    else {}
                ),
                **result,
            }
        )
    n4_enabled = n4_checkpoint is not None
    report = {
        "schema_version": N4_REPORT_SCHEMA if n4_enabled else REPORT_SCHEMA,
        "status": (
            "PASS_EXPOSED_SIBETAN_N4_SUBSTITUTION_DIAGNOSTIC"
            if n4_enabled
            else "PASS_EXPOSED_SIBETAN_FROZEN_TRANSFER_DIAGNOSTIC"
        ),
        "interpretation": INTERPRETATION,
        **(
            {
                "execution": {
                    "device": args.device,
                    "nose_device": args.nose_device,
                    "batch_size": args.batch_size,
                }
            }
            if n4_enabled
            else {}
        ),
        "protocol": {
            "panel_membership": "IMMUTABLE_PROTECTED_K1_K3_K5",
            "missing_evidence": "MASKED_WITHOUT_SENTINEL_BACKFILL_OR_IDENTITY_FILTERING",
            "fusion_weight_source": "YT_DEV_ONLY_EXTERNAL_SHA256_PIN",
            "sibetan_labels_used_for_policy_selection": False,
            "retrieval": "COSINE_OVER_L2_NORMALIZED_AVAILABLE_BRANCH_PROTOTYPES",
            "fusion": "MASKED_ROW_ZSCORE_THEN_CANDIDATE_RENORMALIZED_FROZEN_WEIGHTED_SUM",
            "branch_effective_k": "AVAILABLE_FIXED_SOURCE_OBSERVATIONS_ONLY",
            "transfer_preprocessing": "RECEIPT_BOUND_SHORTEST_EDGE_CENTER_CROP",
            "external_control_used_in_fusion": False,
            "reliability": "YT_DEV_FROZEN_CONTINUOUS_FACE_AND_NOSE_QUALITY",
            **(
                {
                    "n4_substitution": "N3_EMBEDDING_VECTOR_ONLY",
                    "n4_selection": n4_checkpoint["config"]["selection"],
                    "sibetan_labels_used_for_n4_selection": False,
                    "n4_availability_quality_and_frozen_weights_unchanged": True,
                }
                if n4_enabled
                else {}
            ),
            "bootstrap": {
                "cluster_unit": "protected_identity_token",
                "resamples": args.bootstrap_resamples,
                "base_seed": args.bootstrap_seed,
                "confidence_level": 0.95,
            },
        },
        "transfer_weights": transfer_weights,
        "evidence_state_counts": evidence_manifest["state_counts"],
        "panels": panel_results,
        "input_bindings": {
            "split_receipt_sha256": args.split_receipt_sha256,
            "source_bundle_sha256": source.bundle_sha256,
            "assignment_sha256": content_sha256(assignment),
            "evidence_file_sha256": evidence_document.raw_sha256,
            "evidence_manifest_sha256": args.evidence_manifest_sha256,
            "yt_policy_file_sha256": yt_document.raw_sha256,
            "yt_policy_report_sha256": args.yt_policy_report_sha256,
            "frozen_dinov2_sha256": dino.model_sha256,
            "nose_runtime_manifest_sha256": nose_document.canonical_payload_sha256,
            "nose_onnx_sha256": args.nose_onnx_sha256,
            **(
                {
                    "n4_checkpoint_file_sha256": args.n4_checkpoint_sha256,
                    "n4_checkpoint_payload_sha256": n4_checkpoint[
                        "checkpoint_payload_sha256"
                    ],
                }
                if n4_enabled
                else {}
            ),
            "publisher_archives": publisher_provenance,
            "code_sha256s": {
                row["relative_path"]: row["content_sha256"]
                for row in build_source_provenance(
                    (repository / "archive/appearance_face_nose/commands/evaluate_sibetan_multievidence.py",)
                )["code_source_files"]
            },
        },
    }
    report = json.loads(json.dumps(report, allow_nan=False))
    bundle = {
        "schema_version": N4_BUNDLE_SCHEMA if n4_enabled else BUNDLE_SCHEMA,
        "report_sha256": content_sha256(report),
        "report": report,
    }
    if content_sha256(bundle["report"]) != bundle["report_sha256"]:
        raise RuntimeError("SiBeTan report digest differs")
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": report["status"],
                "output": os.fspath(output),
                "report_sha256": bundle["report_sha256"],
                "panels": [
                    {
                        "shot": panel["shot"],
                        "methods": {
                            method: {
                                "evaluated_query_count": value["evaluated_query_count"],
                                "Rank-1": value["metrics"]["Rank-1"] if value["metrics"] else None,
                            }
                            for method, value in panel["methods"].items()
                        },
                    }
                    for panel in panel_results
                ],
            },
            allow_nan=False,
            sort_keys=True,
        ),
        flush=True,
    )


if __name__ == "__main__":
    main()
