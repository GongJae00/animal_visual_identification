"""Shared deterministic derivation of public canine semantic receipts."""

from __future__ import annotations

from pathlib import Path

from data.public.public_canine_manifest import (
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    YT_DATASET,
    ArchiveReceiptBinding,
    PublicCanineManifest,
    parse_dogfacenet224,
    parse_mpdd,
    parse_sibetan,
    parse_yt_bb_dog,
)
from data.public.public_canine_semantic_receipt import (
    PublicCanineSemanticReceipt,
    summarize_public_canine_manifest,
)


def derive_public_canine_semantics(
    *,
    dataset_name: str,
    archive_path: Path,
    binding: ArchiveReceiptBinding,
    dogface_classes_train: Path | None = None,
    dogface_classes_test: Path | None = None,
) -> tuple[tuple[PublicCanineManifest, ...], PublicCanineSemanticReceipt]:
    facts: dict[str, int] = {}
    if dataset_name == DOGFACE_DATASET:
        result = parse_dogfacenet224(
            archive_path=archive_path,
            binding=binding,
            classes_train_path=dogface_classes_train,
            classes_test_path=dogface_classes_test,
        )
        manifests = (result.manifest,)
        facts["basename_identity_mismatches"] = result.basename_identity_mismatches
        if result.class_split_receipt is None:
            raise ValueError("protected DogFace receipt requires official class files")
        facts.update(
            official_test_identities=result.class_split_receipt.test_identities,
            official_test_images=result.class_split_receipt.test_lines,
            official_train_identities=result.class_split_receipt.train_identities,
            official_train_images=result.class_split_receipt.train_lines,
        )
    elif dataset_name == MPDD_DATASET:
        manifests = (parse_mpdd(archive_path=archive_path, binding=binding),)
        facts["actual_archive_identities"] = manifests[0].identity_count
    elif dataset_name == SIBETAN_DATASET:
        result = parse_sibetan(archive_path=archive_path, binding=binding)
        manifests = (result.manifest,)
        facts.update(
            clusters=result.cluster_count,
            gt_identities=result.gt_identity_count,
            no_mono_clusters=result.no_mono_cluster_count,
            no_mono_identities=result.no_mono_identity_count,
            no_mono_images=result.no_mono_image_count,
        )
    elif dataset_name == YT_DATASET:
        result = parse_yt_bb_dog(archive_path=archive_path, binding=binding)
        manifests = (result.original, result.random_background)
        facts.update(
            missing_random_background_images=result.missing_random_background_images,
            paired_test_images=result.paired_test_images,
        )
    else:
        raise ValueError("unsupported public canine dataset")
    variants = tuple(
        sorted(
            (
                summary
                for manifest in manifests
                for summary in summarize_public_canine_manifest(manifest)
            ),
            key=lambda item: item.source_variant,
        )
    )
    receipt = PublicCanineSemanticReceipt(
        dataset_name=dataset_name,
        variants=variants,
        audited_facts=tuple(sorted(facts.items())),
    )
    return manifests, receipt
