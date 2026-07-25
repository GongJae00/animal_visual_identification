"""Assemble a PublicSplitSourceBundle from audited public canine ZIP archives."""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from pathlib import Path

from cvi.provenance import content_sha256
from cvi.public_canine_manifest import (
    ArchiveReceiptBinding,
    DOGFACE_DATASET,
    MPDD_DATASET,
    SIBETAN_DATASET,
    YT_DATASET,
    parse_dogfacenet224,
    parse_mpdd,
    parse_sibetan,
    parse_yt_bb_dog,
)
from cvi.public_canine_semantic_receipt import (
    PublicCanineSemanticReceipt,
    summarize_public_canine_manifest,
)
from cvi.protected_public_split import (
    PublicSplitSample,
    PublicSplitSourceBundle,
)
from cvi.source_provenance import build_offline_tool_provenance


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _sample_token(source_sample_id: str) -> str:
    return _sha256("sample\x00" + source_sample_id)


def _identity_token(dataset_identity_id: str) -> str:
    return _sha256("identity\x00" + dataset_identity_id)


def _sequence_token(sequence_id: str | None, identity_token: str) -> str:
    if sequence_id is not None:
        return _sha256("sequence\x00" + sequence_id)
    return identity_token


def _extract_raw_frame_index(record) -> int:
    m = re.search(r"frame:(\d+)", record.source_sample_id)
    if m:
        return int(m.group(1))
    m = re.search(r"image:(\d+)\.(\d+)", record.source_sample_id)
    if m:
        return int(m.group(2))
    m = re.search(r"_(\d+)\.jpg$", record.member_path)
    if m:
        return int(m.group(1))
    m = re.search(r"clip:(\d+):frame:(\d+)", record.source_sample_id)
    if m:
        return int(m.group(1)) * 1000 + int(m.group(2))
    return 0


def _read_archive_receipt_bundle(path: Path) -> dict:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "receipt" not in payload:
        raise ValueError(f"archive receipt bundle expected: {path}")
    return payload


def _read_semantic_receipt_bundle(path: Path) -> PublicCanineSemanticReceipt:
    payload = json.loads(path.read_text())
    if not isinstance(payload, dict) or "receipt" not in payload:
        raise ValueError(f"semantic receipt bundle expected: {path}")
    r = payload["receipt"]
    if isinstance(r, dict) and "dataset_name" in r:
        from cvi.public_canine_semantic_receipt import PublicCanineVariantSummary
        variants = tuple(
            PublicCanineVariantSummary(
                dataset_name=r["dataset_name"],
                dataset_version=v["dataset_version"],
                source_variant=v["source_variant"],
                source_archive_sha256=v["source_archive_sha256"],
                source_archive_receipt_sha256=v["source_archive_receipt_sha256"],
                image_count=v["image_count"],
                identity_count=v["identity_count"],
                record_manifest_sha256=v["record_manifest_sha256"],
                identity_semantics_counts=tuple(tuple(p) for p in v["identity_semantics_counts"]),
                region_counts=tuple(tuple(p) for p in v["region_counts"]),
                split_image_counts=tuple(tuple(p) for p in v["split_image_counts"]),
                split_identity_counts=tuple(tuple(p) for p in v["split_identity_counts"]),
                verified_camera_token_count=v["verified_camera_token_count"],
            )
            for v in r["variants"]
        )
        return PublicCanineSemanticReceipt(
            dataset_name=r["dataset_name"],
            variants=variants,
            audited_facts=tuple(tuple(p) for p in r.get("audited_facts", [])),
        )
    raise ValueError(f"unexpected semantic receipt structure: {path}")


def _records_to_samples(
    records, dataset_name: str, *, use_identity_as_sequence: bool = False
) -> list[PublicSplitSample]:
    samples: list[PublicSplitSample] = []
    for record in records:
        token = _sample_token(record.source_sample_id)
        id_token = _identity_token(record.dataset_identity_id)
        if use_identity_as_sequence:
            seq_token = id_token
        else:
            seq_token = _sequence_token(record.sequence_id, id_token)
        frame_index = _extract_raw_frame_index(record)
        region = (
            record.region.value
            if hasattr(record.region, "value")
            else str(record.region)
        )
        samples.append(
            PublicSplitSample(
                sample_token=token,
                identity_token=id_token,
                sequence_token=seq_token,
                source_sample_id=record.source_sample_id,
                dataset_identity_id=record.dataset_identity_id,
                dataset_name=dataset_name,
                source_variant=record.source_variant,
                original_split=record.original_split,
                raw_frame_index=frame_index,
                paired_source_sample_id=record.paired_source_sample_id,
                in_no_mono_subset=record.in_no_mono_subset,
                region=region,
            )
        )
    return samples


def _load_and_verify(
    archive_path: Path,
    archive_receipt_path: Path,
    dataset_name: str,
    semantic_receipt: PublicCanineSemanticReceipt,
    parse_fn,
    **parse_kwargs,
):
    if semantic_receipt.dataset_name != dataset_name:
        raise ValueError(
            f"semantic receipt dataset mismatch: "
            f"{semantic_receipt.dataset_name} != {dataset_name}"
        )

    bundle = _read_archive_receipt_bundle(archive_receipt_path)
    binding = ArchiveReceiptBinding(
        dataset_name=dataset_name,
        archive_sha256=bundle["receipt"]["archive_sha256"],
        archive_receipt_sha256=bundle["receipt_sha256"],
    )

    result = parse_fn(archive_path=archive_path, binding=binding, **parse_kwargs)
    return result


def _verify_manifest(
    manifest, semantic_receipt: PublicCanineSemanticReceipt
) -> None:
    summaries = summarize_public_canine_manifest(manifest)
    for summary in summaries:
        match = [
            v
            for v in semantic_receipt.variants
            if v.source_variant == summary.source_variant
        ]
        if not match:
            raise ValueError(
                f"variant {summary.source_variant} not in semantic receipt"
            )
        expected = match[0]
        if summary.record_manifest_sha256 != expected.record_manifest_sha256:
            raise ValueError(
                f"record manifest SHA-256 mismatch for "
                f"{manifest.dataset_name}/{summary.source_variant}: "
                f"got {summary.record_manifest_sha256}, "
                f"expected {expected.record_manifest_sha256}"
            )
        if summary.image_count != expected.image_count:
            raise ValueError(
                f"image count mismatch for "
                f"{manifest.dataset_name}/{summary.source_variant}: "
                f"got {summary.image_count}, expected {expected.image_count}"
            )


def _safe_read_json(path: Path) -> dict:
    if not path.exists():
        return {"schema_version": "cvi.empty.v1", "status": "PLACEHOLDER"}
    payload = json.loads(path.read_text())
    if isinstance(payload, dict):
        return payload
    return {"schema_version": "cvi.wrapped.v1", "payload": payload}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, type=Path)
    args = parser.parse_args()

    config = json.loads(args.config.read_text())

    evidence_cfg = config.get("evidence", {})
    out_path = Path(config["output"])

    def _r(key: str) -> Path:
        return Path(evidence_cfg[key])

    semantic_receipt_dir = _r("semantic_receipt_dir")
    image_content_receipt_dir = _r("image_content_receipt_dir")

    all_samples: list[PublicSplitSample] = []

    # --- YT-BB-Dog ---
    yt_cfg = config["yt_bb_dog"]
    yt_sem = _read_semantic_receipt_bundle(
        semantic_receipt_dir / "yt-bb-dog-semantic-receipt.json"
    )
    yt_result = _load_and_verify(
        Path(yt_cfg["archive"]),
        Path(yt_cfg["archive_receipt"]),
        YT_DATASET,
        yt_sem,
        parse_yt_bb_dog,
    )
    _verify_manifest(yt_result.original, yt_sem)
    all_samples.extend(_records_to_samples(yt_result.original.records, YT_DATASET))
    _verify_manifest(yt_result.random_background, yt_sem)
    all_samples.extend(
        _records_to_samples(yt_result.random_background.records, YT_DATASET)
    )

    # --- DogFaceNet224 ---
    df_cfg = config["dogfacenet224"]
    df_sem = _read_semantic_receipt_bundle(
        semantic_receipt_dir / "dogfacenet-semantic-receipt.json"
    )
    df_result = _load_and_verify(
        Path(df_cfg["archive"]),
        Path(df_cfg["archive_receipt"]),
        DOGFACE_DATASET,
        df_sem,
        parse_dogfacenet224,
        classes_train_path=(
            Path(df_cfg["classes_train"]) if "classes_train" in df_cfg else None
        ),
        classes_test_path=(
            Path(df_cfg["classes_test"]) if "classes_test" in df_cfg else None
        ),
    )
    _verify_manifest(df_result.manifest, df_sem)
    all_samples.extend(
        _records_to_samples(
            df_result.manifest.records, DOGFACE_DATASET,
            use_identity_as_sequence=True,
        )
    )

    # --- MPDD ---
    mpdd_cfg = config["mpdd"]
    mpdd_sem = _read_semantic_receipt_bundle(
        semantic_receipt_dir / "mpdd-semantic-receipt.json"
    )
    mpdd_manifest = _load_and_verify(
        Path(mpdd_cfg["archive"]),
        Path(mpdd_cfg["archive_receipt"]),
        MPDD_DATASET,
        mpdd_sem,
        parse_mpdd,
    )
    _verify_manifest(mpdd_manifest, mpdd_sem)
    all_samples.extend(
        _records_to_samples(
            mpdd_manifest.records, MPDD_DATASET,
            use_identity_as_sequence=True,
        )
    )

    # --- Sibetan ---
    sib_cfg = config["sibetan"]
    sib_sem = _read_semantic_receipt_bundle(
        semantic_receipt_dir / "sibetan-semantic-receipt.json"
    )
    sib_result = _load_and_verify(
        Path(sib_cfg["archive"]),
        Path(sib_cfg["archive_receipt"]),
        SIBETAN_DATASET,
        sib_sem,
        parse_sibetan,
    )
    _verify_manifest(sib_result.manifest, sib_sem)
    all_samples.extend(
        _records_to_samples(sib_result.manifest.records, SIBETAN_DATASET)
    )

    # --- Evidence bindings ---
    binding_names = (
        "exact_duplicate_graph_sha256",
        "geometric_verifier_sha256",
        "image_content_receipts_sha256",
        "pdq_candidates_sha256",
        "phash_candidates_sha256",
        "review_adjudication_sha256",
        "semantic_receipts_sha256",
    )

    binding_files: list[dict] = []
    for name in binding_names:
        field_name = name.removesuffix("_sha256")
        file_path = evidence_cfg.get(field_name)
        if file_path:
            binding_files.append(_safe_read_json(Path(file_path)))
        else:
            binding_files.append({"schema_version": "cvi.empty.v1"})

    binding_tuples = tuple(
        sorted(
            (name, content_sha256(doc))
            for name, doc in zip(binding_names, binding_files)
        )
    )

    source = PublicSplitSourceBundle(
        evidence_bindings=binding_tuples,
        samples=tuple(all_samples),
    )

    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(
        json.dumps(source.to_dict(), ensure_ascii=False, sort_keys=True, indent=2)
        + "\n"
    )

    print(
        json.dumps(
            {
                "status": "CREATED",
                "dataset": "public_canine_combined",
                "sample_count": len(all_samples),
                "identity_count": len(
                    {s.identity_token for s in all_samples}
                ),
                "bundle_sha256": source.bundle_sha256,
                "output": str(out_path),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
