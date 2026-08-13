from __future__ import annotations

import hashlib

import pytest

from contracts.animal_parsing_runtime import (
    LEGACY_BUNDLE_SCHEMA,
    LEGACY_MANIFEST_SCHEMA,
    QUALIFICATION,
    AnimalParsingRuntimeManifest,
    ParsingEvaluationBinding,
    animal_parsing_runtime_bundle,
)
from contracts.model_file_binding import ModelFileBinding
from foundation.provenance import content_sha256
from parsing.full_segment.animal_parsing import (
    PARSING_ONTOLOGY,
    PARSING_ONTOLOGY_DESCRIPTION,
    AnimalParsingPolicy,
)


def _manifest() -> AnimalParsingRuntimeManifest:
    policy = AnimalParsingPolicy()
    source_sha = hashlib.sha256(b"source").hexdigest()
    report_sha = hashlib.sha256(b"report").hexdigest()
    return AnimalParsingRuntimeManifest(
        parser_family="fixture-parser",
        qualification=QUALIFICATION,
        ontology=PARSING_ONTOLOGY,
        ontology_description=PARSING_ONTOLOGY_DESCRIPTION,
        supported_classes=tuple(sorted(policy.class_names)),
        policy=policy.to_dict(),
        policy_sha256=policy.policy_sha256,
        foreground_model_manifest_sha256="1" * 64,
        foreground_model_bundle_raw_sha256="2" * 64,
        instance_model_manifest_sha256="3" * 64,
        instance_model_bundle_raw_sha256="4" * 64,
        inference_batching={
            "job_batch_size": 4,
            "instance_batch_size": 4,
            "foreground_batch_size": 4,
            "job_ordering": "SOURCE_SHA256_ASC",
            "publication_workers": 4,
            "shape_policy": "EXACT_PREPROCESSED_SHAPE_BUCKETS",
            "oom_policy": "FAIL_CLOSED_NO_RETRY",
        },
        frozen_cache={
            "array_encoding": "BASE64_ZLIB_C_ORDER",
            "zlib_level": 1,
            "retained_arrays": [
                "instance_probability",
                "foreground_probability",
                "ownership_probability",
                "hard_mask",
            ],
        },
        runtime_libraries={
            "numpy": "2.5.1",
            "pillow": "12.3.0",
            "torch": "2.11.0+cu128",
            "torchvision": "0.26.0+cu128",
            "transformers": "5.14.1",
        },
        source_files=(ModelFileBinding("parsing/a.py", 6, source_sha),),
        evaluation_reports=(
            ParsingEvaluationBinding(
                "evaluation",
                "cvi.test.v1",
                "TEST_ONLY",
                6,
                report_sha,
                report_sha,
            ),
        ),
    )


def test_animal_parsing_runtime_manifest_round_trips_exactly() -> None:
    manifest = _manifest()
    restored = AnimalParsingRuntimeManifest.from_dict(manifest.to_dict())
    assert restored == manifest
    bundle = animal_parsing_runtime_bundle(restored)
    assert bundle["manifest_sha256"] == content_sha256(bundle["manifest"])


def test_animal_parsing_runtime_manifest_reads_historical_source_paths() -> None:
    payload = _manifest().to_dict()
    payload["source_files"][0]["relative_path"] = "localization/a.py"

    restored = AnimalParsingRuntimeManifest.from_dict(payload)

    assert restored.source_files[0].relative_path == "localization/a.py"


def test_animal_parsing_runtime_manifest_rejects_policy_tampering() -> None:
    payload = _manifest().to_dict()
    payload["policy"]["minimum_mask_pixels"] += 1
    with pytest.raises(ValueError, match="policy digest"):
        AnimalParsingRuntimeManifest.from_dict(payload)


def test_animal_parsing_runtime_manifest_accepts_bound_v5_policy() -> None:
    manifest = _manifest()
    legacy = AnimalParsingPolicy(
        class_names=("dog", "cat"),
        schema_version="cvi.animal_parsing_policy.v5",
    )
    payload = manifest.to_dict()
    payload["supported_classes"] = ["cat", "dog"]
    payload["policy"] = legacy.to_dict()
    payload["policy_sha256"] = legacy.policy_sha256
    assert AnimalParsingRuntimeManifest.from_dict(payload).policy == legacy.to_dict()


def test_animal_parsing_runtime_manifest_accepts_persisted_v4_policy() -> None:
    manifest = _manifest()
    persisted = AnimalParsingPolicy(
        class_names=("dog", "cat"),
        schema_version="cvi.animal_parsing_policy.v4",
    )
    payload = manifest.to_dict()
    payload["supported_classes"] = ["cat", "dog"]
    payload["policy"] = persisted.to_dict()
    payload["policy_sha256"] = persisted.policy_sha256

    restored = AnimalParsingRuntimeManifest.from_dict(payload)

    assert restored.policy == persisted.to_dict()
    assert restored.policy_sha256 == persisted.policy_sha256


def test_legacy_runtime_manifest_round_trips_without_v2_fields() -> None:
    payload = _manifest().to_dict()
    payload["schema_version"] = LEGACY_MANIFEST_SCHEMA
    for field in ("inference_batching", "frozen_cache", "runtime_libraries"):
        del payload[field]

    restored = AnimalParsingRuntimeManifest.from_dict(payload)

    assert restored.to_dict() == payload
    assert animal_parsing_runtime_bundle(restored) == {
        "schema_version": LEGACY_BUNDLE_SCHEMA,
        "manifest_sha256": content_sha256(payload),
        "manifest": payload,
    }


def test_animal_parsing_runtime_manifest_rejects_unbound_policy_classes() -> None:
    payload = _manifest().to_dict()
    payload["supported_classes"] = ["cat", "dog"]
    with pytest.raises(ValueError, match="classes differ"):
        AnimalParsingRuntimeManifest.from_dict(payload)


def test_animal_parsing_runtime_manifest_rejects_v6_non_dog_policy() -> None:
    payload = _manifest().to_dict()
    payload["supported_classes"] = ["cat", "dog"]
    payload["policy"]["class_names"] = ["dog", "cat"]
    payload["policy_sha256"] = content_sha256(payload["policy"])
    with pytest.raises(ValueError, match="classes differ"):
        AnimalParsingRuntimeManifest.from_dict(payload)
