from __future__ import annotations

import hashlib

import pytest

from artifact_contracts.animal_parsing_runtime import (
    QUALIFICATION,
    AnimalParsingRuntimeManifest,
    ParsingEvaluationBinding,
    animal_parsing_runtime_bundle,
)
from artifact_contracts.model_file_binding import ModelFileBinding
from foundation.provenance import content_sha256
from localization.animal_parsing import (
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
        supported_classes=("cat", "dog"),
        policy=policy.to_dict(),
        policy_sha256=policy.policy_sha256,
        foreground_model_manifest_sha256="1" * 64,
        foreground_model_bundle_raw_sha256="2" * 64,
        instance_model_manifest_sha256="3" * 64,
        instance_model_bundle_raw_sha256="4" * 64,
        source_files=(ModelFileBinding("localization/a.py", 6, source_sha),),
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


def test_animal_parsing_runtime_manifest_rejects_policy_tampering() -> None:
    payload = _manifest().to_dict()
    payload["policy"]["minimum_mask_pixels"] += 1
    with pytest.raises(ValueError, match="policy digest"):
        AnimalParsingRuntimeManifest.from_dict(payload)
