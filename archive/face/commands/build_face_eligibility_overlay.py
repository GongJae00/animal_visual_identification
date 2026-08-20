"""Build a score-blind face-eligibility sidecar for one Full128 route plan."""

from __future__ import annotations

from archive.root import repository_root as find_repo_root
import argparse
import json
import os
from collections.abc import Sequence
from pathlib import Path, PurePosixPath

from data.public_sources.public_canine_manifest import (
    DOGFACE_TEST_MD5,
    DOGFACE_TEST_SHA256,
    DOGFACE_TRAIN_MD5,
    DOGFACE_TRAIN_SHA256,
    _read_published_class_file,
)
from data.source_lock import get_record
from shared.foundation.protected_io import (
    read_strict_json_document,
    write_private_json_bundle,
)
from shared.foundation.protected_publication import admit_new_external_output
from evaluation.splits.face.face_eligibility import (
    DogFaceSplitEvidence,
    build_face_eligibility_overlay,
)

_ROUTE_LIMITS = {
    "maximum_bytes": 2_147_483_648,
    "maximum_nodes": 25_000_000,
    "maximum_keys": 10_000_000,
    "maximum_array_length": 1_000_000,
}


def _parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--route-plan", required=True, type=Path)
    parser.add_argument("--dogface-classes-train", required=True, type=Path)
    parser.add_argument("--dogface-classes-test", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    args = _parse_args(argv)
    output = _new_external_output(args.output)
    route = read_strict_json_document(args.route_plan, **_ROUTE_LIMITS).payload
    train_path = _regular_file(args.dogface_classes_train, "DogFace train classes")
    test_path = _regular_file(args.dogface_classes_test, "DogFace test classes")
    train_values, train_sha256, _ = _read_published_class_file(
        train_path,
        expected_sha256=DOGFACE_TRAIN_SHA256,
        expected_md5=DOGFACE_TRAIN_MD5,
    )
    test_values, test_sha256, _ = _read_published_class_file(
        test_path,
        expected_sha256=DOGFACE_TEST_SHA256,
        expected_md5=DOGFACE_TEST_MD5,
    )
    annotations = _load_ap10k_annotations(route)
    bundle = build_face_eligibility_overlay(
        route,
        dogface_split=DogFaceSplitEvidence(
            train_values=train_values,
            test_values=test_values,
            train_sha256=train_sha256,
            test_sha256=test_sha256,
        ),
        ap10k_annotations_by_sha256=annotations,
    )
    write_private_json_bundle(((output, bundle),))
    print(
        json.dumps(
            {
                "status": "CREATED_FACE_ELIGIBILITY_OVERLAY",
                "overlay_sha256": bundle["overlay_sha256"],
                "observation_count": bundle["census"]["observation_count"],
                "eligible_count": bundle["census"]["eligible_count"],
                "score_inputs_used": False,
                "learned_candidate_used": False,
                "output": str(output),
            },
            sort_keys=True,
        )
    )
    return 0


def _load_ap10k_annotations(
    route: object,
) -> dict[str, dict[str, object]]:
    if not isinstance(route, dict):
        raise TypeError("Full128 route plan must be an object")
    try:
        records = route["plan"]["records"]
    except (KeyError, TypeError) as exc:
        raise ValueError("Full128 route plan structure differs") from exc
    bindings: dict[str, str] = {}
    for row in records:
        if not isinstance(row, dict) or row.get("dataset_name") != "ap10k-dog":
            continue
        artifact = row["route_evidence"]["annotation_artifact"]
        previous = bindings.setdefault(artifact["sha256"], artifact["relative_path"])
        if previous != artifact["relative_path"]:
            raise ValueError("AP-10K annotation hash has ambiguous relative paths")
    root = Path(get_record("ap10k-dog").data_root)
    result: dict[str, dict[str, object]] = {}
    for expected_sha256, relative in sorted(bindings.items()):
        path = _bound_file(root, relative, "AP-10K annotation artifact")
        document = read_strict_json_document(
            path,
            maximum_bytes=16_777_216,
            maximum_nodes=2_000_000,
            maximum_keys=1_000_000,
            maximum_array_length=100_000,
        )
        if document.raw_sha256 != expected_sha256:
            raise ValueError("AP-10K annotation bytes differ from route-plan binding")
        result[expected_sha256] = document.payload
    return result


def _bound_file(root: Path, relative: str, subject: str) -> Path:
    pure = PurePosixPath(relative)
    if (
        pure.is_absolute()
        or relative != pure.as_posix()
        or "\\" in relative
        or any(part in {"", ".", ".."} for part in pure.parts)
    ):
        raise ValueError(f"{subject} relative path is unsafe")
    absolute_root = Path(os.path.abspath(os.fspath(root)))
    candidate = absolute_root.joinpath(*pure.parts)
    resolved = candidate.resolve()
    if (
        candidate.is_symlink()
        or not candidate.is_file()
        or not resolved.is_relative_to(absolute_root.resolve())
    ):
        raise ValueError(f"{subject} must be a bound regular file")
    return candidate


def _regular_file(path: Path, subject: str) -> Path:
    absolute = Path(os.path.abspath(os.fspath(path)))
    if absolute.is_symlink() or not absolute.is_file():
        raise ValueError(f"{subject} must be a regular file")
    return absolute


def _new_external_output(path: Path) -> Path:
    return admit_new_external_output(
        path,
        repository_root=find_repo_root(__file__),
        repository_error="face-eligibility output must remain outside the repository",
        overwrite_error="refusing to overwrite face-eligibility output",
    )


if __name__ == "__main__":
    raise SystemExit(main())
