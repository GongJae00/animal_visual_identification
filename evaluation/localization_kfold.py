"""Identity-free, source-group-safe K-fold protocol for localization datasets."""

from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

from data.types import UnifiedCanidSample
from shared.foundation.protected_io import read_strict_json_document
from shared.foundation.provenance import content_sha256

_DATASETS = frozenset({"ap10k-dog", "dogflw"})
_SHA256 = re.compile(r"[0-9a-f]{64}\Z")
_INTERPRETATION = (
    "IDENTITY_FREE_LOCALIZATION_CROSS_VALIDATION;SOURCE_GROUP_AND_EXACT_IMAGE_CLOSED;"
    "EXPOSED_DIAGNOSTIC_NOT_FINAL_EVALUATION"
)
_BUNDLE_SCHEMA = "evaluation.localization_kfold_manifest_bundle.v1"


@dataclass(frozen=True, slots=True)
class LocalizationKFoldPolicy:
    fold_count: int = 5
    dev_offset: int = 1
    schema_version: str = "evaluation.localization_kfold_policy.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.localization_kfold_policy.v1":
            raise ValueError("unsupported localization K-fold policy schema")
        if (
            isinstance(self.fold_count, bool)
            or not isinstance(self.fold_count, int)
            or self.fold_count < 3
        ):
            raise ValueError("localization fold_count must be at least three")
        if (
            isinstance(self.dev_offset, bool)
            or not isinstance(self.dev_offset, int)
            or not 1 <= self.dev_offset < self.fold_count
        ):
            raise ValueError("localization dev_offset must select a distinct fold")

    @property
    def policy_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalizationKFoldPolicy:
        _exact_keys(payload, set(cls.__dataclass_fields__), "localization K-fold policy")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class LocalizationFoldAssignment:
    sample_id: str
    dataset_name: str
    dataset_version: str
    source_group_token: str
    image_sha256: str
    publisher_split: str
    allocation_unit_token: str
    home_fold: int
    identity_target_mode: str = "NONE"
    schema_version: str = "evaluation.localization_fold_assignment.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.localization_fold_assignment.v1":
            raise ValueError("unsupported localization fold assignment schema")
        for value, name in (
            (self.sample_id, "sample_id"),
            (self.source_group_token, "source_group_token"),
            (self.image_sha256, "image_sha256"),
            (self.allocation_unit_token, "allocation_unit_token"),
        ):
            _require_sha256(value, name)
        if self.dataset_name not in _DATASETS:
            raise ValueError("localization fold dataset differs")
        for value, name in (
            (self.dataset_version, "dataset_version"),
            (self.publisher_split, "publisher_split"),
        ):
            _require_text(value, name)
        if isinstance(self.home_fold, bool) or not isinstance(self.home_fold, int) or self.home_fold < 0:
            raise ValueError("localization home_fold must be nonnegative")
        if self.identity_target_mode != "NONE":
            raise ValueError("localization folds must not carry identity targets")

    def to_dict(self) -> dict[str, Any]:
        return {name: getattr(self, name) for name in self.__dataclass_fields__}

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalizationFoldAssignment:
        _exact_keys(payload, set(cls.__dataclass_fields__), "localization fold assignment")
        return cls(**payload)


@dataclass(frozen=True, slots=True)
class LocalizationKFoldManifest:
    protocol_name: str
    policy: LocalizationKFoldPolicy
    policy_sha256: str
    source_manifest_sha256s: tuple[tuple[str, str], ...]
    sample_set_sha256: str
    dataset_unit_counts: tuple[tuple[str, tuple[int, ...]], ...]
    assignments: tuple[LocalizationFoldAssignment, ...]
    identity_target_mode: str = "NONE"
    score_inputs_used: bool = False
    final_evaluation_permitted: bool = False
    interpretation: str = _INTERPRETATION
    schema_version: str = "evaluation.localization_kfold_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "evaluation.localization_kfold_manifest.v1":
            raise ValueError("unsupported localization K-fold manifest schema")
        _require_text(self.protocol_name, "protocol_name")
        if not isinstance(self.policy, LocalizationKFoldPolicy):
            raise TypeError("policy must be LocalizationKFoldPolicy")
        if self.policy_sha256 != self.policy.policy_sha256:
            raise ValueError("localization K-fold policy hash differs")
        if (
            len(self.source_manifest_sha256s) != len(_DATASETS)
            or self.source_manifest_sha256s
            != tuple(sorted(self.source_manifest_sha256s))
            or {dataset for dataset, _ in self.source_manifest_sha256s} != _DATASETS
        ):
            raise ValueError("localization source manifest bindings differ")
        for _, digest in self.source_manifest_sha256s:
            _require_sha256(digest, "localization source manifest SHA-256")
        _require_sha256(self.sample_set_sha256, "localization sample-set SHA-256")
        if self.identity_target_mode != "NONE":
            raise ValueError("localization manifest must not carry identity targets")
        if self.score_inputs_used is not False or self.final_evaluation_permitted is not False:
            raise ValueError("localization K-fold cannot use scores or permit final evaluation")
        if self.interpretation != _INTERPRETATION:
            raise ValueError("localization K-fold interpretation differs")
        if not self.assignments or self.assignments != tuple(
            sorted(self.assignments, key=lambda item: item.sample_id)
        ):
            raise ValueError("localization fold assignments must be nonempty and sorted")
        if self.sample_set_sha256 != _assignment_sample_set_sha256(self.assignments):
            raise ValueError("localization sample-set digest differs")
        self._validate_closure()

    @property
    def manifest_sha256(self) -> str:
        return content_sha256(self.to_dict())

    def _validate_closure(self) -> None:
        seen: set[str] = set()
        fold_by_unit: dict[str, int] = {}
        fold_by_source_group: dict[tuple[str, str], int] = {}
        fold_by_image: dict[str, int] = {}
        units_by_dataset_fold: dict[str, dict[int, set[str]]] = defaultdict(
            lambda: defaultdict(set)
        )
        for item in self.assignments:
            if item.sample_id in seen:
                raise ValueError("localization fold sample IDs repeat")
            seen.add(item.sample_id)
            if item.home_fold >= self.policy.fold_count:
                raise ValueError("localization home fold exceeds policy")
            for mapping, key, label in (
                (fold_by_unit, item.allocation_unit_token, "allocation unit"),
                (
                    fold_by_source_group,
                    (item.dataset_name, item.source_group_token),
                    "source group",
                ),
                (fold_by_image, item.image_sha256, "exact image"),
            ):
                prior = mapping.setdefault(key, item.home_fold)
                if prior != item.home_fold:
                    raise ValueError(f"localization {label} crosses folds")
            units_by_dataset_fold[item.dataset_name][item.home_fold].add(
                item.allocation_unit_token
            )
        expected = tuple(
            (
                dataset,
                tuple(
                    len(units_by_dataset_fold[dataset][fold])
                    for fold in range(self.policy.fold_count)
                ),
            )
            for dataset in sorted(_DATASETS)
        )
        if any(
            len(counts) != self.policy.fold_count
            or any(
                isinstance(count, bool) or not isinstance(count, int)
                for count in counts
            )
            for _, counts in self.dataset_unit_counts
        ):
            raise TypeError("localization dataset unit counts must be integer fold arrays")
        if self.dataset_unit_counts != expected:
            raise ValueError("localization dataset unit counts differ")
        if any(count <= 0 for _, counts in expected for count in counts):
            raise ValueError("every localization dataset must appear in every fold")

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "protocol_name": self.protocol_name,
            "policy": self.policy.to_dict(),
            "policy_sha256": self.policy_sha256,
            "source_manifest_sha256s": dict(self.source_manifest_sha256s),
            "sample_set_sha256": self.sample_set_sha256,
            "dataset_unit_counts": {
                dataset: list(counts) for dataset, counts in self.dataset_unit_counts
            },
            "assignments": [item.to_dict() for item in self.assignments],
            "identity_target_mode": self.identity_target_mode,
            "score_inputs_used": self.score_inputs_used,
            "final_evaluation_permitted": self.final_evaluation_permitted,
            "interpretation": self.interpretation,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> LocalizationKFoldManifest:
        _exact_keys(payload, set(cls.__dataclass_fields__), "localization K-fold manifest")
        if not isinstance(payload["source_manifest_sha256s"], Mapping) or not isinstance(
            payload["dataset_unit_counts"], Mapping
        ):
            raise TypeError("localization K-fold count and binding fields must be objects")
        if not isinstance(payload["assignments"], list):
            raise TypeError("localization K-fold assignments must be an array")
        return cls(
            protocol_name=payload["protocol_name"],
            policy=LocalizationKFoldPolicy.from_dict(payload["policy"]),
            policy_sha256=payload["policy_sha256"],
            source_manifest_sha256s=tuple(
                sorted(payload["source_manifest_sha256s"].items())
            ),
            sample_set_sha256=payload["sample_set_sha256"],
            dataset_unit_counts=tuple(
                (dataset, tuple(payload["dataset_unit_counts"][dataset]))
                for dataset in sorted(payload["dataset_unit_counts"])
            ),
            assignments=tuple(
                LocalizationFoldAssignment.from_dict(item)
                for item in payload["assignments"]
            ),
            identity_target_mode=payload["identity_target_mode"],
            score_inputs_used=payload["score_inputs_used"],
            final_evaluation_permitted=payload["final_evaluation_permitted"],
            interpretation=payload["interpretation"],
            schema_version=payload["schema_version"],
        )


def build_localization_kfold_manifest(
    samples: Sequence[UnifiedCanidSample],
    *,
    protocol_name: str,
    policy: LocalizationKFoldPolicy,
    source_manifests: Mapping[str, Mapping[str, Any]],
) -> LocalizationKFoldManifest:
    """Assign whole source groups and byte-identical images to rotating folds."""

    if not samples:
        raise ValueError("localization K-fold requires samples")
    _require_text(protocol_name, "protocol_name")
    if set(source_manifests) != _DATASETS:
        raise ValueError("localization source manifests must cover AP-10K and DogFLW")
    if {sample.dataset_name for sample in samples} != _DATASETS:
        raise ValueError("localization samples must cover AP-10K and DogFLW")
    if any(
        sample.registered_identity_id is not None
        or sample.generated_identity_id is not None
        or sample.raw_identity_id is not None
        for sample in samples
    ):
        raise ValueError("identity-free localization samples must not carry identity targets")
    sample_ids = [sample.sample_id for sample in samples]
    if len(sample_ids) != len(set(sample_ids)):
        raise ValueError("localization sample IDs must be unique")
    for dataset in sorted(_DATASETS):
        expected = build_localization_source_manifest(samples, dataset=dataset)
        if source_manifests[dataset] != expected:
            raise ValueError(
                f"{dataset} live adapter projection differs from source manifest"
            )
    source_manifest_sha256s = {
        dataset: content_sha256(source_manifests[dataset])
        for dataset in sorted(_DATASETS)
    }

    groups = _close_allocation_units(samples)
    dataset_by_unit: dict[str, Counter[str]] = {
        unit: Counter(
            {samples[index].dataset_name: 1 for index in indices}
        )
        for unit, indices in groups.items()
    }
    fold_by_unit = _assign_units(
        dataset_by_unit,
        protocol_name=protocol_name,
        evidence_root=content_sha256(
            {
                "policy_sha256": policy.policy_sha256,
                "source_manifest_sha256s": dict(sorted(source_manifest_sha256s.items())),
                "sample_set": sorted(
                    (sample.sample_id, sample.image_sha256) for sample in samples
                ),
            }
        ),
        fold_count=policy.fold_count,
    )
    unit_by_index = {
        index: unit for unit, indices in groups.items() for index in indices
    }
    assignments = tuple(
        sorted(
            (
                LocalizationFoldAssignment(
                    sample_id=sample.sample_id,
                    dataset_name=sample.dataset_name,
                    dataset_version=sample.dataset_version,
                    source_group_token=content_sha256(
                        {
                            "dataset_name": sample.dataset_name,
                            "source_group_id": sample.source_group_id,
                        }
                    ),
                    image_sha256=sample.image_sha256,
                    publisher_split=sample.split_role,
                    allocation_unit_token=unit_by_index[index],
                    home_fold=fold_by_unit[unit_by_index[index]],
                )
                for index, sample in enumerate(samples)
            ),
            key=lambda item: item.sample_id,
        )
    )
    units_by_dataset_fold: dict[str, dict[int, set[str]]] = defaultdict(
        lambda: defaultdict(set)
    )
    for item in assignments:
        units_by_dataset_fold[item.dataset_name][item.home_fold].add(
            item.allocation_unit_token
        )
    counts = tuple(
        (
            dataset,
            tuple(
                len(units_by_dataset_fold[dataset][fold])
                for fold in range(policy.fold_count)
            ),
        )
        for dataset in sorted(_DATASETS)
    )
    return LocalizationKFoldManifest(
        protocol_name=protocol_name,
        policy=policy,
        policy_sha256=policy.policy_sha256,
        source_manifest_sha256s=tuple(sorted(source_manifest_sha256s.items())),
        sample_set_sha256=_assignment_sample_set_sha256(assignments),
        dataset_unit_counts=counts,
        assignments=assignments,
    )


def build_localization_source_manifest(
    samples: Sequence[UnifiedCanidSample], *, dataset: str
) -> dict[str, Any]:
    """Project live adapter samples into an admission-bindable source manifest."""

    if dataset not in _DATASETS:
        raise ValueError("unsupported localization source dataset")
    selected = [sample for sample in samples if sample.dataset_name == dataset]
    if not selected:
        raise ValueError("localization source manifest dataset has no samples")
    versions = {sample.dataset_version for sample in selected}
    if len(versions) != 1:
        raise ValueError("localization source dataset versions differ")
    records = [
        {
            "sample_id": sample.sample_id,
            "source_group_id": sample.source_group_id,
            "image_path": sample.image_path,
            "image_sha256": sample.image_sha256,
            "width": sample.width,
            "height": sample.height,
            "publisher_split": sample.split_role,
        }
        for sample in sorted(selected, key=lambda item: item.sample_id)
    ]
    return {
        "schema_version": "evaluation.localization_source_manifest.v1",
        "dataset_name": dataset,
        "dataset_version": next(iter(versions)),
        "identity_target_mode": "NONE",
        "records": records,
    }


def materialize_localization_fold(
    manifest: LocalizationKFoldManifest, fold_index: int
) -> dict[str, Any]:
    if isinstance(fold_index, bool) or not isinstance(fold_index, int) or not 0 <= fold_index < manifest.policy.fold_count:
        raise ValueError("localization fold_index lies outside policy")

    def stage(home_fold: int) -> str:
        if home_fold == fold_index:
            return "TEST"
        if home_fold == (fold_index + manifest.policy.dev_offset) % manifest.policy.fold_count:
            return "DEV"
        return "TRAIN"

    view = {
        "schema_version": "evaluation.localization_kfold_view.v1",
        "parent_manifest_sha256": manifest.manifest_sha256,
        "fold_index": fold_index,
        "assignments": [
            {
                "sample_id": item.sample_id,
                "dataset_name": item.dataset_name,
                "allocation_unit_token": item.allocation_unit_token,
                "stage": stage(item.home_fold),
                "identity_target_mode": "NONE",
            }
            for item in manifest.assignments
        ],
        "final_evaluation_permitted": False,
        "interpretation": "EXPOSED_LOCALIZATION_CROSS_VALIDATION_VIEW_NOT_FINAL_TEST",
    }
    return {
        "schema_version": "evaluation.localization_kfold_view_bundle.v1",
        "view_sha256": content_sha256(view),
        "view": view,
    }


def localization_kfold_bundle(manifest: LocalizationKFoldManifest) -> dict[str, Any]:
    return {
        "schema_version": _BUNDLE_SCHEMA,
        "manifest_sha256": manifest.manifest_sha256,
        "manifest": manifest.to_dict(),
    }


def read_localization_kfold(path: Any) -> LocalizationKFoldManifest:
    payload = read_strict_json_document(path).payload
    if (
        set(payload) != {"schema_version", "manifest_sha256", "manifest"}
        or payload["schema_version"] != _BUNDLE_SCHEMA
    ):
        raise ValueError("localization K-fold bundle schema differs")
    _require_sha256(payload["manifest_sha256"], "localization manifest SHA-256")
    if not isinstance(payload["manifest"], Mapping) or content_sha256(
        payload["manifest"]
    ) != payload["manifest_sha256"]:
        raise ValueError("localization K-fold bundle digest differs")
    manifest = LocalizationKFoldManifest.from_dict(payload["manifest"])
    if manifest.manifest_sha256 != payload["manifest_sha256"]:
        raise ValueError("localization K-fold manifest digest differs")
    return manifest


class _DisjointSet:
    def __init__(self, size: int) -> None:
        self.parent = list(range(size))

    def find(self, value: int) -> int:
        root = value
        while self.parent[root] != root:
            root = self.parent[root]
        while self.parent[value] != value:
            parent = self.parent[value]
            self.parent[value] = root
            value = parent
        return root

    def union(self, left: int, right: int) -> None:
        left_root, right_root = self.find(left), self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def _close_allocation_units(
    samples: Sequence[UnifiedCanidSample],
) -> dict[str, tuple[int, ...]]:
    disjoint = _DisjointSet(len(samples))
    first_by_group: dict[tuple[str, str], int] = {}
    first_by_image: dict[str, int] = {}
    for index, sample in enumerate(samples):
        _require_sha256(sample.sample_id, "localization sample_id")
        _require_sha256(sample.image_sha256, "localization image SHA-256")
        _require_text(sample.source_group_id, "source_group_id")
        group_key = (sample.dataset_name, sample.source_group_id)
        for mapping, key in ((first_by_group, group_key), (first_by_image, sample.image_sha256)):
            prior = mapping.setdefault(key, index)
            disjoint.union(prior, index)
    grouped: dict[int, list[int]] = defaultdict(list)
    for index in range(len(samples)):
        grouped[disjoint.find(index)].append(index)
    return {
        content_sha256(
            {
                "samples": sorted(samples[index].sample_id for index in indices),
                "images": sorted({samples[index].image_sha256 for index in indices}),
            }
        ): tuple(sorted(indices))
        for indices in grouped.values()
    }


def _assign_units(
    units: Mapping[str, Counter[str]],
    *,
    protocol_name: str,
    evidence_root: str,
    fold_count: int,
) -> dict[str, int]:
    totals: Counter[str] = Counter()
    for counts in units.values():
        totals.update(counts)
    if set(totals) != _DATASETS or any(count < fold_count for count in totals.values()):
        raise ValueError("each localization dataset needs at least one unit per fold")
    targets = {
        dataset: tuple(
            count // fold_count + (fold < count % fold_count)
            for fold in range(fold_count)
        )
        for dataset, count in totals.items()
    }
    observed: dict[str, Counter[int]] = defaultdict(Counter)

    def rank(unit: str) -> str:
        return hashlib.sha256(
            f"LOCALIZATION_KFOLD_UNIT_V1\0{protocol_name}\0{evidence_root}\0{unit}".encode()
        ).hexdigest()

    result: dict[str, int] = {}
    for unit in sorted(units, key=lambda item: (-sum(units[item].values()), rank(item), item)):
        def objective(
            candidate: int, unit_for_assignment: str = unit
        ) -> tuple[int, int, int]:
            error = 0
            overflow = 0
            for dataset, target in targets.items():
                for fold in range(fold_count):
                    value = observed[dataset][fold] + (
                        units[unit_for_assignment][dataset]
                        if fold == candidate
                        else 0
                    )
                    delta = value - target[fold]
                    error += abs(delta)
                    overflow += max(0, delta)
            return error, overflow, candidate

        selected = min(range(fold_count), key=objective)
        result[unit] = selected
        for dataset, count in units[unit].items():
            observed[dataset][selected] += count
    return result


def _assignment_sample_set_sha256(
    assignments: Iterable[LocalizationFoldAssignment],
) -> str:
    return content_sha256(
        [
            [
                item.sample_id,
                item.dataset_name,
                item.dataset_version,
                item.source_group_token,
                item.image_sha256,
                item.publisher_split,
            ]
            for item in assignments
        ]
    )


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise ValueError(f"{name} must be a lowercase SHA-256")


def _require_text(value: object, name: str) -> None:
    if not isinstance(value, str) or not value or len(value.encode("utf-8")) > 2048:
        raise ValueError(f"{name} must be bounded non-empty text")


def _exact_keys(payload: object, expected: set[str], name: str) -> None:
    if not isinstance(payload, Mapping) or set(payload) != expected:
        raise ValueError(f"{name} schema differs")


__all__ = [
    "LocalizationFoldAssignment",
    "LocalizationKFoldManifest",
    "LocalizationKFoldPolicy",
    "build_localization_kfold_manifest",
    "build_localization_source_manifest",
    "localization_kfold_bundle",
    "materialize_localization_fold",
    "read_localization_kfold",
]
