"""Deterministic dataset/view-balanced P x K full-segment sampler."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass

import torch
from torch.utils.data import Sampler

DatasetView = tuple[str, str]
DEFAULT_GROUP_QUOTAS: dict[DatasetView, int] = {
    ("dogfacenet224", "body"): 9,
    ("yt-bb-dog", "body"): 18,
}


@dataclass(frozen=True, slots=True)
class SampleProvenance:
    index: int
    identity_id: str
    observation_id: str
    dataset_name: str
    view: str


class DatasetViewBalancedPKSampler(Sampler[list[int]]):
    """Draw fixed identity quotas per dataset/view and K unique observations."""

    def __init__(
        self,
        identity_ids: Sequence[str],
        observation_ids: Sequence[str],
        dataset_names: Sequence[str],
        views: Sequence[str],
        *,
        group_quotas: Mapping[DatasetView, int] = DEFAULT_GROUP_QUOTAS,
        samples_per_identity: int = 2,
        seed: int = 0,
    ) -> None:
        lengths = {
            len(identity_ids),
            len(observation_ids),
            len(dataset_names),
            len(views),
        }
        if lengths != {len(identity_ids)} or not identity_ids:
            raise ValueError("sampler provenance arrays must be non-empty and aligned")
        if samples_per_identity != 2:
            raise ValueError("Full128 baseline sampling requires K=2")
        if isinstance(seed, bool) or not isinstance(seed, int):
            raise TypeError("sampler seed must be an integer")
        quotas = {
            _normalize_group(group): quota for group, quota in group_quotas.items()
        }
        if len(quotas) != len(group_quotas):
            raise ValueError(
                "dataset/view quota groups must be unique after normalization"
            )
        if not quotas or any(
            isinstance(quota, bool) or not isinstance(quota, int) or quota <= 0
            for quota in quotas.values()
        ):
            raise ValueError("dataset/view identity quotas must be positive integers")
        if any(group[0] == "petface-dog" for group in quotas):
            raise ValueError(
                "PetFace is blocked and cannot have a Full128 sampler quota"
            )

        records: list[SampleProvenance] = []
        grouped: dict[DatasetView, dict[str, list[int]]] = defaultdict(
            lambda: defaultdict(list)
        )
        identity_groups: dict[str, DatasetView] = {}
        observation_ids_seen: set[str] = set()
        for index, values in enumerate(
            zip(identity_ids, observation_ids, dataset_names, views, strict=True)
        ):
            identity, observation, dataset, view = values
            if any(not isinstance(value, str) or not value.strip() for value in values):
                raise ValueError("sampler provenance values must be non-empty strings")
            group = _normalize_group((dataset, view))
            previous = identity_groups.setdefault(identity, group)
            if previous != group:
                raise ValueError("an identity cannot span dataset/view sampling groups")
            if observation in observation_ids_seen:
                raise ValueError("observation IDs must be globally unique")
            observation_ids_seen.add(observation)
            grouped[group][identity].append(index)
            records.append(SampleProvenance(index, identity, observation, *group))

        if set(grouped) != set(quotas):
            raise ValueError(
                "observed dataset/view groups must exactly match sampler quotas"
            )
        for group, identities in grouped.items():
            if len(identities) < quotas[group]:
                raise ValueError(
                    f"insufficient identities for dataset/view group {group!r}"
                )
            if any(
                len(indices) < samples_per_identity for indices in identities.values()
            ):
                raise ValueError(
                    f"every identity in {group!r} requires K distinct observations"
                )

        self.records = tuple(records)
        self.grouped = {
            group: {
                identity: tuple(indices) for identity, indices in identities.items()
            }
            for group, identities in grouped.items()
        }
        self.group_quotas = dict(sorted(quotas.items()))
        self.samples_per_identity = samples_per_identity
        self.seed = seed
        self.epoch = 0

    @property
    def identities_per_batch(self) -> int:
        return sum(self.group_quotas.values())

    def set_epoch(self, epoch: int) -> None:
        if isinstance(epoch, bool) or not isinstance(epoch, int) or epoch < 0:
            raise ValueError("sampler epoch must be a non-negative integer")
        self.epoch = epoch

    def provenance_for_batch(
        self, batch: Sequence[int]
    ) -> tuple[SampleProvenance, ...]:
        if any(
            isinstance(index, bool) or not isinstance(index, int) for index in batch
        ):
            raise TypeError("batch indices must be integers")
        if any(index < 0 or index >= len(self.records) for index in batch):
            raise ValueError("batch index is outside sampler provenance")
        records = tuple(self.records[index] for index in batch)
        if len({record.index for record in records}) != len(records):
            raise ValueError("batch repeats a sample index")
        return records

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        identity_orders: dict[DatasetView, list[str]] = {}
        for group in self.group_quotas:
            identities = sorted(self.grouped[group])
            order = torch.randperm(len(identities), generator=generator).tolist()
            identity_orders[group] = [identities[index] for index in order]
        for batch_index in range(len(self)):
            batch: list[int] = []
            for group, quota in self.group_quotas.items():
                identities = identity_orders[group]
                selected = [
                    identities[(batch_index * quota + offset) % len(identities)]
                    for offset in range(quota)
                ]
                for identity in selected:
                    candidates = self.grouped[group][identity]
                    order = torch.randperm(
                        len(candidates), generator=generator
                    ).tolist()
                    batch.extend(
                        candidates[index]
                        for index in order[: self.samples_per_identity]
                    )
            yield batch

    def __len__(self) -> int:
        return max(
            math.ceil(len(self.grouped[group]) / quota)
            for group, quota in self.group_quotas.items()
        )


def _normalize_group(group: DatasetView) -> DatasetView:
    if not isinstance(group, tuple) or len(group) != 2:
        raise ValueError("dataset/view group keys must be two-item tuples")
    dataset, view = group
    if any(not isinstance(value, str) or not value.strip() for value in group):
        raise ValueError("dataset/view names must be non-empty strings")
    return dataset.strip().lower(), view.strip().lower()


__all__ = [
    "DEFAULT_GROUP_QUOTAS",
    "DatasetViewBalancedPKSampler",
    "SampleProvenance",
]
