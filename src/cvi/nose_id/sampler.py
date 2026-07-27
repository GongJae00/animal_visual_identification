"""Cross-session P x K batch construction with hard-neighbor insertion."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class CrossSessionPKBatchSampler(Sampler[list[int]]):
    def __init__(
        self,
        identity_ids: Sequence[str],
        session_ids: Sequence[str],
        *,
        identities_per_batch: int = 16,
        samples_per_identity: int = 4,
        hard_neighbors: dict[str, Sequence[str]] | None = None,
        seed: int = 0,
    ) -> None:
        if len(identity_ids) != len(session_ids) or not identity_ids:
            raise ValueError("identity/session arrays must be non-empty and aligned")
        if identities_per_batch != 16 or samples_per_identity != 4:
            raise ValueError("NoseID-v1 sampler requires P=16 and K=4")
        self.identities_per_batch = identities_per_batch
        self.samples_per_identity = samples_per_identity
        self.seed = seed
        self.epoch = 0
        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for index, (identity, session) in enumerate(zip(identity_ids, session_ids, strict=True)):
            if not identity or not session:
                raise ValueError("identity and session IDs must be non-empty")
            grouped[identity][session].append(index)
        if len(grouped) < identities_per_batch:
            raise ValueError("not enough identities for one P x K batch")
        if any(
            len(sessions) < 2
            or sum(len(values) for values in sessions.values()) < samples_per_identity
            or sorted((len(values) for values in sessions.values()), reverse=True)[1] < 2
            for sessions in grouped.values()
        ):
            raise ValueError("every identity needs two sessions with two samples each")
        self.grouped = {identity: dict(sessions) for identity, sessions in grouped.items()}
        self.identities = tuple(sorted(self.grouped))
        self.hard_neighbors = {
            identity: tuple(neighbor for neighbor in neighbors if neighbor in self.grouped and neighbor != identity)
            for identity, neighbors in (hard_neighbors or {}).items()
        }

    def set_epoch(self, epoch: int) -> None:
        if epoch < 0:
            raise ValueError("epoch must be non-negative")
        self.epoch = epoch

    def _select_samples(self, identity: str, generator: torch.Generator) -> list[int]:
        sessions = self.grouped[identity]
        session_order = list(sessions)
        permutation = torch.randperm(len(session_order), generator=generator).tolist()
        session_order = [session_order[index] for index in permutation]
        selected_sessions = [
            session for session in session_order if len(sessions[session]) >= 2
        ][:2]
        selected: list[int] = []
        for session in selected_sessions:
            candidates = sessions[session]
            order = torch.randperm(len(candidates), generator=generator).tolist()
            selected.extend(candidates[index] for index in order[:2])
        if len(selected) != self.samples_per_identity:
            raise RuntimeError("unable to construct K distinct samples")
        return selected

    def _identity_batch(self, start: int, generator: torch.Generator) -> list[str]:
        anchor_count = self.identities_per_batch // 2
        anchors = [self.identities[(start + offset) % len(self.identities)] for offset in range(anchor_count)]
        chosen = list(anchors)
        for anchor in anchors:
            for neighbor in self.hard_neighbors.get(anchor, ()):
                if neighbor not in chosen:
                    chosen.append(neighbor)
                    break
        for identity in self.identities:
            if len(chosen) == self.identities_per_batch:
                break
            if identity not in chosen:
                chosen.append(identity)
        if len(chosen) != self.identities_per_batch:
            raise RuntimeError("unable to fill identity batch")
        return chosen

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.identities), generator=generator).tolist()
        shuffled = tuple(self.identities[index] for index in order)
        original = self.identities
        self.identities = shuffled
        try:
            for start in range(0, len(self.identities), self.identities_per_batch // 2):
                identities = self._identity_batch(start, generator)
                yield [index for identity in identities for index in self._select_samples(identity, generator)]
        finally:
            self.identities = original

    def __len__(self) -> int:
        return math.ceil(len(self.identities) / (self.identities_per_batch // 2))


__all__ = ["CrossSessionPKBatchSampler"]
