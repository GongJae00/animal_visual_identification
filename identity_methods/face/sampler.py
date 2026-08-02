"""Face ReID sampler — identity-balanced, provenance-aware."""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Iterator, Sequence

import torch
from torch.utils.data import Sampler


class PositiveStrength:
    STRONG = "STRONG"
    MEDIUM = "MEDIUM"
    WEAK = "WEAK"


class FaceReIDSampler(Sampler[list[int]]):
    def __init__(
        self,
        identity_ids: Sequence[str],
        session_ids: Sequence[str],
        *,
        identities_per_batch: int = 16,
        samples_per_identity: int = 4,
        seed: int = 0,
    ) -> None:
        if len(identity_ids) != len(session_ids) or not identity_ids:
            raise ValueError("identity/session arrays must be non-empty and aligned")
        if identities_per_batch != 16 or samples_per_identity != 4:
            raise ValueError("FaceID sampler requires P=16 and K=4")

        grouped: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        for idx, (identity, session) in enumerate(zip(identity_ids, session_ids, strict=True)):
            if not identity or not session:
                raise ValueError("identity and session IDs must be non-empty")
            grouped[identity][session].append(idx)

        if len(grouped) < identities_per_batch:
            raise ValueError(f"need >= {identities_per_batch} identities")

        self.grouped = {k: dict(v) for k, v in grouped.items()}
        self.identities = tuple(sorted(self.grouped))
        self.p = identities_per_batch
        self.k = samples_per_identity
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def _select_samples(self, identity: str, generator: torch.Generator) -> list[int]:
        sessions = self.grouped[identity]
        ordered = list(sessions)
        perm = torch.randperm(len(ordered), generator=generator).tolist()
        ordered = [ordered[i] for i in perm]

        selected: list[int] = []
        for session in ordered:
            candidates = sessions[session]
            if len(candidates) >= 2:
                indices = torch.randperm(len(candidates), generator=generator).tolist()[:2]
                selected.extend(candidates[j] for j in indices)
                if len(selected) >= self.k:
                    break

        if len(selected) < self.k and len(selected) > 0:
            missing = self.k - len(selected)
            for session in ordered:
                for idx in sessions[session]:
                    if idx not in selected:
                        selected.append(idx)
                        missing -= 1
                        if missing == 0:
                            return selected[: self.k]
            return selected[: self.k] if len(selected) >= 2 else selected * (self.k // len(selected) + 1)[: self.k]

        if len(selected) < self.k:
            selected = selected * (self.k // max(len(selected), 1) + 1)
        return selected[: self.k]

    def __iter__(self) -> Iterator[list[int]]:
        generator = torch.Generator().manual_seed(self.seed + self.epoch)
        order = torch.randperm(len(self.identities), generator=generator).tolist()
        shuffled = [self.identities[i] for i in order]

        for start in range(0, len(shuffled), self.p):
            batch_ids = shuffled[start : start + self.p]
            if len(batch_ids) < self.p:
                batch_ids = shuffled[: self.p]
            yield [
                idx
                for identity in batch_ids
                for idx in self._select_samples(identity, generator)
            ]

    def __len__(self) -> int:
        return max(1, math.ceil(len(self.identities) / self.p))


__all__ = ["FaceReIDSampler", "PositiveStrength"]
