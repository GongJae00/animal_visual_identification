from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np


class AbstractIdentityIndex(ABC):
    @abstractmethod
    def enroll(self, embedding: np.ndarray | dict[str, np.ndarray], registered_dog_id: str,
               metadata: dict | None = None,
               idempotency_key: str | None = None,
               content_sha256: str | None = None) -> int:
        ...

    @abstractmethod
    def search(self, query: np.ndarray | dict[str, np.ndarray], top_k: int = 5
               ) -> list[tuple[int, float, dict]]:
        ...

    @abstractmethod
    def search_with_evidence(self, query: np.ndarray | dict[str, np.ndarray], top_k: int = 5,
                             slices: list[tuple[int, int, str]] | None = None
                             ) -> list[dict[str, Any]]:
        ...

    @abstractmethod
    def remove(self, index: int) -> None:
        ...

    @property
    @abstractmethod
    def size(self) -> int:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    @abstractmethod
    def save(self) -> None:
        ...
