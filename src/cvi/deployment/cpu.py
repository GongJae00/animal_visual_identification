"""CPU deployment path for edge devices (Raspberry Pi, etc.).

Uses ONNX Runtime CPU provider and FAISS CPU IndexFlatIP.
Optimized for low-memory environments with optional quantization.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np


class CVIDeploymentCPU:
    def __init__(self, config: dict[str, Any]):
        self._config = config
        self._onnx_path = Path(config.get("onnx_path", ""))
        self._index_path = Path(config.get("index_path", "cvi_cpu.idx"))
        self._meta_path = Path(config.get("metadata_path", "cvi_cpu.json"))

        import faiss
        if self._index_path.exists():
            self._index = faiss.read_index(str(self._index_path))
        else:
            dim = config.get("embedding_dim", 640)
            self._index = faiss.IndexFlatIP(dim)

        import json
        self._metadata: dict[int, dict] = {}
        if self._meta_path.exists():
            self._metadata = {
                int(k): v for k, v in
                json.loads(self._meta_path.read_text()).items()
            }

    def enroll(self, embedding: np.ndarray, dog_id: str,
               metadata: dict | None = None) -> int:
        emb = embedding.ravel().astype(np.float32)
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        idx = self._index.ntotal
        self._index.add(emb.reshape(1, -1))
        self._metadata[idx] = {"registered_dog_id": dog_id, "metadata": metadata or {}}
        self._save()
        return idx

    def search(self, query: np.ndarray, top_k: int = 10
               ) -> list[tuple[int, float, dict]]:
        if self._index.ntotal == 0:
            return []
        q = query.ravel().astype(np.float32).reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-8)
        scores, indices = self._index.search(q, min(top_k, self._index.ntotal))
        results = []
        for s, i in zip(scores[0], indices[0]):
            if i < 0:
                continue
            results.append((int(i), float(s), self._metadata.get(int(i), {})))
        return results

    @property
    def size(self) -> int:
        return self._index.ntotal

    def _save(self) -> None:
        import faiss, json
        faiss.write_index(self._index, str(self._index_path))
        self._meta_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2)
        )

    def close(self) -> None:
        self._save()
