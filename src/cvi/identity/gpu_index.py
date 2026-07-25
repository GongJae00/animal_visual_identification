from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

import faiss
import numpy as np

from cvi.identity_index import EVIDENCE_SLICES as _DEFAULT_EVIDENCE_SLICES


class GpuIdentityIndex:
    def __init__(self, dim: int = 640,
                 index_path: Path | None = None,
                 metadata_path: Path | None = None) -> None:
        self._dim = dim
        self._index_path = index_path
        self._metadata_path = metadata_path
        self._res = faiss.StandardGpuResources()
        cfg = faiss.GpuIndexFlatConfig()
        cfg.useFloat16CoarseQuantization = True
        self._index = faiss.GpuIndexFlatIP(self._res, dim, cfg)
        self._metadata: dict[int, dict[str, Any]] = {}
        if index_path and index_path.exists():
            self._load()

    def enroll(self, embedding: np.ndarray, registered_dog_id: str,
               metadata: dict | None = None) -> int:
        emb = embedding.ravel().astype(np.float32)
        if emb.shape[0] != self._dim:
            raise ValueError(f"expected {self._dim}-d, got {emb.shape[0]}-d")
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        idx = self._index.ntotal
        self._index.add(emb.reshape(1, -1))
        self._metadata[idx] = {
            "registered_dog_id": registered_dog_id,
            "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            "metadata": metadata or {},
        }
        self._save()
        return idx

    def enroll_batch(self, embeddings: np.ndarray,
                     registered_dog_ids: list[str],
                     metadata_list: list[dict] | None = None) -> list[int]:
        embs = embeddings.astype(np.float32)
        if embs.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-d, got {embs.shape[1]}-d")
        norms = np.linalg.norm(embs, axis=1, keepdims=True).clip(min=1e-8)
        embs = embs / norms
        start = self._index.ntotal
        self._index.add(embs)
        indices = list(range(start, start + len(embs)))
        mds = metadata_list or [{}] * len(embs)
        for i, rid, m in zip(indices, registered_dog_ids, mds):
            self._metadata[i] = {
                "registered_dog_id": rid,
                "enrolled_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                "metadata": m,
            }
        self._save()
        return indices

    def search(self, query: np.ndarray, top_k: int = 5
               ) -> list[tuple[int, float, dict]]:
        if self._index.ntotal == 0:
            return []
        q = query.ravel().astype(np.float32).reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-8)
        scores_np, indices_np = self._index.search(q, min(top_k, self._index.ntotal))
        results: list[tuple[int, float, dict]] = []
        for score, idx in zip(scores_np[0], indices_np[0]):
            if idx < 0:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append((int(idx), float(score), meta))
        return results

    def search_with_evidence(self, query: np.ndarray, top_k: int = 5,
                               slices: list[tuple[int, int, str]] | None = None
                               ) -> list[dict[str, Any]]:
        results = self.search(query, top_k)
        q = query.ravel().astype(np.float32)
        q_norm = q / max(np.linalg.norm(q), 1e-8)
        output: list[dict[str, Any]] = []
        slices = slices or _DEFAULT_EVIDENCE_SLICES
        for idx, score, meta in results:
            vec = self._index.reconstruct(int(idx))
            evidence = {}
            for start, end, name in slices:
                q_s = q_norm[start:end]
                e_s = vec[start:end]
                q_s /= max(np.linalg.norm(q_s), 1e-8)
                e_s /= max(np.linalg.norm(e_s), 1e-8)
                evidence[name] = float(np.dot(q_s, e_s))
            output.append({
                "index": idx,
                "registered_dog_id": meta.get("registered_dog_id", "unknown"),
                "similarity": score,
                "evidence": evidence,
                "metadata": meta.get("metadata", {}),
            })
        return output

    def remove(self, index: int) -> None:
        if index < 0 or index >= self._index.ntotal:
            raise IndexError(f"index {index} out of range, size={self._index.ntotal}")
        vecs = self._index.reconstruct_n(0, self._index.ntotal)
        mask = np.ones(self._index.ntotal, dtype=bool)
        mask[index] = False
        new_idx = faiss.IndexFlatIP(self._dim)
        new_idx.add(vecs[mask])
        self._index.reset()
        self._index.copyFrom(new_idx)
        old_meta = self._metadata.pop(index, {})
        new_metadata: dict[int, dict[str, Any]] = {}
        for old_idx, meta in self._metadata.items():
            new_idx_val = old_idx if old_idx < index else old_idx - 1
            new_metadata[new_idx_val] = meta
        self._metadata = new_metadata
        self._save()

    @property
    def size(self) -> int:
        return self._index.ntotal

    def _save(self) -> None:
        if self._index_path:
            cpu_idx = faiss.index_gpu_to_cpu(self._index)
            faiss.write_index(cpu_idx, str(self._index_path))
        if self._metadata_path:
            self._metadata_path.write_text(
                json.dumps(self._metadata, ensure_ascii=False, indent=2)
            )

    def _load(self) -> None:
        if self._index_path and self._index_path.exists():
            cpu_idx = faiss.read_index(str(self._index_path))
            gpu_idx = faiss.index_cpu_to_gpu(self._res, 0, cpu_idx)
            self._index = gpu_idx
        if self._metadata_path and self._metadata_path.exists():
            self._metadata = {
                int(k): v for k, v in json.loads(
                    self._metadata_path.read_text()
                ).items()
            }

    def to_cpu_index(self, index_path: Path, metadata_path: Path) -> None:
        cpu_idx = faiss.index_gpu_to_cpu(self._index)
        faiss.write_index(cpu_idx, str(index_path))
        metadata_path.write_text(
            json.dumps(self._metadata, ensure_ascii=False, indent=2)
        )

    def close(self) -> None:
        self._save()
        if hasattr(self, "_res"):
            del self._res
