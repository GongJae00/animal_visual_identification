from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

EMBEDDING_DIM = 640
VISUAL_SLICE = (0, 384)
TEXTURE_SLICE = (384, 512)
STRUCTURAL_SLICE = (512, 640)

def make_evidence_slices(
    visual_dim: int = 384,
    texture_dim: int = 128,
    structural_dim: int = 128,
) -> list[tuple[int, int, str]]:
    off = 0
    slices: list[tuple[int, int, str]] = []
    for name, d in [("visual", visual_dim), ("texture", texture_dim),
                     ("structural", structural_dim)]:
        if d > 0:
            slices.append((off, off + d, name))
            off += d
    return slices

EVIDENCE_SLICES = make_evidence_slices()


@dataclass(frozen=True, slots=True)
class IndexedIdentity:
    registered_dog_id: str
    dataset_name: str | None
    enrolled_at: str
    metadata: dict


@dataclass(frozen=True, slots=True)
class SearchResult:
    registered_dog_id: str
    similarity: float
    evidence: EvidenceBreakdown
    metadata: dict | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered_dog_id": self.registered_dog_id,
            "similarity": round(self.similarity, 6),
            "evidence": self.evidence.to_dict(),
        }


@dataclass(frozen=True, slots=True)
class EvidenceBreakdown:
    visual: float
    texture: float
    structural: float

    def to_dict(self) -> dict[str, float]:
        return {
            "visual": round(self.visual, 6),
            "texture": round(self.texture, 6),
            "structural": round(self.structural, 6),
        }


def _slice_embedding(emb: np.ndarray, start: int, end: int) -> np.ndarray:
    if emb.ndim == 1:
        return emb[start:end]
    return emb[:, start:end]


class IdentityIndex:
    """FAISS IndexFlatIP + JSON metadata for identity search.

    Stores L2-normalized d-dimensional embeddings for brute-force inner
    product search (cosine similarity when vectors are L2=1).

    SCALABILITY (현재 한계):
      - IndexFlatIP = O(N) brute force. 100K 등록 ≈ 244 MB, 검색 수백 ms.
      - 1M 등록 시 O(N) 검색이 실용적이지 않음 → IVF-PQ 전환 필요.
      - GPU 배포 시 cvi.gpu_index.GpuIdentityIndex 사용 권장.

    CROSS-DATASET (주의):
      - 같은 개가 여러 데이터셋에 등장해도 서로 다른 registered_dog_id 생성.
      - 교차 데이터셋 매칭은 현재 지원하지 않음.
      - dataset_name은 추적용 메타데이터일 뿐, 통합에 사용되지 않음.
    """

    def __init__(self, index_path: Path, metadata_path: Path,
                 dim: int = EMBEDDING_DIM) -> None:
        import faiss
        self._dim = dim
        self._index_path = index_path
        self._metadata_path = metadata_path
        if index_path.exists():
            self._index = faiss.read_index(str(index_path))
        else:
            self._index = faiss.IndexFlatIP(dim)
        self._metadata: dict[int, IndexedIdentity] = {}
        self._load_metadata()

    def _load_metadata(self) -> None:
        if not self._metadata_path.exists():
            return
        data = json.loads(self._metadata_path.read_text())
        for key, val in data.items():
            self._metadata[int(key)] = IndexedIdentity(**val)

    def _save_metadata(self) -> None:
        import os
        data = {str(k): {
            "registered_dog_id": v.registered_dog_id,
            "dataset_name": v.dataset_name,
            "enrolled_at": v.enrolled_at,
            "metadata": v.metadata,
        } for k, v in self._metadata.items()}
        tmp_path = self._metadata_path.with_suffix(self._metadata_path.suffix + ".tmp")
        tmp_path.write_text(
            json.dumps(data, ensure_ascii=False, sort_keys=True, indent=2)
        )
        os.replace(str(tmp_path), str(self._metadata_path))

    def enroll(self, embedding: np.ndarray, registered_dog_id: str,
               dataset_name: str | None = None,
               metadata: dict | None = None) -> int:
        emb = embedding.ravel().astype(np.float32)
        if emb.shape[0] != self._dim:
            raise ValueError(f"expected {self._dim}-d, got {emb.shape[0]}-d")
        norm = np.linalg.norm(emb)
        if norm > 0:
            emb = emb / norm
        n = self._index.ntotal
        self._index.add(emb[np.newaxis, :])
        self._metadata[n] = IndexedIdentity(
            registered_dog_id=registered_dog_id,
            dataset_name=dataset_name,
            enrolled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
            metadata=metadata or {},
        )
        self._save_metadata()
        import faiss, os
        tmp = str(self._index_path) + ".tmp"
        faiss.write_index(self._index, tmp)
        os.replace(tmp, str(self._index_path))
        return n

    def enroll_batch(self, embeddings: np.ndarray,
                     registered_dog_ids: list[str],
                     dataset_names: list[str | None] | None = None,
                     metadata_list: list[dict] | None = None) -> list[int]:
        embs = embeddings.astype(np.float32)
        if embs.shape[1] != self._dim:
            raise ValueError(f"expected {self._dim}-d, got {embs.shape[1]}-d")
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        embs = embs / norms
        start = self._index.ntotal
        self._index.add(embs)
        indices = list(range(start, start + len(embs)))
        dsn = dataset_names or [None] * len(embs)
        mds = metadata_list or [{}] * len(embs)
        for i, rid, d, m in zip(indices, registered_dog_ids, dsn, mds):
            self._metadata[i] = IndexedIdentity(
                registered_dog_id=rid, dataset_name=d,
                enrolled_at=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
                metadata=m,
            )
        self._save_metadata()
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        return indices

    def search(self, query_emb: np.ndarray, top_k: int = 5,
               fusion_weights: tuple[float, float, float] | None = None
               ) -> list[SearchResult]:
        q = query_emb.ravel().astype(np.float32)
        q_v = _slice_embedding(q, *VISUAL_SLICE)
        q_t = _slice_embedding(q, *TEXTURE_SLICE)
        q_s = _slice_embedding(q, *STRUCTURAL_SLICE)
        q_v_norm = q_v / max(np.linalg.norm(q_v), 1e-8)
        q_t_norm = q_t / max(np.linalg.norm(q_t), 1e-8)
        q_s_norm = q_s / max(np.linalg.norm(q_s), 1e-8)
        n = self._index.ntotal
        if n == 0:
            return []
        candidate_k = min(max(top_k * 5, 50), n)
        sims, indices = self._index.search(q[np.newaxis, :], candidate_k)
        results: list[SearchResult] = []
        fw = fusion_weights or (1.0, 0.5, 0.5)
        w_total = sum(fw)
        fw_norm = (fw[0] / w_total, fw[1] / w_total, fw[2] / w_total)

        for idx, coarse_sim in zip(indices[0], sims[0]):
            if idx < 0 or idx >= n:
                continue
            meta = self._metadata.get(int(idx))
            if meta is None:
                continue
            emb = self._index.reconstruct(int(idx))
            e_v = _slice_embedding(emb, *VISUAL_SLICE)
            e_t = _slice_embedding(emb, *TEXTURE_SLICE)
            e_s = _slice_embedding(emb, *STRUCTURAL_SLICE)
            s_v_n = np.linalg.norm(e_v)
            s_t_n = np.linalg.norm(e_t)
            s_s_n = np.linalg.norm(e_s)
            s_v = float(e_v @ q_v_norm) / max(float(s_v_n), 1e-8)
            s_t = float(e_t @ q_t_norm) / max(float(s_t_n), 1e-8)
            s_s = float(e_s @ q_s_norm) / max(float(s_s_n), 1e-8)
            fused = fw_norm[0] * s_v + fw_norm[1] * s_t + fw_norm[2] * s_s
            results.append(SearchResult(
                registered_dog_id=meta.registered_dog_id,
                similarity=fused,
                evidence=EvidenceBreakdown(visual=s_v, texture=s_t, structural=s_s),
                metadata=meta.metadata,
            ))
            if len(results) >= top_k:
                break
        return sorted(results, key=lambda r: r.similarity, reverse=True)[:top_k]

    @property
    def size(self) -> int:
        return self._index.ntotal

    def save(self) -> None:
        import faiss
        faiss.write_index(self._index, str(self._index_path))
        self._save_metadata()

    def close(self) -> None:
        self.save()
