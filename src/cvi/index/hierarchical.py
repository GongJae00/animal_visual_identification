from __future__ import annotations

from pathlib import Path
from typing import Any

import faiss
import numpy as np

from cvi.index.base import AbstractIdentityIndex
from cvi.identity_index import make_evidence_slices, EVIDENCE_SLICES


class SpeciesFilteredIndex(AbstractIdentityIndex):
    def __init__(self, base_index_dir: Path,
                 breed_mapping: dict[str, list[str]] | None = None,
                 dim: int = 640):
        self._dim = dim
        self._base_dir = base_index_dir
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._metadata: dict[int, dict[str, Any]] = {}
        self._breed_index: dict[int, str] = {}
        self._breed_mapping = breed_mapping or {}
        master_path = self._base_dir / "master.idx"
        if master_path.exists():
            self._index = faiss.read_index(str(master_path))
            self._load_metadata()
            self._load_breeds()
        else:
            self._index = faiss.IndexFlatIP(dim)

    def _load_metadata(self) -> None:
        meta_path = self._base_dir / "metadata.json"
        if meta_path.exists():
            import json
            data = json.loads(meta_path.read_text())
            self._metadata = {int(k): v for k, v in data.items()}

    def _load_breeds(self) -> None:
        breed_path = self._base_dir / "breed_index.json"
        if breed_path.exists():
            import json
            data = json.loads(breed_path.read_text())
            self._breed_index = {int(k): v for k, v in data.items()}

    def _shard_path(self, breed: str) -> Path:
        return self._base_dir / f"{breed}.idx"

    def enroll(self, embedding: np.ndarray, registered_dog_id: str,
               metadata: dict | None = None) -> int:
        return self.enroll_with_breed(embedding, registered_dog_id, "unknown", metadata)

    def enroll_with_breed(self, embedding: np.ndarray, dog_id: str,
                          breed: str, metadata: dict | None = None) -> int:
        # dedup: check if dog already enrolled
        for idx, meta in self._metadata.items():
            if meta.get("registered_dog_id") == dog_id:
                return int(idx)
        emb = embedding.ravel().astype(np.float32)
        if emb.shape[0] != self._dim:
            raise ValueError(f"expected {self._dim}-d, got {emb.shape[0]}-d")
        emb = emb / max(np.linalg.norm(emb), 1e-8)
        idx = self._index.ntotal
        self._index.add(emb.reshape(1, -1))
        self._metadata[idx] = {"registered_dog_id": dog_id, "metadata": metadata or {}}
        self._breed_index[idx] = breed
        return idx

    def search_filtered(self, query: np.ndarray,
                        allowed_breeds: list[str] | None = None,
                        top_k: int = 5) -> list[tuple[int, float, dict]]:
        if self._index.ntotal == 0:
            return []
        q = query.ravel().astype(np.float32).reshape(1, -1)
        q = q / max(np.linalg.norm(q), 1e-8)
        scores_np, indices_np = self._index.search(q, self._index.ntotal)
        results: list[tuple[int, float, dict]] = []
        breed_set = set(allowed_breeds or [])
        for score, idx in zip(scores_np[0], indices_np[0]):
            if idx < 0:
                continue
            if breed_set and self._breed_index.get(int(idx), "") not in breed_set:
                continue
            meta = self._metadata.get(int(idx), {})
            results.append((int(idx), float(score), meta))
            if len(results) >= top_k:
                break
        return results

    def search(self, query: np.ndarray, top_k: int = 5
               ) -> list[tuple[int, float, dict]]:
        return self.search_filtered(query, None, top_k)

    def search_with_evidence(self, query: np.ndarray, top_k: int = 5,
                             slices: list[tuple[int, int, str]] | None = None
                             ) -> list[dict[str, Any]]:
        results = self.search(query, top_k)
        q = query.ravel().astype(np.float32)
        q_norm = q / max(np.linalg.norm(q), 1e-8)
        output: list[dict[str, Any]] = []
        _slices = slices or EVIDENCE_SLICES
        for idx, score, meta in results:
            vec = self._index.reconstruct(int(idx))
            evidence = {}
            for start, end, name in _slices:
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
        raise NotImplementedError("remove not supported for SpeciesFilteredIndex")

    @property
    def size(self) -> int:
        return self._index.ntotal

    def close(self) -> None:
        pass

    def save(self) -> None:
        import json, tempfile, shutil, os
        # Atomic write: temp → rename (crash-safe)
        base = self._base_dir
        # FAISS
        fx = str(base / "master.idx")
        fx_tmp = fx + ".tmp"
        faiss.write_index(self._index, fx_tmp)
        os.replace(fx_tmp, fx)
        # metadata
        mf = base / "metadata.json"
        mf_tmp = base / "metadata.json.tmp"
        mf_tmp.write_text(json.dumps(self._metadata, ensure_ascii=False))
        os.replace(str(mf_tmp), str(mf))
        # breed index
        bf = base / "breed_index.json"
        bf_tmp = base / "breed_index.json.tmp"
        data = {str(k): v for k, v in self._breed_index.items()}
        bf_tmp.write_text(json.dumps(data, ensure_ascii=False))
        os.replace(str(bf_tmp), str(bf))
