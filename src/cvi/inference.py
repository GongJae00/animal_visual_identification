"""Lightweight ONNX Runtime inference for canine re-identification.

No PyTorch dependency — only numpy, PIL, and onnxruntime are required.
This module is designed for easy porting to Rust/C++ with ONNX Runtime.

Pipeline
  1. Load ONNX model → ONNX Runtime session
  2. Load identity registry → registered_dog_id lookup
  3. Load gallery embeddings → numpy matrix
  4. For each query image: preprocess → embed → cosine-similarity → top-k
"""

from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


@dataclass(frozen=True, slots=True)
class InferenceConfig:
    model_path: str
    gallery_embeddings_path: str
    registry_db_path: str
    gallery_labels_path: str | None = None
    top_k: int = 5
    similarity_threshold: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {f.name: getattr(self, f.name) for f in self.__dataclass_fields__.values()}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> InferenceConfig:
        return cls(**{k: v for k, v in payload.items() if k in cls.__dataclass_fields__})


@dataclass(frozen=True, slots=True)
class MatchResult:
    registered_dog_id: str
    similarity: float
    dataset_name: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "registered_dog_id": self.registered_dog_id,
            "similarity": round(self.similarity, 6),
            "dataset_name": self.dataset_name,
        }


@dataclass(frozen=True, slots=True)
class QueryResult:
    query_token: str
    matches: tuple[MatchResult, ...]
    elapsed_ms: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "query_token": self.query_token,
            "matches": [m.to_dict() for m in self.matches],
            "elapsed_ms": round(self.elapsed_ms, 2),
        }


# ---------------------------------------------------------------------------
# Model loader
# ---------------------------------------------------------------------------


class OnnxEmbeddingModel:
    """ONNX Runtime wrapper for embedding extraction."""

    def __init__(self, model_path: Path) -> None:
        import onnxruntime as ort

        self._sess = ort.InferenceSession(
            str(model_path), providers=["CPUExecutionProvider"]
        )
        self._inp = self._sess.get_inputs()[0]
        self._out = self._sess.get_outputs()[0]
        self._input_shape = self._inp.shape
        self._output_dim = self._out.shape[1]

    @property
    def output_dim(self) -> int:
        return self._output_dim

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize((224, 224), Image.BILINEAR)
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - _IMAGENET_MEAN) / _IMAGENET_STD
        return np.transpose(arr, (2, 0, 1))[np.newaxis, :]

    def embed(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image)
        emb = self._sess.run([self._out.name], {self._inp.name: tensor})[0]
        emb = emb.squeeze(0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def embed_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = np.concatenate([self.preprocess(img) for img in images], axis=0)
        embs = self._sess.run([self._out.name], {self._inp.name: tensors})[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embs / norms


# ---------------------------------------------------------------------------
# Gallery
# ---------------------------------------------------------------------------


class Gallery:
    """Fixed gallery of registered embeddings for closed-set search."""

    def __init__(
        self,
        embeddings: np.ndarray,
        labels: list[str],
        registry: IdentityLookup | None = None,
    ) -> None:
        if len(embeddings) != len(labels):
            raise ValueError("embedding count must match label count")
        self._embeddings = embeddings.astype(np.float32)
        self._labels = labels
        self._registry = registry

    @classmethod
    def load(cls, embeddings_path: Path, labels_path: Path | None = None) -> Gallery:
        embeddings = np.load(str(embeddings_path))
        labels: list[str] = []
        if labels_path and labels_path.exists():
            data = json.loads(labels_path.read_text())
            labels = data if isinstance(data, list) else data.get("labels", [])
        if not labels:
            labels = [str(i) for i in range(len(embeddings))]
        return cls(embeddings, labels)

    def search(
        self, query_emb: np.ndarray, top_k: int = 5, threshold: float = 0.0
    ) -> list[MatchResult]:
        sim = self._embeddings @ query_emb.ravel()
        indices = np.argsort(sim)[::-1][:top_k]
        results: list[MatchResult] = []
        for idx in indices:
            score = float(sim[idx])
            if score < threshold:
                break
            label = self._labels[idx]
            dsn = None
            if self._registry:
                rec = self._registry.lookup_by_label(label)
                if rec:
                    dsn = rec.get("dataset_name")
            results.append(MatchResult(label, score, dsn))
        return results

    @property
    def size(self) -> int:
        return len(self._labels)


# ---------------------------------------------------------------------------
# Identity lookup
# ---------------------------------------------------------------------------


class IdentityLookup:
    """Read-only identity registry lookup from SQLite."""

    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(str(db_path))
        self._conn.row_factory = sqlite3.Row

    def lookup(self, identity_token: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT identity_token, dataset_identity_id, registered_dog_id, "
            "       dataset_name, image_count "
            "FROM identity_registry WHERE identity_token = ?",
            (identity_token,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def lookup_by_label(self, registered_dog_id: str) -> dict[str, Any] | None:
        row = self._conn.execute(
            "SELECT identity_token, dataset_identity_id, registered_dog_id, "
            "       dataset_name, image_count "
            "FROM identity_registry WHERE registered_dog_id = ?",
            (registered_dog_id,),
        ).fetchone()
        if row is None:
            return None
        return dict(row)

    def close(self) -> None:
        self._conn.close()


# ---------------------------------------------------------------------------
# Top-level inference
# ---------------------------------------------------------------------------


class EmbeddingInferencePipeline:
    """End-to-end inference: model → gallery → query."""

    def __init__(
        self,
        config: InferenceConfig,
    ) -> None:
        self._model = OnnxEmbeddingModel(Path(config.model_path))
        self._gallery = Gallery.load(
            Path(config.gallery_embeddings_path),
            Path(config.gallery_labels_path) if config.gallery_labels_path else None,
        )
        self._registry = IdentityLookup(Path(config.registry_db_path))
        self._top_k = config.top_k
        self._threshold = config.similarity_threshold

    def identify(self, image: Image.Image, query_token: str = "") -> QueryResult:
        import time
        t0 = time.perf_counter()
        emb = self._model.embed(image)
        matches = self._gallery.search(emb, self._top_k, self._threshold)
        elapsed = (time.perf_counter() - t0) * 1000
        return QueryResult(
            query_token=query_token or "unknown",
            matches=tuple(matches),
            elapsed_ms=elapsed,
        )

    def identify_batch(
        self, images: list[Image.Image], tokens: list[str] | None = None
    ) -> list[QueryResult]:
        import time
        t0 = time.perf_counter()
        embs = self._model.embed_batch(images)
        results: list[QueryResult] = []
        for i, emb in enumerate(embs):
            matches = self._gallery.search(emb, self._top_k, self._threshold)
            token = tokens[i] if tokens and i < len(tokens) else f"query_{i}"
            results.append(QueryResult(token, tuple(matches), 0.0))
        elapsed = (time.perf_counter() - t0) * 1000
        for r in results:
            object.__setattr__(r, "elapsed_ms", elapsed / len(results))
        return results

    def close(self) -> None:
        self._registry.close()
