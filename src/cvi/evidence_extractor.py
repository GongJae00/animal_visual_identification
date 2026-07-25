from __future__ import annotations

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image


_IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


# ---------------------------------------------------------------------------
# Abstract interface
# ---------------------------------------------------------------------------


class EvidenceExtractor(ABC):
    @property
    @abstractmethod
    def output_dim(self) -> int:
        ...

    @abstractmethod
    def extract(self, image: Image.Image) -> np.ndarray:
        ...

    @abstractmethod
    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        ...

    def close(self) -> None:
        pass


# ---------------------------------------------------------------------------
# ONNX-based extractor (generic)
# ---------------------------------------------------------------------------


class OnnxExtractor(EvidenceExtractor):
    def __init__(self, model_path: Path, input_size: int = 224,
                 output_dim: int | None = None,
                 mean: np.ndarray | None = None,
                 std: np.ndarray | None = None,
                 provider: str = "CPUExecutionProvider") -> None:
        import onnxruntime as ort
        self._sess = ort.InferenceSession(str(model_path), providers=[provider])
        self._inp = self._sess.get_inputs()[0]
        self._out = self._sess.get_outputs()[0]
        self._input_size = input_size
        self._dim = output_dim or self._out.shape[1]
        self._mean = mean if mean is not None else _IMAGENET_MEAN
        self._std = std if std is not None else _IMAGENET_STD

    @property
    def output_dim(self) -> int:
        return self._dim

    def preprocess(self, image: Image.Image) -> np.ndarray:
        img = image.convert("RGB").resize(
            (self._input_size, self._input_size), Image.BILINEAR
        )
        arr = np.array(img, dtype=np.float32) / 255.0
        arr = (arr - self._mean) / self._std
        return np.transpose(arr, (2, 0, 1))[np.newaxis, :]

    def extract(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image)
        emb = self._sess.run([self._out.name], {self._inp.name: tensor})[0]
        emb = emb.squeeze(0)
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        tensors = np.concatenate([self.preprocess(img) for img in images], axis=0)
        embs = self._sess.run([self._out.name], {self._inp.name: tensors})[0]
        norms = np.linalg.norm(embs, axis=1, keepdims=True)
        norms = np.where(norms > 0, norms, 1.0)
        return embs / norms

    def run_raw(self, image: Image.Image) -> np.ndarray:
        tensor = self.preprocess(image)
        return self._sess.run([self._out.name], {self._inp.name: tensor})[0].squeeze(0)


# ---------------------------------------------------------------------------
# DogFaceNet adapter (visual channel)
# ---------------------------------------------------------------------------


class DogFaceNetExtractor(EvidenceExtractor):
    def __init__(self, model_path: Path,
                 provider: str = "CPUExecutionProvider") -> None:
        self._onnx = OnnxExtractor(model_path, input_size=224, output_dim=384,
                                   provider=provider)

    @property
    def output_dim(self) -> int:
        return 384

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


# ---------------------------------------------------------------------------
# ConvNeXt adapter (texture channel)
# ---------------------------------------------------------------------------


class ConvNeXtExtractor(EvidenceExtractor):
    def __init__(self, model_path: Path,
                 provider: str = "CPUExecutionProvider") -> None:
        self._onnx = OnnxExtractor(model_path, input_size=224, output_dim=768,
                                   provider=provider)

    @property
    def output_dim(self) -> int:
        return 768

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


# ---------------------------------------------------------------------------
# SuperAnimal adapter (structural / landmarks)
# ---------------------------------------------------------------------------


class SuperAnimalExtractor(EvidenceExtractor):
    def __init__(self, model_path: Path, num_keypoints: int = 17,
                 output_dim: int = 256,
                 provider: str = "CPUExecutionProvider") -> None:
        self._num_keypoints = num_keypoints
        self._dim = output_dim
        self._onnx = OnnxExtractor(model_path, input_size=384, output_dim=None,
                                   provider=provider)

    @property
    def output_dim(self) -> int:
        return self._dim

    def _keypoints_to_embedding(self, kpts: np.ndarray) -> np.ndarray:
        k = kpts.reshape(-1, 2)
        center = k.mean(axis=0)
        centred = k - center
        scale = max(np.linalg.norm(centred, axis=1).max(), 1e-8)
        normalised = centred / scale
        pairwise_dists = []
        pairwise_angles = []
        for i in range(len(normalised)):
            for j in range(i + 1, len(normalised)):
                vec = normalised[j] - normalised[i]
                pairwise_dists.append(np.linalg.norm(vec))
                pairwise_angles.append(np.arctan2(vec[1], vec[0]))
        geom = np.concatenate([pairwise_dists, pairwise_angles]).astype(np.float32)
        norm = np.linalg.norm(geom)
        return geom / norm if norm > 0 else geom

    def extract(self, image: Image.Image) -> np.ndarray:
        raw = self._onnx.run_raw(image)
        kpts = raw[:self._num_keypoints * 2]
        emb = self._keypoints_to_embedding(kpts)
        if len(emb) > self._dim:
            emb = emb[:self._dim]
        elif len(emb) < self._dim:
            emb = np.pad(emb, (0, self._dim - len(emb)))
        return emb

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.array([self.extract(img) for img in images])


# ---------------------------------------------------------------------------
# Pet-ReID adapter (nose print)
# ---------------------------------------------------------------------------


class PetReIDExtractor(EvidenceExtractor):
    def __init__(self, model_path: Path,
                 provider: str = "CPUExecutionProvider") -> None:
        self._onnx = OnnxExtractor(model_path, input_size=224, output_dim=2048,
                                   provider=provider)

    @property
    def output_dim(self) -> int:
        return 2048

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._onnx.extract(image)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._onnx.extract_batch(images)


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------


class EvidenceExtractorRegistry:
    def __init__(self) -> None:
        self._extractors: dict[str, EvidenceExtractor] = {}

    def register(self, name: str, extractor: EvidenceExtractor) -> None:
        self._extractors[name] = extractor

    def get(self, name: str) -> EvidenceExtractor:
        if name not in self._extractors:
            raise KeyError(f"no extractor registered for '{name}'; "
                           f"available: {list(self._extractors)}")
        return self._extractors[name]

    @property
    def names(self) -> list[str]:
        return list(self._extractors)

    @property
    def visual(self) -> EvidenceExtractor | None:
        return self._extractors.get("visual")

    @property
    def texture(self) -> EvidenceExtractor | None:
        return self._extractors.get("texture")

    @property
    def structural(self) -> EvidenceExtractor | None:
        return self._extractors.get("structural")

    @property
    def nose(self) -> EvidenceExtractor | None:
        return self._extractors.get("nose")

    def close(self) -> None:
        for ext in self._extractors.values():
            ext.close()

    @staticmethod
    def from_onnx_dict(paths: dict[str, Path],
                       input_sizes: dict[str, int] | None = None,
                       output_dims: dict[str, int] | None = None
                       ) -> EvidenceExtractorRegistry:
        registry = EvidenceExtractorRegistry()
        for name, p in paths.items():
            if not p.exists():
                continue
            dim = (output_dims or {}).get(name)
            isize = (input_sizes or {}).get(name, 224)
            registry.register(name, OnnxExtractor(p, input_size=isize, output_dim=dim))
        return registry
