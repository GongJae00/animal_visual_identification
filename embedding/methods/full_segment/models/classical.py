"""Deterministic foreground-only classical full-segment descriptor."""

from __future__ import annotations

import io
import json
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

from foundation.provenance import canonical_json_bytes
from embedding.methods.full_segment.preparation.data import Full128Sample, read_full128_crop

_GROUPS = ("hog", "hsv_histogram", "uniform_lbp")
_GROUP_DIMENSIONS = {"hog": 1764, "hsv_histogram": 32, "uniform_lbp": 10}
_STATE_SCHEMA = "cvi.full128_classical_state.v1"
_STATE_ARRAYS = {
    "metadata",
    "scaler_mean",
    "scaler_scale",
    "scaler_var",
    "scaler_n_samples_seen",
    "pca_components",
    "pca_mean",
    "pca_explained_variance",
    "pca_explained_variance_ratio",
    "pca_singular_values",
    "pca_noise_variance",
    "pca_n_samples",
}


@dataclass(frozen=True, slots=True)
class ClassicalFitInput:
    """One descriptor input explicitly admitted to estimator fitting."""

    rgb: np.ndarray
    mask: np.ndarray
    partition: str
    sample_id: str

    def __post_init__(self) -> None:
        if self.partition != "FIT":
            raise ValueError("Classical128 estimator fitting accepts only FIT inputs")
        if not isinstance(self.sample_id, str) or not self.sample_id:
            raise ValueError("Classical128 FIT sample_id must be non-empty")


class ClassicalDescriptorDataset:
    """Compute raw descriptors inside DataLoader workers in source order."""

    def __init__(
        self,
        samples: Sequence[Full128Sample],
        *,
        enabled_groups: Sequence[str] = _GROUPS,
    ) -> None:
        self.samples = tuple(samples)
        if not self.samples:
            raise ValueError("Classical128 descriptor dataset must be non-empty")
        self.model = Classical128(enabled_groups=enabled_groups)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, index: int) -> tuple[int, np.ndarray]:
        rgb, mask = read_full128_crop(self.samples[index])
        return index, self.model.raw_descriptor(rgb, mask)


def collate_classical_descriptors(
    rows: Sequence[tuple[int, np.ndarray]],
) -> tuple[np.ndarray, np.ndarray]:
    """Collate indexed descriptor rows without tensor conversion or reordering."""

    if not rows:
        raise ValueError("Classical128 descriptor batch must be non-empty")
    indices, descriptors = zip(*rows, strict=True)
    return np.asarray(indices, dtype=np.int64), np.stack(descriptors)


def initialize_classical_worker(worker_id: int) -> None:
    """Prevent OpenCV from adding nested thread pools in loader workers."""

    del worker_id
    cv2.setNumThreads(1)


class Classical128:
    """HOG, HSV-histogram, and uniform-LBP followed by fitted PCA to 128D."""

    output_dim = 128

    def __init__(self, *, enabled_groups: Sequence[str] = _GROUPS) -> None:
        groups = tuple(enabled_groups)
        if not groups or len(groups) != len(set(groups)):
            raise ValueError("enabled descriptor groups must be non-empty and unique")
        if any(group not in _GROUPS for group in groups):
            raise ValueError("unsupported Classical128 descriptor group")
        self.enabled_groups = tuple(group for group in _GROUPS if group in groups)
        self.scaler: StandardScaler | None = None
        self.pca: PCA | None = None
        self.fit_sample_ids: tuple[str, ...] = ()

    @property
    def raw_dimension(self) -> int:
        return sum(_GROUP_DIMENSIONS[group] for group in self.enabled_groups)

    @property
    def ablation_metadata(self) -> dict[str, object]:
        offset = 0
        slices: dict[str, dict[str, int]] = {}
        for group in self.enabled_groups:
            width = _GROUP_DIMENSIONS[group]
            slices[group] = {"start": offset, "stop": offset + width}
            offset += width
        return {
            "available_groups": list(_GROUPS),
            "enabled_groups": list(self.enabled_groups),
            "ablated_groups": [group for group in _GROUPS if group not in self.enabled_groups],
            "group_slices": slices,
            "raw_dimension": self.raw_dimension,
            "output_dimension": self.output_dim,
            "output_dimension_semantics": "UNINTERPRETED_PCA_COORDINATES",
        }

    def raw_descriptor(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        image, foreground = _prepare_inputs(rgb, mask)
        components = {
            "hog": _foreground_hog(image, foreground),
            "hsv_histogram": _foreground_hsv_histogram(image, foreground),
            "uniform_lbp": _foreground_uniform_lbp(image, foreground),
        }
        descriptor = np.concatenate([components[group] for group in self.enabled_groups])
        if descriptor.shape != (self.raw_dimension,) or not np.isfinite(descriptor).all():
            raise RuntimeError("Classical128 raw descriptor is invalid")
        return descriptor.astype(np.float32, copy=False)

    def fit(self, inputs: Sequence[ClassicalFitInput]) -> Classical128:
        samples = tuple(inputs)
        if len(samples) < self.output_dim:
            raise ValueError("Classical128 PCA fitting requires at least 128 FIT inputs")
        if any(not isinstance(sample, ClassicalFitInput) for sample in samples):
            raise TypeError("Classical128 fit inputs must be ClassicalFitInput values")
        sample_ids = tuple(sample.sample_id for sample in samples)
        if len(sample_ids) != len(set(sample_ids)):
            raise ValueError("Classical128 FIT sample IDs must be unique")
        matrix = np.stack(
            [self.raw_descriptor(sample.rgb, sample.mask) for sample in samples]
        ).astype(np.float64)
        self.fit_descriptors(matrix, sample_ids=sample_ids)
        return self

    def fit_descriptors(
        self, descriptors: np.ndarray, *, sample_ids: Sequence[str]
    ) -> np.ndarray:
        """Fit a retained raw matrix and return normalized FIT embeddings."""

        matrix = np.asarray(descriptors)
        ids = tuple(sample_ids)
        if matrix.ndim != 2 or matrix.shape[1] != self.raw_dimension:
            raise ValueError("Classical128 raw FIT descriptor matrix shape differs")
        if matrix.shape[0] < self.output_dim:
            raise ValueError("Classical128 PCA fitting requires at least 128 FIT inputs")
        if matrix.shape[0] != len(ids) or any(
            not isinstance(sample_id, str) or not sample_id for sample_id in ids
        ):
            raise ValueError("Classical128 FIT descriptor IDs are incomplete")
        if len(ids) != len(set(ids)):
            raise ValueError("Classical128 FIT sample IDs must be unique")
        if not np.isfinite(matrix).all():
            raise ValueError("Classical128 FIT descriptors must be finite")
        matrix = matrix.astype(np.float64, copy=False)
        scaler = StandardScaler(copy=True)
        scaled = scaler.fit_transform(matrix)
        pca = PCA(n_components=self.output_dim, svd_solver="full")
        pca.fit(scaled)
        transformed = pca.transform(scaled)
        self.scaler = scaler
        self.pca = pca
        self.fit_sample_ids = ids
        return _normalize_embeddings(transformed)

    def transform(self, rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
        return self.transform_descriptors(self.raw_descriptor(rgb, mask)[None, :])[0]

    def transform_descriptors(self, descriptors: np.ndarray) -> np.ndarray:
        """Transform a finite raw descriptor matrix using the fitted state."""

        if self.scaler is None or self.pca is None:
            raise RuntimeError("Classical128 must be fitted before transform")
        matrix = np.asarray(descriptors)
        if matrix.ndim != 2 or matrix.shape[1] != self.raw_dimension:
            raise ValueError("Classical128 raw descriptor matrix shape differs")
        if not np.isfinite(matrix).all():
            raise ValueError("Classical128 raw descriptors must be finite")
        transformed = self.pca.transform(
            self.scaler.transform(matrix.astype(np.float64, copy=False))
        )
        return _normalize_embeddings(transformed)

    def transform_batch(
        self, rgbs: Sequence[np.ndarray], masks: Sequence[np.ndarray]
    ) -> np.ndarray:
        if len(rgbs) != len(masks) or not rgbs:
            raise ValueError("Classical128 RGB and mask batches must be non-empty and aligned")
        return np.stack(
            [self.transform(rgb, mask) for rgb, mask in zip(rgbs, masks, strict=True)]
        )

    def save_state(self, path: Path) -> None:
        """Write fitted estimator state as deterministic array-only NPZ bytes."""

        if self.scaler is None or self.pca is None:
            raise RuntimeError("Classical128 must be fitted before serialization")
        if path.exists() or path.is_symlink():
            raise FileExistsError(f"refusing to overwrite Classical128 state: {path}")
        arrays = self._state_arrays()
        with path.open("xb") as stream, zipfile.ZipFile(
            stream, mode="w", compression=zipfile.ZIP_STORED, strict_timestamps=True
        ) as archive:
            for name in sorted(arrays):
                payload = io.BytesIO()
                np.save(payload, arrays[name], allow_pickle=False)
                info = zipfile.ZipInfo(f"{name}.npy", date_time=(1980, 1, 1, 0, 0, 0))
                info.compress_type = zipfile.ZIP_STORED
                info.external_attr = 0o600 << 16
                archive.writestr(info, payload.getvalue())

    @classmethod
    def load_state(cls, path: Path) -> Classical128:
        """Strictly restore array-only state without permitting pickle payloads."""

        if path.is_symlink() or not path.is_file():
            raise ValueError("Classical128 state must be a regular non-symlink NPZ file")
        if not 0 < path.stat().st_size <= 128 * 1024 * 1024:
            raise ValueError("Classical128 state byte size differs")
        arrays: dict[str, np.ndarray] = {}
        try:
            with zipfile.ZipFile(path, mode="r") as archive:
                names = archive.namelist()
                expected = {f"{name}.npy" for name in _STATE_ARRAYS}
                if set(names) != expected or len(names) != len(expected):
                    raise ValueError("Classical128 state arrays differ")
                if any(
                    info.is_dir()
                    or info.compress_type != zipfile.ZIP_STORED
                    or info.file_size > 64 * 1024 * 1024
                    for info in archive.infolist()
                ):
                    raise ValueError("Classical128 state archive entry differs")
                for name in sorted(_STATE_ARRAYS):
                    arrays[name] = np.load(
                        io.BytesIO(archive.read(f"{name}.npy")), allow_pickle=False
                    )
        except (OSError, zipfile.BadZipFile) as exc:
            raise ValueError("Classical128 state is not a valid NPZ archive") from exc
        return cls._from_state_arrays(arrays)

    def _state_arrays(self) -> dict[str, np.ndarray]:
        assert self.scaler is not None and self.pca is not None
        metadata = {
            "schema_version": _STATE_SCHEMA,
            "enabled_groups": list(self.enabled_groups),
            "fit_sample_ids": list(self.fit_sample_ids),
            "raw_dimension": self.raw_dimension,
            "output_dimension": self.output_dim,
            "svd_solver": "full",
        }
        return {
            "metadata": np.frombuffer(canonical_json_bytes(metadata), dtype=np.uint8),
            "scaler_mean": np.asarray(self.scaler.mean_, dtype="<f8"),
            "scaler_scale": np.asarray(self.scaler.scale_, dtype="<f8"),
            "scaler_var": np.asarray(self.scaler.var_, dtype="<f8"),
            "scaler_n_samples_seen": np.asarray(self.scaler.n_samples_seen_, dtype="<i8"),
            "pca_components": np.asarray(self.pca.components_, dtype="<f8"),
            "pca_mean": np.asarray(self.pca.mean_, dtype="<f8"),
            "pca_explained_variance": np.asarray(
                self.pca.explained_variance_, dtype="<f8"
            ),
            "pca_explained_variance_ratio": np.asarray(
                self.pca.explained_variance_ratio_, dtype="<f8"
            ),
            "pca_singular_values": np.asarray(self.pca.singular_values_, dtype="<f8"),
            "pca_noise_variance": np.asarray(self.pca.noise_variance_, dtype="<f8"),
            "pca_n_samples": np.asarray(self.pca.n_samples_, dtype="<i8"),
        }

    @classmethod
    def _from_state_arrays(cls, arrays: dict[str, np.ndarray]) -> Classical128:
        metadata_array = arrays["metadata"]
        if metadata_array.ndim != 1 or metadata_array.dtype != np.uint8:
            raise ValueError("Classical128 state metadata array differs")
        try:
            metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Classical128 state metadata is invalid") from exc
        expected_metadata = {
            "schema_version",
            "enabled_groups",
            "fit_sample_ids",
            "raw_dimension",
            "output_dimension",
            "svd_solver",
        }
        if not isinstance(metadata, dict) or set(metadata) != expected_metadata:
            raise ValueError("Classical128 state metadata fields differ")
        if metadata["schema_version"] != _STATE_SCHEMA or metadata["svd_solver"] != "full":
            raise ValueError("Classical128 state schema or solver differs")
        model = cls(enabled_groups=metadata["enabled_groups"])
        fit_ids = metadata["fit_sample_ids"]
        if (
            metadata["raw_dimension"] != model.raw_dimension
            or metadata["output_dimension"] != model.output_dim
            or not isinstance(fit_ids, list)
            or len(fit_ids) < model.output_dim
            or len(fit_ids) != len(set(fit_ids))
            or any(not isinstance(value, str) or not value for value in fit_ids)
        ):
            raise ValueError("Classical128 state metadata values differ")
        raw = model.raw_dimension
        output = model.output_dim
        _require_array(arrays["scaler_mean"], (raw,), np.float64, "scaler mean")
        _require_array(arrays["scaler_scale"], (raw,), np.float64, "scaler scale")
        _require_array(arrays["scaler_var"], (raw,), np.float64, "scaler variance")
        _require_array(arrays["pca_components"], (output, raw), np.float64, "PCA components")
        _require_array(arrays["pca_mean"], (raw,), np.float64, "PCA mean")
        for name in (
            "pca_explained_variance",
            "pca_explained_variance_ratio",
            "pca_singular_values",
        ):
            _require_array(arrays[name], (output,), np.float64, name)
        for name, dtype in (
            ("scaler_n_samples_seen", np.int64),
            ("pca_noise_variance", np.float64),
            ("pca_n_samples", np.int64),
        ):
            _require_array(arrays[name], (), dtype, name)
        if (
            int(arrays["scaler_n_samples_seen"]) != len(fit_ids)
            or int(arrays["pca_n_samples"]) != len(fit_ids)
            or np.any(arrays["scaler_scale"] <= 0)
            or np.any(arrays["scaler_var"] < 0)
        ):
            raise ValueError("Classical128 fitted population state differs")
        scaler = StandardScaler(copy=True)
        scaler.mean_ = arrays["scaler_mean"].astype(np.float64, copy=True)
        scaler.scale_ = arrays["scaler_scale"].astype(np.float64, copy=True)
        scaler.var_ = arrays["scaler_var"].astype(np.float64, copy=True)
        scaler.n_samples_seen_ = np.int64(len(fit_ids))
        scaler.n_features_in_ = raw
        pca = PCA(n_components=output, svd_solver="full")
        pca.components_ = arrays["pca_components"].astype(np.float64, copy=True)
        pca.mean_ = arrays["pca_mean"].astype(np.float64, copy=True)
        pca.explained_variance_ = arrays["pca_explained_variance"].astype(
            np.float64, copy=True
        )
        pca.explained_variance_ratio_ = arrays[
            "pca_explained_variance_ratio"
        ].astype(np.float64, copy=True)
        pca.singular_values_ = arrays["pca_singular_values"].astype(
            np.float64, copy=True
        )
        pca.noise_variance_ = float(arrays["pca_noise_variance"])
        pca.n_samples_ = len(fit_ids)
        pca.n_components_ = output
        pca.n_features_in_ = raw
        model.scaler = scaler
        model.pca = pca
        model.fit_sample_ids = tuple(fit_ids)
        return model


def _prepare_inputs(rgb: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    image = np.asarray(rgb)
    foreground = np.asarray(mask)
    if image.ndim != 3 or image.shape[2] != 3 or image.shape[:2] != foreground.shape:
        raise ValueError("Classical128 requires aligned RGB [H,W,3] and mask [H,W]")
    if min(image.shape[:2]) < 3 or not np.isfinite(image).all():
        raise ValueError("Classical128 RGB must be finite and at least 3x3")
    if not np.issubdtype(foreground.dtype, np.bool_) and (
        not np.isfinite(foreground).all()
        or not np.all((foreground == 0) | (foreground == 1))
    ):
        raise ValueError("Classical128 mask must be binary")
    foreground = foreground.astype(bool)
    if np.count_nonzero(foreground) < 9:
        raise ValueError("Classical128 mask must contain at least nine foreground pixels")
    if np.issubdtype(image.dtype, np.floating):
        if float(image.min()) < 0.0 or float(image.max()) > 1.0:
            raise ValueError("floating-point Classical128 RGB must be in [0,1]")
        image = np.rint(image * 255.0).astype(np.uint8)
    elif image.dtype != np.uint8:
        if float(image.min()) < 0.0 or float(image.max()) > 255.0:
            raise ValueError("integer Classical128 RGB must be in [0,255]")
        image = image.astype(np.uint8)
    neutral = np.median(image[foreground], axis=0)
    image = np.where(foreground[..., None], image, neutral).astype(np.uint8)
    image = cv2.resize(image, (64, 64), interpolation=cv2.INTER_AREA)
    foreground = cv2.resize(
        foreground.astype(np.uint8), (64, 64), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    if np.count_nonzero(foreground) < 9:
        raise ValueError("resized Classical128 mask has insufficient foreground")
    return image, foreground


def _normalize_embeddings(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != 128 or not np.isfinite(matrix).all():
        raise RuntimeError("Classical128 produced invalid embeddings")
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    if np.any(norms <= 1e-12):
        raise RuntimeError("Classical128 produced a zero embedding")
    return np.asarray(matrix / norms, dtype=np.float32)


def _require_array(
    value: np.ndarray, shape: tuple[int, ...], dtype: type[np.generic], label: str
) -> None:
    if value.shape != shape or value.dtype != np.dtype(dtype) or not np.isfinite(value).all():
        raise ValueError(f"Classical128 state {label} array differs")


def _foreground_hog(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    neutral = np.median(rgb[mask], axis=0)
    masked_rgb = np.where(mask[..., None], rgb, neutral).astype(np.uint8)
    gray = cv2.cvtColor(masked_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32) / 255.0
    grad_x = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=1)
    grad_y = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=1)
    magnitude, angle = cv2.cartToPolar(grad_x, grad_y, angleInDegrees=True)
    angle %= 180.0
    supported = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    magnitude *= supported
    cells = np.zeros((8, 8, 9), dtype=np.float32)
    bin_position = angle / 20.0
    lower = np.floor(bin_position).astype(np.int32) % 9
    upper = (lower + 1) % 9
    upper_weight = bin_position - np.floor(bin_position)
    for cell_y in range(8):
        ys = slice(cell_y * 8, (cell_y + 1) * 8)
        for cell_x in range(8):
            xs = slice(cell_x * 8, (cell_x + 1) * 8)
            mag = magnitude[ys, xs].ravel()
            cells[cell_y, cell_x] += np.bincount(
                lower[ys, xs].ravel(), weights=mag * (1.0 - upper_weight[ys, xs].ravel()), minlength=9
            ).astype(np.float32)
            cells[cell_y, cell_x] += np.bincount(
                upper[ys, xs].ravel(), weights=mag * upper_weight[ys, xs].ravel(), minlength=9
            ).astype(np.float32)
    blocks: list[np.ndarray] = []
    for y in range(7):
        for x in range(7):
            block = cells[y : y + 2, x : x + 2].ravel()
            block /= np.sqrt(float(block @ block) + 1e-6)
            blocks.append(block)
    return np.concatenate(blocks).astype(np.float32)


def _foreground_hsv_histogram(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    hsv = cv2.cvtColor(rgb, cv2.COLOR_RGB2HSV)
    values = hsv[mask]
    histograms = []
    for channel, bins, value_range in ((0, 16, (0, 180)), (1, 8, (0, 256)), (2, 8, (0, 256))):
        histogram, _ = np.histogram(values[:, channel], bins=bins, range=value_range)
        normalized = histogram.astype(np.float32)
        normalized /= max(float(normalized.sum()), 1.0)
        histograms.append(normalized)
    return np.concatenate(histograms)


def _foreground_uniform_lbp(rgb: np.ndarray, mask: np.ndarray) -> np.ndarray:
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
    center = gray[1:-1, 1:-1]
    neighbors = (
        gray[:-2, :-2],
        gray[:-2, 1:-1],
        gray[:-2, 2:],
        gray[1:-1, 2:],
        gray[2:, 2:],
        gray[2:, 1:-1],
        gray[2:, :-2],
        gray[1:-1, :-2],
    )
    bits = np.stack([neighbor >= center for neighbor in neighbors], axis=-1)
    transitions = np.count_nonzero(bits != np.roll(bits, 1, axis=-1), axis=-1)
    uniform_code = np.where(transitions <= 2, bits.sum(axis=-1), 9)
    supported = cv2.erode(mask.astype(np.uint8), np.ones((3, 3), np.uint8)).astype(bool)
    histogram = np.bincount(
        uniform_code[supported[1:-1, 1:-1]].astype(np.intp), minlength=10
    ).astype(np.float32)
    histogram /= max(float(histogram.sum()), 1.0)
    return histogram


__all__ = [
    "Classical128",
    "ClassicalDescriptorDataset",
    "ClassicalFitInput",
    "collate_classical_descriptors",
    "initialize_classical_worker",
]
