"""Receipt-bound DINOv2 appearance embedding."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image

from shared.contracts.dinov2_contract import Dinov2LocalArtifactContract
from representation.evidence.base import AbstractEvidencer


class ReceiptBoundDinov2Small(AbstractEvidencer):
    """DINOv2-small loaded only from source-bound local Hugging Face files."""

    name = "appearance"
    output_dim = 384

    def __init__(
        self,
        *,
        model_directory: str | Path,
        weight_intake_bundle: str | Path,
        preprocessor_intake_bundle: str | Path,
        device: str = "cpu",
        max_batch_size: int = 32,
    ) -> None:
        if device not in {"cpu", "cuda"}:
            raise ValueError("device must be 'cpu' or 'cuda'")
        if (
            isinstance(max_batch_size, bool)
            or not isinstance(max_batch_size, int)
            or max_batch_size <= 0
        ):
            raise ValueError("max_batch_size must be a positive integer")
        self._contract = Dinov2LocalArtifactContract.load(
            model_directory=Path(model_directory),
            weight_intake_bundle=Path(weight_intake_bundle),
            preprocessor_intake_bundle=Path(preprocessor_intake_bundle),
        )
        self._device_name = device
        self._max_batch_size = max_batch_size
        self._backbone = None

    @property
    def model_sha256(self) -> str:
        return self._contract.model_sha256

    @property
    def preprocessor_sha256(self) -> str:
        return self._contract.preprocessor_sha256

    @property
    def weight_receipt_sha256(self) -> str:
        return self._contract.weight_receipt_sha256

    @property
    def preprocessor_receipt_sha256(self) -> str:
        return self._contract.preprocessor_receipt_sha256

    @property
    def config_sha256(self) -> str:
        return self._contract.config_sha256

    @property
    def gallery_contract_fields(self) -> dict[str, str]:
        return {
            "model_sha256": self.model_sha256,
            "model_config_sha256": self.config_sha256,
            "preprocessor_sha256": self.preprocessor_sha256,
            "weight_intake_receipt_sha256": self.weight_receipt_sha256,
            "preprocessor_intake_receipt_sha256": (
                self.preprocessor_receipt_sha256
            ),
        }

    def _ensure_loaded(self) -> None:
        if self._backbone is not None:
            return
        import torch

        if self._device_name == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        self._contract.revalidate_local_files()
        from transformers import Dinov2Model

        backbone = Dinov2Model.from_pretrained(
            str(self._contract.model_directory),
            local_files_only=True,
            trust_remote_code=False,
            use_safetensors=True,
        )
        if not isinstance(backbone, torch.nn.Module):
            raise TypeError("local DINOv2 loader must return a torch.nn.Module")
        self._contract.revalidate_local_files()
        backbone.to(torch.device(self._device_name)).eval()
        for parameter in backbone.parameters():
            parameter.requires_grad = False
        self._backbone = backbone

    def _preprocess(self, images: list[Image.Image]):
        import torch

        if not images:
            raise ValueError("at least one image is required")
        if len(images) > self._max_batch_size:
            raise ValueError(
                f"batch size {len(images)} exceeds cap {self._max_batch_size}"
            )
        processor = self._contract.preprocessor
        shortest_edge = processor["size"]["shortest_edge"]
        crop_height = processor["crop_size"]["height"]
        crop_width = processor["crop_size"]["width"]
        arrays: list[np.ndarray] = []
        for image in images:
            rgb = image.convert("RGB")
            width, height = rgb.size
            if width <= height:
                resized_width = shortest_edge
                resized_height = int(shortest_edge * height / width)
            else:
                resized_height = shortest_edge
                resized_width = int(shortest_edge * width / height)
            resized = rgb.resize(
                (resized_width, resized_height),
                Image.Resampling.BICUBIC,
            )
            left = (resized_width - crop_width) // 2
            top = (resized_height - crop_height) // 2
            arrays.append(
                np.asarray(
                    resized.crop(
                        (left, top, left + crop_width, top + crop_height)
                    ),
                    dtype=np.uint8,
                )
            )
        batch = np.stack(arrays).transpose(0, 3, 1, 2)
        device = torch.device(self._device_name)
        tensor = torch.from_numpy(batch).to(
            device=device,
            dtype=torch.float32,
            non_blocking=device.type == "cuda",
        )
        tensor.mul_(processor["rescale_factor"])
        mean = torch.tensor(
            processor["image_mean"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        std = torch.tensor(
            processor["image_std"], device=device, dtype=torch.float32
        ).view(1, 3, 1, 1)
        return tensor.sub_(mean).div_(std)

    def _extract_tensor(self, images: list[Image.Image]) -> np.ndarray:
        self._ensure_loaded()
        import torch

        pixels = self._preprocess(images)
        with torch.inference_mode():
            output = self._backbone(pixel_values=pixels)
        embeddings = getattr(output, "pooler_output", None)
        if not isinstance(embeddings, torch.Tensor) or embeddings.shape != (
            len(images),
            self.output_dim,
        ):
            raise RuntimeError(
                "DINOv2 pooler output must have shape "
                f"[{len(images)}, {self.output_dim}]"
            )
        if not torch.isfinite(embeddings).all():
            raise RuntimeError("DINOv2 output contains non-finite values")
        norms = torch.linalg.vector_norm(embeddings.float(), dim=1, keepdim=True)
        if torch.any(norms <= 1e-8):
            raise RuntimeError("DINOv2 output contains a zero embedding")
        return (embeddings.float() / norms).cpu().numpy().astype(
            np.float32, copy=False
        )

    def extract(self, image: Image.Image) -> np.ndarray:
        return self._extract_tensor([image])[0]

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return self._extract_tensor(images)
