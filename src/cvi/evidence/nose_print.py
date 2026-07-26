from __future__ import annotations

from pathlib import Path
import warnings

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cvi.evidence.base import AbstractEvidencer
from cvi.evidence.miewid import MiewIDReIDExtractor


class TinyViTBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        raise RuntimeError(
            "TinyViTBackbone is disabled: the implementation was an untrained "
            "three-layer CNN, not TinyViT. Select a checkpoint-backed backbone."
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        raise RuntimeError("TinyViTBackbone is disabled")


class MagFaceNoseHead(nn.Module):
    def __init__(self, embedding_dim: int, num_classes: int,
                 scale: float = 64.0, margin: float = 0.45):
        super().__init__()
        self.W = nn.Parameter(torch.randn(embedding_dim, num_classes) * 0.01)
        self._scale = scale
        self._margin = margin

    def forward(self, embeddings: torch.Tensor, labels: torch.Tensor
                ) -> tuple[torch.Tensor, torch.Tensor]:
        w = F.normalize(self.W, p=2, dim=0)
        x = F.normalize(embeddings, p=2, dim=1)
        cos_theta = torch.mm(x, w).clamp(-1.0 + 1e-7, 1.0 - 1e-7)
        theta = torch.acos(cos_theta)
        one_hot = F.one_hot(labels, num_classes=self.W.shape[1]).to(embeddings.dtype)
        target_logits = torch.cos(theta + self._margin)
        other_logits = cos_theta
        output = one_hot * target_logits + (1.0 - one_hot) * other_logits
        output *= self._scale
        norm_batch = torch.norm(embeddings, dim=1)
        mag_loss = torch.mean(torch.exp(-norm_batch))
        return output, mag_loss


class NoseEnhancer:
    def __init__(self, target_size: tuple[int, int] = (224, 224)):
        self._target_size = target_size

    def enhance(self, nose_crop: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(nose_crop, cv2.COLOR_RGB2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        l_eq = clahe.apply(l)
        enhanced = cv2.merge([l_eq, a, b])
        enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
        resized = cv2.resize(enhanced_rgb, self._target_size, interpolation=cv2.INTER_CUBIC)
        return resized


class MiewIDNoseExtractor(MiewIDReIDExtractor):
    """Deprecated compatibility alias for the former incorrect channel name."""

    def __init__(self, onnx_path: Path, input_size: int = 440):
        if input_size != 440:
            raise ValueError("MiewID-msv3 only supports its official 440x440 input")
        warnings.warn(
            "MiewIDNoseExtractor is deprecated; use MiewIDReIDExtractor. "
            "MiewID is a whole-crop wildlife ReID model, not a nose biometric.",
            DeprecationWarning,
            stacklevel=2,
        )
        super().__init__(onnx_path)


class YoloNoseDetector:
    def __init__(self, model_path: Path | None = None, conf_threshold: float = 0.5):
        self._conf = conf_threshold
        self._model = None
        if model_path and model_path.exists():
            self._model = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, image: Image.Image) -> tuple[int, int, int, int] | None:
        if self._model is None:
            return None
        img_np = np.array(image)
        blob = cv2.dnn.blobFromImage(img_np, 1/255, (416, 416), swapRB=True)
        self._model.setInput(blob)
        outputs = self._model.forward()
        boxes = []
        for out in outputs:
            for det in out:
                scores = det[5:]
                cls_id = int(scores.argmax())
                if scores[cls_id] < self._conf:
                    continue
                cx, cy, w, h = det[:4]
                x0 = int((cx - w/2) * image.width)
                y0 = int((cy - h/2) * image.height)
                x1 = int((cx + w/2) * image.width)
                y1 = int((cy + h/2) * image.height)
                boxes.append((x0, y0, x1, y1, float(scores[cls_id])))
        if not boxes:
            return None
        best = max(boxes, key=lambda b: b[4])
        return (best[0], best[1], best[2], best[3])


class DNPMask:
    def __init__(self, model: nn.Module | None = None):
        self._model = model

    def _build_unet(self) -> nn.Module:
        class UNet(nn.Module):
            def __init__(self):
                super().__init__()
                self._enc1 = nn.Sequential(
                    nn.Conv2d(3, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2))
                self._enc2 = nn.Sequential(
                    nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2))
                self._enc3 = nn.Sequential(
                    nn.Conv2d(128, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU(),
                    nn.MaxPool2d(2))
                self._bridge = nn.Sequential(
                    nn.Conv2d(256, 512, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(512, 512, 3, padding=1), nn.ReLU())
                self._up1 = nn.ConvTranspose2d(512, 256, 2, stride=2)
                self._dec1 = nn.Sequential(
                    nn.Conv2d(512, 256, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())
                self._up2 = nn.ConvTranspose2d(256, 128, 2, stride=2)
                self._dec2 = nn.Sequential(
                    nn.Conv2d(256, 128, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(128, 128, 3, padding=1), nn.ReLU())
                self._up3 = nn.ConvTranspose2d(128, 64, 2, stride=2)
                self._dec3 = nn.Sequential(
                    nn.Conv2d(128, 64, 3, padding=1), nn.ReLU(),
                    nn.Conv2d(64, 64, 3, padding=1), nn.ReLU())
                self._out = nn.Conv2d(64, 1, 1)

            def forward(self, x):
                e1 = self._enc1(x)
                e2 = self._enc2(e1)
                e3 = self._enc3(e2)
                b = self._bridge(e3)
                u1 = F.interpolate(self._up1(b), size=e3.shape[2:], mode="bilinear", align_corners=False)
                d1 = self._dec1(torch.cat([u1, e3], dim=1))
                u2 = F.interpolate(self._up2(d1), size=e2.shape[2:], mode="bilinear", align_corners=False)
                d2 = self._dec2(torch.cat([u2, e2], dim=1))
                u3 = F.interpolate(self._up3(d2), size=e1.shape[2:], mode="bilinear", align_corners=False)
                d3 = self._dec3(torch.cat([u3, e1], dim=1))
                return torch.sigmoid(self._out(d3))
        return UNet()

    def apply(self, nose_crop: np.ndarray) -> np.ndarray:
        import cv2
        if self._model is None:
            return nose_crop.astype(np.uint8, copy=True)
        h, w = nose_crop.shape[:2]
        if h < 16 or w < 16:
            return nose_crop.astype(np.uint8)
        tensor = torch.from_numpy(nose_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            mask = self._model(tensor).squeeze(0, 1).numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)
        return nose_crop * mask_bin[:, :, None] + 128 * (1 - mask_bin[:, :, None]).astype(np.uint8)
