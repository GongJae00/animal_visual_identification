from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cvi.evidence.base import AbstractEvidencer


class TinyViTBackbone(nn.Module):
    def __init__(self, embedding_dim: int = 512):
        super().__init__()
        self._conv1 = nn.Conv2d(3, 64, kernel_size=3, stride=2, padding=1)
        self._bn1 = nn.BatchNorm2d(64)
        self._conv2 = nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1)
        self._bn2 = nn.BatchNorm2d(128)
        self._conv3 = nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1)
        self._bn3 = nn.BatchNorm2d(256)
        self._pool = nn.AdaptiveAvgPool2d(1)
        self._fc = nn.Linear(256, embedding_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._bn1(self._conv1(x)))
        x = F.relu(self._bn2(self._conv2(x)))
        x = F.relu(self._bn3(self._conv3(x)))
        x = self._pool(x).flatten(1)
        x = self._fc(x)
        return F.normalize(x, p=2, dim=1)


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


class MiewIDNoseExtractor(AbstractEvidencer):
    name = "miewid"
    output_dim = 2152

    def __init__(self, onnx_path: Path, input_size: int = 160):
        import onnxruntime as ort
        if not onnx_path.exists():
            raise FileNotFoundError(
                f"MiewID ONNX model not found: {onnx_path}\n"
                f"  다운로드: python tools/download_models.py --model miewid"
            )
        self._sess = ort.InferenceSession(str(onnx_path))
        self._input_name = self._sess.get_inputs()[0].name
        self._input_size = input_size

    def extract(self, image: Image.Image) -> np.ndarray:
        img = image.resize((self._input_size, self._input_size))
        arr = np.array(img, dtype=np.float32).transpose(2, 0, 1)[None] / 255.0
        out = self._sess.run(None, {self._input_name: arr})[0][0]
        return out / max(np.linalg.norm(out), 1e-8)

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([self.extract(img) for img in images])


class YoloNoseDetector:
    def __init__(self, model_path: Path | None = None, conf_threshold: float = 0.5):
        self._conf = conf_threshold
        self._model = None
        if model_path and model_path.exists():
            self._model = cv2.dnn.readNetFromONNX(str(model_path))

    def detect(self, image: Image.Image) -> tuple[int, int, int, int] | None:
        if self._model is None:
            w, h = image.size
            return (w // 4, h // 3, 3 * w // 4, 2 * h // 3)
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
    def __init__(self):
        self._model = self._build_unet()

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
        h, w = nose_crop.shape[:2]
        if h < 16 or w < 16:
            return nose_crop.astype(np.uint8)
        tensor = torch.from_numpy(nose_crop).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            mask = self._model(tensor).squeeze(0, 1).numpy()
        mask_resized = cv2.resize(mask, (w, h), interpolation=cv2.INTER_LINEAR)
        mask_bin = (mask_resized > 0.5).astype(np.uint8)
        return nose_crop * mask_bin[:, :, None] + 128 * (1 - mask_bin[:, :, None]).astype(np.uint8)
