from __future__ import annotations

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from cvi.evidence.base import AbstractEvidencer


DOGFLW_LANDMARKS: list[str] = [
    "left_eye", "right_eye", "nose_tip", "left_ear_base", "right_ear_base",
    "left_ear_tip", "right_ear_tip", "muzzle_left", "muzzle_right",
    "mouth_center", "chin", "left_cheek", "right_cheek",
    "forehead_center", "crown", "left_eye_corner", "right_eye_corner",
]


class HRNetHeatmap(nn.Module):
    def __init__(self, num_keypoints: int = 17):
        super().__init__()
        self._conv1 = nn.Conv2d(3, 64, 3, padding=1)
        self._bn1 = nn.BatchNorm2d(64)
        self._conv2 = nn.Conv2d(64, 128, 3, stride=2, padding=1)
        self._bn2 = nn.BatchNorm2d(128)
        self._conv3 = nn.Conv2d(128, 256, 3, stride=2, padding=1)
        self._bn3 = nn.BatchNorm2d(256)
        self._conv4 = nn.Conv2d(256, 512, 3, stride=2, padding=1)
        self._bn4 = nn.BatchNorm2d(512)
        self._heatmap = nn.Conv2d(512, num_keypoints, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self._bn1(self._conv1(x)))
        x = F.relu(self._bn2(self._conv2(x)))
        x = F.relu(self._bn3(self._conv3(x)))
        x = F.relu(self._bn4(self._conv4(x)))
        heatmaps = self._heatmap(x)
        return heatmaps


def heatmap_to_points(heatmaps: torch.Tensor) -> np.ndarray:
    b, k, h, w = heatmaps.shape
    flat = heatmaps.view(b, k, -1)
    max_idx = flat.argmax(dim=2)
    ys = (max_idx // w).float().cpu().numpy()
    xs = (max_idx % w).float().cpu().numpy()
    return np.stack([xs, ys], axis=-1)


def compute_pairwise_distances(points: np.ndarray) -> np.ndarray:
    diff = points[:, None] - points[None, :]
    dists = np.sqrt(np.sum(diff ** 2, axis=-1))
    return dists


class EdgeConv(nn.Module):
    def __init__(self, in_features: int, out_features: int, k: int = 8):
        super().__init__()
        self._k = k
        self._mlp = nn.Sequential(
            nn.Linear(in_features * 2, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, n, d = x.shape
        k = min(self._k, n - 1)
        idx = self._knn(x, k)
        idx_offset = idx + torch.arange(b, device=x.device).view(b, 1, 1) * n
        idx_flat = idx_offset.contiguous().view(b * n * k)
        x_flat = x.view(b * n, d)
        neighbors = x_flat[idx_flat].view(b, n, k, d)
        center = x.unsqueeze(2).expand(-1, -1, k, -1)
        edge_features = torch.cat([center, neighbors - center], dim=-1)
        out = self._mlp(edge_features.view(b * n * k, -1))
        out = out.view(b, n, k, -1).max(dim=2).values
        return out

    def _knn(self, x: torch.Tensor, k: int) -> torch.Tensor:
        inner = -2 * torch.matmul(x, x.transpose(2, 1))
        sq = (x ** 2).sum(dim=2, keepdim=True)
        pairwise = sq + inner + sq.transpose(1, 2)
        _, idx = pairwise.topk(k=k + 1, dim=-1, largest=False)
        return idx[:, :, 1:]


class LandmarkGraphEmbedder(nn.Module):
    def __init__(self, num_keypoints: int = 17, embedding_dim: int = 256):
        super().__init__()
        self._pos_enc = nn.Linear(2, 64)
        self._dist_enc = nn.Linear(num_keypoints * num_keypoints, 128)
        self._ec1 = EdgeConv(64, 128, k=6)
        self._ec2 = EdgeConv(128, 256, k=6)
        self._pool = nn.AdaptiveAvgPool1d(1)
        self._out = nn.Linear(256, embedding_dim)

    def forward(self, points: torch.Tensor) -> torch.Tensor:
        b, n, _ = points.shape
        pos_feat = F.relu(self._pos_enc(points))
        x = self._ec1(pos_feat)
        x = self._ec2(x)
        x = x.permute(0, 2, 1)
        x = self._pool(x).squeeze(-1)
        x = self._out(x)
        return F.normalize(x, p=2, dim=1)


class LandmarkEvidencer(AbstractEvidencer):
    name = "landmark"
    output_dim = 256

    def __init__(self, heatmap_model: nn.Module | None = None,
                 graph_model: nn.Module | None = None):
        if heatmap_model is None or graph_model is None:
            raise RuntimeError(
                "Landmark evidence is disabled until checkpoint-backed heatmap "
                "and graph models are supplied. Random placeholder models are "
                "not valid inference evidence."
            )
        self._heatmap = heatmap_model
        self._graph = graph_model
        self._heatmap.eval()
        self._graph.eval()

    def extract(self, image: Image.Image) -> np.ndarray:
        img = np.array(image.resize((224, 224)))
        tensor = torch.from_numpy(img).permute(2, 0, 1).float().unsqueeze(0) / 255.0
        with torch.no_grad():
            hm = self._heatmap(tensor)
            pts = heatmap_to_points(hm)
            pts_t = torch.from_numpy(pts).float()
            emb = self._graph(pts_t)
        return emb.squeeze(0).numpy()

    def extract_batch(self, images: list[Image.Image]) -> np.ndarray:
        return np.stack([self.extract(img) for img in images])
