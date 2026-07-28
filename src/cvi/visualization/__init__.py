"""Visual diagnostic writers with caller-controlled output locations."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from torch.nn import functional as F


def _ensure_dir(output_dir: Path) -> Path:
    path = Path(output_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def contact_sheet(
    images: list[Image.Image],
    *,
    output_dir: Path,
    grid_size: int = 8,
    thumb_size: int = 224,
    title: str = "",
) -> Path:
    """Save a contact sheet grid of thumbnail images."""
    out = _ensure_dir(output_dir)
    n = min(len(images), grid_size * grid_size)
    cols = min(grid_size, math.ceil(math.sqrt(n)))
    rows = math.ceil(n / cols)

    canvas = Image.new("RGB", (cols * thumb_size, rows * thumb_size), (40, 40, 40))
    for i, img in enumerate(images[:n]):
        r, c = divmod(i, cols)
        thumb = img.copy().resize((thumb_size, thumb_size), Image.Resampling.LANCZOS)
        canvas.paste(thumb, (c * thumb_size, r * thumb_size))

    path = out / f"{title or 'contact_sheet'}.jpg"
    canvas.save(path, quality=92)
    return path


def box_overlay(
    image: Image.Image,
    boxes: list[tuple[float, float, float, float]],
    *,
    labels: list[str] | None = None,
    colors: list[tuple[int, int, int]] | None = None,
    output_dir: Path,
    name: str = "overlay",
) -> Path:
    """Draw bounding boxes on an image and save."""
    out = _ensure_dir(output_dir)
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)
    if colors is None:
        colors = [(0, 255, 0)] * len(boxes)
    if labels is None:
        labels = [""] * len(boxes)

    for (x1, y1, x2, y2), label, color in zip(boxes, labels, colors, strict=True):
        draw.rectangle([x1, y1, x2, y2], outline=color, width=2)
        if label:
            draw.rectangle([x1, y1 - 14, x1 + len(label) * 7 + 4, y1], fill=color)
            draw.text((x1 + 2, y1 - 12), label, fill=(0, 0, 0))

    path = out / f"{name}.jpg"
    draw_img.save(path, quality=92)
    return path


def keypoint_overlay(
    image: Image.Image,
    keypoints: dict[str, tuple[float, float, float]],
    *,
    output_dir: Path,
    name: str = "keypoints",
    radius: int = 3,
) -> Path:
    """Draw keypoints with confidence-colored dots."""
    out = _ensure_dir(output_dir)
    draw_img = image.copy().convert("RGB")
    draw = ImageDraw.Draw(draw_img)

    for kp_name, (x, y, conf) in sorted(keypoints.items()):
        if conf <= 0:
            continue
        r = int(255 * (1 - conf))
        g = int(255 * conf)
        draw.ellipse(
            [x - radius, y - radius, x + radius, y + radius],
            fill=(r, g, 0),
            outline=(255, 255, 255),
        )

    path = out / f"{name}.jpg"
    draw_img.save(path, quality=92)
    return path


def attention_heatmap(
    image: Image.Image,
    attention_weights: torch.Tensor,
    *,
    output_dir: Path,
    name: str = "heatmap",
    patch_size: int = 14,
) -> Path:
    """Overlay DINOv2 attention as a heatmap on the original image."""
    if attention_weights.ndim == 3:
        attention_weights = attention_weights.mean(dim=0)
    if attention_weights.ndim == 2:
        attention_weights = attention_weights.mean(dim=0)

    attn = attention_weights.detach().float().cpu().numpy()
    n_patches = int(math.sqrt(len(attn)))
    if n_patches * n_patches != len(attn):
        attn = attn[: n_patches * n_patches]
        n_patches = int(math.sqrt(len(attn)))

    grid = attn.reshape(n_patches, n_patches)
    grid = (grid - grid.min()) / (grid.max() - grid.min() + 1e-6)

    heatmap = Image.fromarray((grid * 255).astype(np.uint8)).resize(
        image.size, Image.Resampling.BILINEAR
    )
    heatmap_rgb = Image.fromarray(
        np.stack(
            [
                np.asarray(heatmap),
                np.zeros_like(np.asarray(heatmap)),
                (255 - np.asarray(heatmap)) * 0.5,
            ],
            axis=-1,
        ).astype(np.uint8)
    )

    blended = Image.blend(image.convert("RGB"), heatmap_rgb, 0.5)
    out = _ensure_dir(output_dir)
    path = out / f"{name}.jpg"
    blended.save(path, quality=92)
    return path


def channel_evidence_grid(
    images: dict[str, Image.Image],
    *,
    output_dir: Path,
    name: str = "channels",
) -> Path:
    """Show original + per-channel evidence crops side by side."""
    out = _ensure_dir(output_dir)
    channels = sorted(images)
    if not channels:
        return out / f"{name}.jpg"

    w, h = next(iter(images.values())).size
    cols = len(channels)
    canvas = Image.new("RGB", (cols * w, h), (0, 0, 0))
    for i, ch in enumerate(channels):
        canvas.paste(images[ch].resize((w, h)), (i * w, 0))

    path = out / f"{name}.jpg"
    canvas.save(path, quality=92)
    return path


def gradcam_heatmap(
    model: torch.nn.Module,
    image_tensor: torch.Tensor,
    target_layer: str,
    *,
    output_dir: Path,
    name: str = "gradcam",
) -> Path:
    """Simple GradCAM — hook target layer, compute gradient-weighted activation map."""
    activations: dict[str, torch.Tensor] = {}
    gradients: dict[str, torch.Tensor] = {}

    def forward_hook(name):
        def hook(module, inp, out):
            activations[name] = out.detach()

        return hook

    def backward_hook(name):
        def hook(module, grad_in, grad_out):
            gradients[name] = grad_out[0].detach()

        return hook

    target_module = None
    for n, m in model.named_modules():
        if n == target_layer:
            target_module = m
            break

    if target_module is None:
        raise ValueError(f"layer {target_layer} not found")

    handle_fwd = target_module.register_forward_hook(forward_hook(target_layer))
    handle_bwd = target_module.register_full_backward_hook(backward_hook(target_layer))

    img = image_tensor.unsqueeze(0).requires_grad_(True)
    output = model(img)
    if isinstance(output, dict):
        score = output["embedding"].sum()
    else:
        score = output.sum()

    model.zero_grad()
    score.backward()

    handle_fwd.remove()
    handle_bwd.remove()

    act = activations[target_layer]
    grad = gradients[target_layer]

    weights = grad.mean(dim=(2, 3), keepdim=True)
    cam = (weights * act).sum(dim=1).squeeze(0)
    cam = F.relu(cam)
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-6)

    cam_img = Image.fromarray((cam.cpu().numpy() * 255).astype(np.uint8)).resize(
        (image_tensor.shape[2], image_tensor.shape[1]), Image.Resampling.BILINEAR
    )

    original = Image.fromarray(
        (image_tensor.permute(1, 2, 0).cpu().numpy() * 255).astype(np.uint8)
    )
    heatmap_rgb = Image.fromarray(
        np.stack(
            [
                np.asarray(cam_img),
                np.zeros_like(np.asarray(cam_img)),
                (255 - np.asarray(cam_img)) * 0.5,
            ],
            axis=-1,
        ).astype(np.uint8)
    )
    blended = Image.blend(original, heatmap_rgb, 0.4)

    out = _ensure_dir(output_dir)
    path = out / f"{name}.jpg"
    blended.save(path, quality=92)
    return path


__all__ = [
    "attention_heatmap",
    "box_overlay",
    "channel_evidence_grid",
    "contact_sheet",
    "gradcam_heatmap",
    "keypoint_overlay",
]
