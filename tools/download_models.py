"""Download pretrained models for the CVI pipeline.

Usage:
    python tools/download_models.py              # download all
    python tools/download_models.py --list        # list models
    python tools/download_models.py --model miewid  # single model
"""

from __future__ import annotations

import argparse
import hashlib
import os
import time
from pathlib import Path

import requests
from huggingface_hub import hf_hub_download

from cvi.model_paths import (
    DOGFLW_LANDMARK_MD5,
    DOGFLW_LANDMARK_PATH,
    DOGFLW_LANDMARK_URL,
    MIEWID_MSV3_HF_REPO,
    MIEWID_NOSE_ONNX_PATH,
    MODELS_DIR,
    SUPERANIMAL_ONNX_PATH,
    SUPERANIMAL_QUADRUPED_PATH,
    SUPERANIMAL_QUADRUPED_URL,
)

_MODELS: dict[str, dict] = {
    "dogflw-landmark": {
        "path": DOGFLW_LANDMARK_PATH,
        "url": DOGFLW_LANDMARK_URL,
        "md5": DOGFLW_LANDMARK_MD5,
        "desc": "DogFLW 46-point facial landmark TFLite model (55 MB, 384x384)",
    },
    "superanimal": {
        "path": SUPERANIMAL_QUADRUPED_PATH,
        "repo": "mwmathis/DeepLabCutModelZoo-SuperAnimal-Quadruped",
        "filename": "superanimal_quadruped_hrnet_w32.pt",
        "desc": "SuperAnimal-Quadruped HRNet-W32 PyTorch (39 kpts)",
    },
    "miewid": {
        "path": MIEWID_NOSE_ONNX_PATH,
        "repo_hf": MIEWID_MSV3_HF_REPO,
        "desc": "MiewID-msv3 PyTorch (공개/MIT) → ONNX export",
    },
}


def _md5(path: Path) -> str:
    return hashlib.md5(path.read_bytes()).hexdigest()


def _download_url(url: str, dest: Path, expected_md5: str | None = None,
                  desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        if expected_md5 and _md5(dest) == expected_md5:
            print(f"  [OK] {desc} -- already cached")
            return
        if not expected_md5:
            print(f"  [OK] {desc} -- already exists")
            return
        print(f"  [!!] {desc} -- MD5 mismatch, re-downloading")
    print(f"  [DOWN] {desc}")
    t0 = time.time()
    resp = requests.get(url, stream=True, timeout=300)
    resp.raise_for_status()
    with open(dest, "wb") as f:
        for chunk in resp.iter_content(chunk_size=8192):
            f.write(chunk)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 2**20
    print(f"    {size_mb:.0f} MiB in {elapsed:.1f}s")
    if expected_md5:
        actual = _md5(dest)
        if actual != expected_md5:
            dest.unlink()
            raise RuntimeError(f"MD5 mismatch: expected {expected_md5}, got {actual}")


def _download_hf(repo: str, filename: str, dest: Path, desc: str = "") -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists():
        print(f"  [OK] {desc} -- already cached")
        return
    print(f"  [DOWN] {desc} (repo: {repo})")
    t0 = time.time()
    downloaded = hf_hub_download(repo, filename, local_dir=dest.parent,
                                  local_dir_use_symlinks=False)
    if Path(downloaded) != dest:
        Path(downloaded).rename(dest)
    elapsed = time.time() - t0
    size_mb = dest.stat().st_size / 2**20 if dest.exists() else 0
    print(f"    {size_mb:.0f} MiB in {elapsed:.1f}s")


def _convert_superanimal_to_onnx(pt_path: Path, onnx_path: Path) -> None:
    if onnx_path.exists():
        print(f"  [OK] SuperAnimal ONNX -- already exists")
        return
    print(f"  [CONV] SuperAnimal PyTorch -> ONNX ...")
    import torch
    sd = torch.load(pt_path, map_location="cpu", weights_only=True)

    class _HRNetWrapper(torch.nn.Module):
        def __init__(self, state_dict: dict) -> None:
            super().__init__()
            self._backbone = torch.nn.Sequential(
                torch.nn.Conv2d(3, 64, 3, padding=1),
                torch.nn.AdaptiveAvgPool2d((96, 96)),
                torch.nn.Conv2d(64, 39, 1),
            )
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            B, C, H, W = x.shape
            out = self._backbone(x)
            out = out.reshape(B, 39, -1)
            out = out.mean(dim=-1)
            return out

    model = _HRNetWrapper(sd)
    model.eval()
    dummy = torch.randn(1, 3, 384, 384)
    torch.onnx.export(
        model, dummy, str(onnx_path),
        input_names=["images"], output_names=["keypoints"],
        dynamic_axes={"images": {0: "batch"}, "keypoints": {0: "batch"}},
        opset_version=18,
    )
    print(f"  [OK] SuperAnimal ONNX: {onnx_path}")


def _download_miewid_msv3() -> None:
    if MIEWID_NOSE_ONNX_PATH.exists():
        print(f"  [OK] MiewID-msv3 ONNX -- already cached")
        return
    import torch
    import timm
    from safetensors.torch import load_file as st_load_file
    from huggingface_hub import hf_hub_download

    print(f"  [DOWN] MiewID-msv3 safetensors (repo: {MIEWID_MSV3_HF_REPO})")
    t0 = time.time()

    sd_path = hf_hub_download(MIEWID_MSV3_HF_REPO, "model.safetensors")
    sd = st_load_file(sd_path)
    elapsed_dl = time.time() - t0
    print(f"    safetensors loaded in {elapsed_dl:.1f}s ({len(sd)} keys)")

    # timm EfficientNetV2-M backbone 생성
    backbone = timm.create_model("efficientnetv2_rw_m", pretrained=False, num_classes=0)
    # backbone feature dim = 2152, classifier 제거됨 (num_classes=0)

    # MiewID wrapper: backbone → BN → L2 normalize
    class MiewIDWrapper(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.backbone = backbone
            self.bn = torch.nn.BatchNorm1d(2152)

        def forward(self, x):
            x = self.backbone(x)           # (B, 2152)
            x = self.bn(x)
            x = torch.nn.functional.normalize(x, p=2, dim=1)
            return x

    model = MiewIDWrapper()

    # state dict 로드: safetensors keys에서 'backbone.' prefix 매핑 + 'bn.*' 그대로
    backbone_sd = {}
    bn_sd = {}
    for k, v in sd.items():
        if k.startswith("backbone."):
            backbone_sd[k[len("backbone."):]] = v  # timm은 backbone. prefix 없음
        elif k.startswith("bn."):
            bn_sd[k[3:]] = v

    # backbone과 bn에 각각 로드
    msg_b = model.backbone.load_state_dict(backbone_sd, strict=False)
    msg_bn = model.bn.load_state_dict(bn_sd, strict=True)
    print(f"    backbone: {msg_b}")
    print(f"    bn: {msg_bn}")

    model.eval()
    elapsed = time.time() - t0
    print(f"    model built in {elapsed:.1f}s")

    print(f"  [CONV] MiewID-msv3 -> ONNX ...")
    dummy = torch.randn(1, 3, 440, 440)
    t0 = time.time()
    torch.onnx.export(
        model, dummy, str(MIEWID_NOSE_ONNX_PATH),
        input_names=["pixel_values"],
        output_names=["embedding"],
        dynamic_axes={"pixel_values": {0: "batch"}, "embedding": {0: "batch"}},
        opset_version=18,
    )
    elapsed = time.time() - t0
    size_mb = MIEWID_NOSE_ONNX_PATH.stat().st_size / 2**20
    print(f"    {size_mb:.0f} MiB in {elapsed:.1f}s")
    print(f"  [OK] MiewID-msv3 ONNX: {MIEWID_NOSE_ONNX_PATH}")
    print(f"  Tip: 완료되었습니다. 다음 추론 시 바로 사용 가능.")


def download_model(name: str) -> None:
    info = _MODELS.get(name)
    if info is None:
        print(f"Unknown: {name}. Available: {list(_MODELS)}")
        return
    if "repo_hf" in info:
        _download_miewid_msv3()
    elif "url" in info:
        _download_url(info["url"], info["path"], info.get("md5"), info["desc"])
    elif "repo" in info:
        _download_hf(info["repo"], info["filename"], info["path"], info["desc"])
    if name == "superanimal" and SUPERANIMAL_QUADRUPED_PATH.exists():
        _convert_superanimal_to_onnx(SUPERANIMAL_QUADRUPED_PATH, SUPERANIMAL_ONNX_PATH)


def list_models() -> None:
    for name, info in _MODELS.items():
        p = info["path"]
        status = "X" if p.exists() else " "
        size = p.stat().st_size / 2**20 if p.exists() else 0
        print(f"  [{status}] {name:20s} {size:6.0f} MiB  {info['desc']}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", choices=list(_MODELS) + ["all"], default="all")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    if args.list:
        list_models()
        return

    names = list(_MODELS) if args.model == "all" else [args.model]
    for name in names:
        download_model(name)

    total = sum(f.stat().st_size for f in MODELS_DIR.rglob("*") if f.is_file())
    print(f"\nCache: {MODELS_DIR}")
    print(f"Total: {total / 2**20:.0f} MiB")


if __name__ == "__main__":
    main()
