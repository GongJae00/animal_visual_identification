"""Download and prepare public datasets for CVI training.

Downloads verified public datasets to the data directory.

Usage:
    uv run python tools/download_datasets.py                    # download all available
    uv run python tools/download_datasets.py --dataset dogfacenet  # single dataset
    uv run python tools/download_datasets.py --list                # show status

Data root resolution (in priority order):
    1. --data-root CLI argument
    2. CVI_DATA_DIR environment variable
    3. ~/cvi_data (symlink or directory)
    4. data/ in project root

Public datasets:
    dogfacenet  — DogFaceNet_224resize from HuggingFace (8,363 imgs, 1,393 dogs)
                  Public, no auth needed.
    yt-bb-dog   — YT-BB-Dog from LIRMM (27,036 imgs, 2,723 dogs)
                  Public, direct download.
    sibetan     — SiBeTan from LIRMM (1,755 imgs, 59 dogs)
                  Public, direct download.
    mpdd        — MPetDoorDataset from Mendeley (1,657 imgs)
                  Requires signed license agreement — manual download.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

import requests
from cvi.model_paths import DATA_DIR, DATASETS_DIR, SUPPORTED_DATASETS


def _resolve_data_root(cli_root: str | None) -> Path:
    if cli_root:
        return Path(cli_root)
    return DATA_DIR


def _download_url(url: str, dest: Path, desc: str = "") -> None:
    dest.mkdir(parents=True, exist_ok=True)
    if dest.exists() and any(dest.iterdir()):
        print(f"  [OK] {desc} -- already exists")
        return
    print(f"  [DOWN] {desc}")
    print(f"    URL: {url}")
    print(f"    수동 다운로드 후 압축해제: {dest}")
    print(f"    (자동 다운로드는 추후 지원 예정)")


def _dogfacenet_download(output_dir: Path, *, variant: str = "224resize") -> None:
    import io
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download
    from PIL import Image

    repo_map = {
        "224resize": ("dimidagd/DogFaceNet_224resize", ["data/train-00000-of-00001.parquet"]),
        "large": ("dimidagd/DogFaceNet_large", [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ]),
    }
    repo_id, files = repo_map[variant]
    dest = output_dir

    existing = list(dest.rglob("*.jpg")) if dest.exists() else []
    if existing:
        print(f"  [OK] DogFaceNet ({variant}) — already {len(existing)} images in {dest}")
        return

    print(f"  [DOWN] DogFaceNet ({variant}) from {repo_id} ...")
    dest.mkdir(parents=True, exist_ok=True)

    total_saved = 0
    for parquet_file in files:
        p = hf_hub_download(repo_id, parquet_file, repo_type="dataset")
        t = pq.read_table(p)
        images_col = t.column("image")
        labels_col = t.column("label")

        for i in range(t.num_rows):
            img_info = images_col[i].as_py()
            label = labels_col[i].as_py()

            img_bytes = img_info["bytes"]
            img_path_rel = img_info["path"]

            dog_dir = dest / str(label)
            dog_dir.mkdir(parents=True, exist_ok=True)

            if img_path_rel and img_path_rel != "None":
                fname = Path(img_path_rel).name
                if not fname.lower().endswith((".jpg", ".jpeg", ".png")):
                    fname = f"{fname}.jpg"
            else:
                fname = f"{i:06d}.jpg"

            out_path = dog_dir / fname
            if out_path.exists():
                out_path = dog_dir / f"{i:06d}.jpg"

            try:
                img = Image.open(io.BytesIO(img_bytes))
                img = img.convert("RGB")
                img.save(out_path, "JPEG", quality=95)
                total_saved += 1
            except Exception as e:
                print(f"    [WARN] skip {i}: {e}")

    print(f"    {total_saved} images saved to {dest}")
    summary = {
        "dataset": f"DogFaceNet ({variant})",
        "source": repo_id,
        "total_images": total_saved,
        "unique_dogs": len(set(os.listdir(dest))),
        "layout": "dest/{dog_id}/{image}.jpg",
    }
    (dest / "_download_info.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n"
    )


_DATASET_HANDLERS: dict[str, dict] = {
    "dogfacenet": {
        "fn": _dogfacenet_download,
        "kwargs": {"variant": "224resize"},
        "desc": "DogFaceNet 224x224 (8,363 imgs, 1,393 dogs)",
        "auth": False,
    },
    "dogfacenet-large": {
        "fn": _dogfacenet_download,
        "kwargs": {"variant": "large"},
        "desc": "DogFaceNet large (more images)",
        "auth": False,
    },
    "yt-bb-dog": {
        "fn": lambda dest: _download_url(
            "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
            dest,
            "YT-BB-Dog (27,036 images)",
        ),
        "kwargs": {},
        "desc": "YT-BB-Dog short-term re-id (27,036 imgs, 2,723 dogs)",
        "auth": False,
    },
    "sibetan": {
        "fn": lambda dest: _download_url(
            "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
            dest,
            "SiBeTan long-term re-id (1,755 imgs, 59 dogs)",
        ),
        "kwargs": {},
        "desc": "SiBeTan cross-camera long-term re-id (1,755 imgs, 59 dogs)",
        "auth": False,
    },
    "mpdd": {
        "fn": lambda dest: _download_url(
            "https://github.com/hacilab/MPDD",
            dest,
            "MPetDoorDataset (1,657 imgs) — 라이선스 동의 필요",
        ),
        "kwargs": {},
        "desc": "MPetDoorDataset (1,657 imgs) — 라이선스 동의 필요",
        "auth": True,
    },
}


def _list_datasets(data_root: Path) -> None:
    for name, info in _DATASET_HANDLERS.items():
        dataset_info = SUPPORTED_DATASETS.get(name)
        if not dataset_info:
            continue
        dest = data_root / "datasets" / dataset_info["dir"]
        count = len(list(dest.rglob("*.jpg"))) if dest.exists() else 0
        auth = "token" if info["auth"] else "공개"
        status = f"{count} images" if count else "미다운로드"
        print(f"  {name:20s}  [{auth:5s}]  {info['desc']:55s}  {status}")


def download_dataset(name: str, data_root: Path) -> None:
    info = _DATASET_HANDLERS.get(name)
    if info is None:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(_DATASET_HANDLERS)}")
        return
    dataset_info = SUPPORTED_DATASETS.get(name)
    if not dataset_info:
        print(f"Dataset '{name}' not in SUPPORTED_DATASETS mapping")
        return
    dest = data_root / "datasets" / dataset_info["dir"]
    dest.mkdir(parents=True, exist_ok=True)
    info["fn"](dest, **info["kwargs"])


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=list(_DATASET_HANDLERS) + ["all"])
    parser.add_argument("--data-root", default=None,
                        help="Data root directory (default: CVI_DATA_DIR or ~/cvi_data)")
    parser.add_argument("--list", action="store_true")
    args = parser.parse_args()

    data_root = _resolve_data_root(args.data_root)
    print(f"Data root: {data_root}")

    if args.list:
        _list_datasets(data_root)
        return

    names = list(_DATASET_HANDLERS) if args.dataset == "all" else [args.dataset]
    t0 = time.time()
    for name in names:
        download_dataset(name, data_root)
    print(f"\nDone in {time.time()-t0:.1f}s")


if __name__ == "__main__":
    main()
