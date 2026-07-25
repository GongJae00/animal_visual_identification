"""Download and prepare all public datasets for CVI training.

Downloads verified public datasets to the data directory and converts
them to the oracle-crop format expected by the training pipeline.

Usage:
    python tools/download_datasets.py                    # download all available
    python tools/download_datasets.py --dataset dogfacenet  # single dataset
    python tools/download_datasets.py --list                # show status

Data root resolution (in priority order):
    1. --data-root CLI argument
    2. CVI_DATA_DIR environment variable
    3. ~/cvi_data (symlink or directory)
    4. data/ in project root

Supported datasets (verified working):
    dogfacenet  — DogFaceNet_224resize from HuggingFace (8,363 imgs, 1,393 dogs)
                  Public, no auth needed.
    dogfacenet-large — DogFaceNet_large from HuggingFace (more images)
                       Public, no auth needed.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from pathlib import Path

from cvi.model_paths import DATA_DIR, DATA_RAW_DIR, SUPPORTED_DATASETS


def _resolve_data_root(cli_root: str | None) -> Path:
    """Resolve the data root directory."""
    if cli_root:
        return Path(cli_root)
    return DATA_DIR


def _dogfacenet_download(output_dir: Path, *, variant: str = "224resize") -> None:
    """Download DogFaceNet from HuggingFace and convert to oracle crop layout.

    Creates: {output_dir}/dogfacenet/{dog_id}/{idx}.jpg
    """
    import io

    import numpy as np
    import pyarrow.parquet as pq
    from huggingface_hub import hf_hub_download, list_repo_files
    from PIL import Image

    repo_map = {
        "224resize": ("dimidagd/DogFaceNet_224resize", ["data/train-00000-of-00001.parquet"]),
        "large": ("dimidagd/DogFaceNet_large", [
            "data/train-00000-of-00002.parquet",
            "data/train-00001-of-00002.parquet",
        ]),
    }
    repo_id, files = repo_map[variant]
    dest = output_dir / "dogfacenet" if variant == "224resize" else output_dir / "dogfacenet_large"

    # Check if already downloaded (count .jpg files)
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

            # Extract image bytes from HF Image struct
            img_bytes = img_info["bytes"]
            img_path_rel = img_info["path"]

            # Save as JPEG under dest/{label}/{idx}.jpg
            dog_dir = dest / str(label)
            dog_dir.mkdir(parents=True, exist_ok=True)

            # Use original path as filename if possible, else index
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
    # Write a summary
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
}


def _list_datasets(data_root: Path) -> None:
    for name, info in _DATASET_HANDLERS.items():
        dest = data_root / "raw" / name.replace("-large", "_large")
        count = len(list(dest.rglob("*.jpg"))) if dest.exists() else 0
        auth = "token 필요" if info["auth"] else "공개"
        status = f"{count} images" if count else "미다운로드"
        print(f"  {name:20s}  [{auth}]  {info['desc']:50s}  {status}")


def download_dataset(name: str, data_root: Path) -> None:
    info = _DATASET_HANDLERS.get(name)
    if info is None:
        print(f"Unknown dataset: {name}")
        print(f"Available: {list(_DATASET_HANDLERS)}")
        return
    raw_dir = data_root / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    info["fn"](raw_dir, **info["kwargs"])


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
