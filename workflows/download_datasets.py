"""Show dataset acquisition status and manual source tips.

No dataset has an admitted automatic download. Every named selector is
disabled and fails with manual-acquisition guidance before network or model
framework imports. The default ``all`` selection is an intentional no-op.

Usage:
    uv run python workflows/download_datasets.py
    uv run python workflows/download_datasets.py --dataset dogfacenet
    uv run python workflows/download_datasets.py --dataset yt-bb-dog
    uv run python workflows/download_datasets.py --list

Data root resolution:
    1. ``--data-root``
    2. ``CANINE_IDENTITY_DATA_DIR`` when set before startup
    3. ``~/canine_identity_data``
"""

from __future__ import annotations

import argparse
from pathlib import Path

from contracts.model_paths import DATA_DIR, SUPPORTED_DATASETS


class ManualAcquisitionRequired(RuntimeError):
    """Raised when a disabled selector requires manual acquisition."""


_DATASETS: dict[str, dict[str, str]] = {
    "dogfacenet": {
        "mode": "disabled/manual",
        "description": "DogFaceNet from Hugging Face",
        "source": "https://huggingface.co/datasets/dimidagd/DogFaceNet_224resize",
        "reason": (
            "the source tip is not pinned to a revision, content hash, and "
            "license receipt"
        ),
    },
    "ap10k-dog": {
        "mode": "disabled/manual",
        "description": "AP-10K domestic dog subset",
        "source": "https://github.com/AlexTheBad/AP-10K",
        "reason": "no automatic acquisition contract is admitted",
    },
    "dogflw": {
        "mode": "disabled/manual",
        "description": "Dog Facial Landmarks in the Wild",
        "source": "https://www.kaggle.com/datasets/georgemartvel/dogflw",
        "reason": "no automatic acquisition contract is admitted",
    },
    "yt-bb-dog": {
        "mode": "disabled/manual",
        "description": "YT-BB-Dog",
        "source": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "reason": "no automatic acquisition contract is admitted",
    },
    "sibetan": {
        "mode": "disabled/manual",
        "description": "SiBeTan",
        "source": "https://www.lirmm.fr/YT-BB-Dog_Sibetan/",
        "reason": "no automatic acquisition contract is admitted",
    },
    "mpdd": {
        "mode": "disabled/manual",
        "description": "Multi-pose dog dataset (MPDD)",
        "source": "https://data.mendeley.com/datasets/v5j6m8dzhv/1",
        "reason": "no automatic acquisition contract is admitted",
    },
}


def _resolve_data_root(cli_root: str | None) -> Path:
    return Path(cli_root).expanduser() if cli_root else DATA_DIR


def _image_count(path: Path) -> int:
    if not path.exists():
        return 0
    return sum(
        entry.is_file() and entry.suffix.lower() in {".jpg", ".jpeg", ".png"}
        for entry in path.rglob("*")
    )


def _dataset_destination(name: str, data_root: Path) -> Path:
    try:
        directory = SUPPORTED_DATASETS[name]["dir"]
    except KeyError as exc:
        raise ValueError(f"Dataset metadata is missing for {name!r}") from exc
    return data_root / "datasets" / directory


def _list_datasets(data_root: Path) -> None:
    for name, dataset in _DATASETS.items():
        destination = _dataset_destination(name, data_root)
        count = _image_count(destination)
        local_status = f"{count} local images" if count else "not present locally"
        operation = (
            f"{dataset['mode']} acquisition only ({dataset['reason']}): "
            f"{dataset['source']}"
        )
        print(f"{name}: {operation}; {local_status}")


def download_dataset(name: str, data_root: Path) -> None:
    try:
        dataset = _DATASETS[name]
    except KeyError as exc:
        available = ", ".join(_DATASETS)
        raise ValueError(f"Unknown dataset {name!r}. Available: {available}") from exc

    raise ManualAcquisitionRequired(
        f"{dataset['description']} automatic acquisition is disabled because "
        f"{dataset['reason']}. Review the source and applicable terms at "
        f"{dataset['source']}, then place authorized material under "
        f"{_dataset_destination(name, data_root)}. No network request or directory "
        "creation was attempted."
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="all", choices=[*_DATASETS, "all"])
    parser.add_argument(
        "--data-root",
        default=None,
        help="Data root directory (default: CANINE_IDENTITY_DATA_DIR or ~/canine_identity_data)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="Show supported operations and local status",
    )
    args = parser.parse_args()

    data_root = _resolve_data_root(args.data_root)
    if args.list:
        _list_datasets(data_root)
        return

    if args.dataset == "all":
        print(
            "Intentional no-op: no dataset has an admitted automatic download; "
            "no network request or filesystem change was attempted."
        )
        return

    try:
        download_dataset(args.dataset, data_root)
    except ManualAcquisitionRequired as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
