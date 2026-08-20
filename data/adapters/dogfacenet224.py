"""DogFaceNet 224 resized web-album crops.

Canonical folder ``dogfacenet224``. Publisher tree:

    after_4_bis/<id>/<filename_id>.<index>.jpg
    classes_train.txt
    classes_test.txt

``after_4_bis/<id>/`` is one web-album identity; ``<id>`` is a digit folder
name. The image basename prefix ``<filename_id>`` is not the identity
(published members can differ). ``after_4_bis`` itself, class files, empty
folders, and non-image files are not identities. Publisher class files are
authenticated identity-disjoint split lists (integer folder ids, multiplicity
matches extracted images) and default to the dataset root.

Used by identification training.
"""

from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

from data.public_sources.public_canine_manifest import (
    _read_published_class_file,
)
from data.types import (
    CaptureGroupKind,
    UnifiedCanidSample,
)
from shared.contracts.identity_ids import (
    compute_registered_dog_id,
    compute_sample_token,
)
from data.adapters.io import (
    _file_sha256,
    _image_dims,
    _verified_path,
    _verified_regular_file,
)


def adapt_dogfacenet224(
    data_root: Path,
    *,
    classes_train_path: Path | None = None,
    classes_test_path: Path | None = None,
) -> tuple[UnifiedCanidSample, ...]:
    """DogFaceNet 224 web-folder crops with the authenticated publisher split."""

    root = Path(os.path.abspath(os.fspath(data_root)))
    base = root / "after_4_bis"
    if not base.is_dir():
        raise FileNotFoundError(f"DogFaceNet base not found: {base}")
    if (classes_train_path is None) != (classes_test_path is None):
        raise ValueError("DogFace train and test class paths must be provided together")
    if classes_train_path is None:
        classes_train_path = _verified_path(root, "classes_train.txt")
        classes_test_path = _verified_path(root, "classes_test.txt")
    else:
        classes_train_path = _verified_regular_file(
            classes_train_path, "DogFace train class file"
        )
        classes_test_path = _verified_regular_file(
            classes_test_path, "DogFace test class file"
        )
    import data.adapters as adapters

    train_values, _, _ = _read_published_class_file(
        classes_train_path,
        expected_sha256=adapters.DOGFACE_TRAIN_SHA256,
        expected_md5=adapters.DOGFACE_TRAIN_MD5,
    )
    test_values, _, _ = _read_published_class_file(
        classes_test_path,
        expected_sha256=adapters.DOGFACE_TEST_SHA256,
        expected_md5=adapters.DOGFACE_TEST_MD5,
    )
    train_counts = Counter(train_values)
    test_counts = Counter(test_values)
    if set(train_counts) & set(test_counts):
        raise ValueError("DogFace class-file identity partition differs")

    images_by_identity: list[tuple[str, tuple[Path, ...]]] = []
    observed_counts: Counter[int] = Counter()
    for identity_dir in sorted(base.iterdir(), key=lambda path: path.name):
        if not identity_dir.is_dir():
            continue
        folder_id = identity_dir.name
        if not folder_id.isdigit():
            raise ValueError(f"DogFaceNet identity folder differs: {identity_dir}")
        image_files = tuple(
            image_file
            for image_file in sorted(identity_dir.iterdir())
            if image_file.suffix.lower() in (".jpg", ".jpeg", ".png")
        )
        if image_files:
            images_by_identity.append((folder_id, image_files))
            observed_counts[int(folder_id)] = len(image_files)
    if set(train_counts) | set(test_counts) != set(observed_counts):
        raise ValueError("DogFace class-file identity partition differs")
    if train_counts + test_counts != observed_counts:
        raise ValueError("DogFace class-file multiplicities differ from extracted images")

    split_by_identity = {
        **{identity: "train" for identity in train_counts},
        **{identity: "test" for identity in test_counts},
    }
    result: list[UnifiedCanidSample] = []
    for folder_id, image_files in images_by_identity:
        dataset_identity = f"dogfacenet224:v1:web-folder:{folder_id}"
        reg_id = compute_registered_dog_id(dataset_identity)
        for image_file in image_files:
            image_file = _verified_path(root, image_file.relative_to(root).as_posix())
            sample_id = compute_sample_token(
                f"dogfacenet224:{folder_id}:{image_file.stem}"
            )
            width, height = _image_dims(image_file)
            sha = _file_sha256(image_file)
            relative = str(image_file.relative_to(root))
            result.append(
                UnifiedCanidSample(
                    sample_id=sample_id,
                    dataset_name="dogfacenet224",
                    dataset_version="zenodo-12578449-v1",
                    source_group_id=folder_id,
                    image_path=relative,
                    image_sha256=sha,
                    width=width,
                    height=height,
                    registered_identity_id=reg_id,
                    raw_identity_id=folder_id,
                    capture_group_id=folder_id,
                    capture_group_kind=CaptureGroupKind.ALBUM_OR_SOURCE_GROUP,
                    split_role=split_by_identity[int(folder_id)],
                )
            )
    return tuple(result)


__all__ = ["adapt_dogfacenet224"]
