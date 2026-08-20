from __future__ import annotations

from pathlib import Path

import pytest
from PIL import Image

from data.adapters.io import file_sha256
from data.adapters.oxford_pets_dog import adapt_oxford_pets_dog
from shared.contracts.identity_ids import compute_sample_token


def _write_oxford_sample(
    root: Path,
    stem: str,
    *,
    head_xml: str | None = None,
) -> None:
    image_dir = root / "images"
    trimap_dir = root / "annotations" / "trimaps"
    xml_dir = root / "annotations" / "xmls"
    image_dir.mkdir(parents=True, exist_ok=True)
    trimap_dir.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (40, 30)).save(image_dir / f"{stem}.jpg")
    Image.new("L", (40, 30), color=1).save(trimap_dir / f"{stem}.png")
    if head_xml is not None:
        xml_dir.mkdir(parents=True, exist_ok=True)
        (xml_dir / f"{stem}.xml").write_text(head_xml, encoding="utf-8")


def _write_splits(
    root: Path,
    *,
    trainval: str,
    test: str = "",
) -> None:
    annotation_root = root / "annotations"
    annotation_root.mkdir(parents=True, exist_ok=True)
    (annotation_root / "trainval.txt").write_text(trainval, encoding="utf-8")
    (annotation_root / "test.txt").write_text(test, encoding="utf-8")


def _valid_head_xml(stem: str) -> str:
    return (
        "<annotation>"
        f"<filename>{stem}.jpg</filename>"
        "<object><name>dog</name><pose>Frontal</pose>"
        "<truncated>0</truncated><occluded>0</occluded><difficult>0</difficult>"
        "<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax>"
        "<ymax>22</ymax></bndbox></object></annotation>"
    )


def test_oxford_pets_dog_fails_closed_when_publisher_base_is_missing(
    tmp_path: Path,
) -> None:
    with pytest.raises(FileNotFoundError, match="Oxford-IIIT Pet base not found"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_official_split_is_missing(
    tmp_path: Path,
) -> None:
    (tmp_path / "images").mkdir()
    (tmp_path / "annotations").mkdir()
    (tmp_path / "annotations" / "trainval.txt").write_text("", encoding="utf-8")

    with pytest.raises(FileNotFoundError, match="Oxford official split not found"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_on_split_row_schema(tmp_path: Path) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    _write_splits(tmp_path, trainval="beagle_1 3 2\n")

    with pytest.raises(ValueError, match="Oxford split row schema differs"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_on_split_labels(tmp_path: Path) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    _write_splits(tmp_path, trainval="beagle_1 3 3 2\n")

    with pytest.raises(ValueError, match="Oxford split labels differ"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_splits_repeat_image(tmp_path: Path) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    _write_splits(
        tmp_path,
        trainval="beagle_1 3 2 2\n",
        test="beagle_1 3 2 2\n",
    )

    with pytest.raises(ValueError, match="Oxford official splits repeat image"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_dog_image_name_differs(
    tmp_path: Path,
) -> None:
    _write_oxford_sample(tmp_path, "beagle")
    _write_splits(tmp_path, trainval="beagle 3 2 2\n")

    with pytest.raises(ValueError, match="Oxford dog image name differs"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_official_splits_contain_no_dogs(
    tmp_path: Path,
) -> None:
    (tmp_path / "images").mkdir()
    _write_splits(tmp_path, trainval="Abyssinian_1 1 1 1\n")

    with pytest.raises(
        ValueError, match="Oxford official splits contain no dog images"
    ):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_required_image_is_missing(
    tmp_path: Path,
) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    (tmp_path / "images" / "beagle_1.jpg").unlink()
    _write_splits(tmp_path, trainval="beagle_1 3 2 2\n")

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_oxford_pets_dog(tmp_path)


@pytest.mark.parametrize(
    ("xml_body", "match"),
    (
        (
            "<!DOCTYPE annotation [<!ENTITY x 'x'>]>" + _valid_head_xml("beagle_1"),
            "unsafe Oxford head annotation",
        ),
        (
            _valid_head_xml("other_1"),
            "Oxford head annotation filename differs",
        ),
        (
            "<not-xml",
            "Oxford head annotation is malformed",
        ),
        (
            (
                "<annotation><filename>beagle_1.jpg</filename>"
                "<object><name>cat</name>"
                "<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>20</xmax>"
                "<ymax>22</ymax></bndbox></object></annotation>"
            ),
            "Oxford dog head annotation schema differs",
        ),
        (
            (
                "<annotation><filename>beagle_1.jpg</filename>"
                "<object><name>dog</name></object></annotation>"
            ),
            "Oxford dog head box is missing",
        ),
        (
            (
                "<annotation><filename>beagle_1.jpg</filename>"
                "<object><name>dog</name>"
                "<bndbox><xmin>x</xmin><ymin>3</ymin><xmax>20</xmax>"
                "<ymax>22</ymax></bndbox></object></annotation>"
            ),
            "Oxford dog head box is malformed",
        ),
        (
            (
                "<annotation><filename>beagle_1.jpg</filename>"
                "<object><name>dog</name>"
                "<bndbox><xmin>2</xmin><ymin>3</ymin><xmax>99</xmax>"
                "<ymax>22</ymax></bndbox></object></annotation>"
            ),
            "Oxford dog head box is out of bounds",
        ),
    ),
)
def test_oxford_pets_dog_fails_closed_on_invalid_head_xml(
    tmp_path: Path,
    xml_body: str,
    match: str,
) -> None:
    _write_oxford_sample(tmp_path, "beagle_1", head_xml=xml_body)
    _write_splits(tmp_path, trainval="beagle_1 3 2 2\n")

    with pytest.raises(ValueError, match=match):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_head_xml_is_a_symlink(
    tmp_path: Path,
) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    xml_dir = tmp_path / "annotations" / "xmls"
    xml_dir.mkdir(parents=True)
    target = tmp_path / "outside.xml"
    target.write_text(_valid_head_xml("beagle_1"), encoding="utf-8")
    (xml_dir / "beagle_1.xml").symlink_to(target)
    _write_splits(tmp_path, trainval="beagle_1 3 2 2\n")

    with pytest.raises(ValueError, match="not a regular file"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_fails_closed_when_head_annotation_exceeds_size_limit(
    tmp_path: Path,
) -> None:
    xml_dir = tmp_path / "annotations" / "xmls"
    xml_dir.mkdir(parents=True)
    oversized = xml_dir / "beagle_1.xml"
    oversized.write_bytes(b"<annotation>" + (b"x" * (1024 * 1024)))
    _write_oxford_sample(tmp_path, "beagle_1")
    _write_splits(tmp_path, trainval="beagle_1 3 2 2\n")

    with pytest.raises(ValueError, match="Oxford head annotation exceeds size limit"):
        adapt_oxford_pets_dog(tmp_path)


def test_oxford_pets_dog_ignores_list_txt_and_binds_official_sample_hash(
    tmp_path: Path,
) -> None:
    _write_oxford_sample(tmp_path, "beagle_1")
    _write_splits(tmp_path, trainval="beagle_1 3 2 2\n")
    (tmp_path / "annotations" / "list.txt").write_text(
        "this is not a valid catalog row\n",
        encoding="utf-8",
    )

    samples = adapt_oxford_pets_dog(tmp_path)

    assert len(samples) == 1
    sample = samples[0]
    image_path = tmp_path / "images" / "beagle_1.jpg"
    assert sample.sample_id == compute_sample_token(
        "oxford-pets-dog:official:trainval:beagle_1"
    )
    assert sample.dataset_name == "oxford-pets-dog"
    assert sample.dataset_version == "publisher-splits-v1"
    assert sample.source_group_id == "beagle_1"
    assert sample.image_path == "images/beagle_1.jpg"
    assert sample.image_sha256 == file_sha256(image_path)
    assert sample.foreground_mask_path == "annotations/trimaps/beagle_1.png"
    assert sample.metadata["class_id"] == 3
    assert sample.metadata["species_id"] == 2
    assert sample.metadata["breed_id"] == 2
    assert sample.raw_identity_id is None
    assert sample.registered_identity_id is None
    assert sample.label_availability["identity"] is False
