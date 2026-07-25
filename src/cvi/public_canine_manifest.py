"""Semantic manifests for the four audited public canine ZIP datasets.

This module intentionally reads only ZIP central directories and the small
Sibetan ground-truth JSON files.  It never decodes an image, creates a split,
or tests image-content duplication.  Member paths, identity labels, camera
tokens, sequence tokens, and split labels are provenance/audit metadata only;
they MUST NOT be supplied to a visual model.  Accordingly, manifest records
expose an empty :attr:`PublicCanineRecord.visual_model_input_fields` tuple.
"""

from __future__ import annotations

import hashlib
import json
import re
import stat
import zipfile
from collections import Counter
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path, PurePosixPath


class IdentitySemantics(StrEnum):
    """What a source label actually denotes; none means registered identity."""

    WEB_FOLDER = "WEB_FOLDER"
    DEVICE_CAPTURE = "DEVICE_CAPTURE"
    GT_JSON = "GT_JSON"
    VIDEO_TRACK = "VIDEO_TRACK"


class CanineRegion(StrEnum):
    FACE = "FACE"
    DOG_CROP = "DOG_CROP"


@dataclass(frozen=True, slots=True)
class PublicCanineSchemaPolicy:
    """Exact audited semantic cardinalities, independent of image payloads."""

    dataset_name: str
    image_count: int
    identity_count: int
    split_image_counts: tuple[tuple[str, int], ...] = ()
    split_identity_counts: tuple[tuple[str, int], ...] = ()

    def __post_init__(self) -> None:
        _require_token(self.dataset_name, "dataset_name")
        for name in ("image_count", "identity_count"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                raise ValueError(f"{name} must be a positive integer")
        for name in ("split_image_counts", "split_identity_counts"):
            values = getattr(self, name)
            keys: list[str] = []
            for key, value in values:
                _require_token(key, f"{name} key")
                if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
                    raise ValueError(f"{name} values must be positive integers")
                keys.append(key)
            if keys != sorted(keys) or len(keys) != len(set(keys)):
                raise ValueError(f"{name} must be sorted with unique keys")


@dataclass(frozen=True, slots=True)
class ArchiveReceiptBinding:
    """Bind semantic parsing to an already audited source archive receipt."""

    dataset_name: str
    archive_sha256: str
    archive_receipt_sha256: str
    schema_version: str = "cvi.archive_receipt_binding.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.archive_receipt_binding.v1":
            raise ValueError("unsupported archive receipt binding")
        _require_token(self.dataset_name, "dataset_name")
        _require_sha256(self.archive_sha256, "archive_sha256")
        _require_sha256(self.archive_receipt_sha256, "archive_receipt_sha256")


@dataclass(frozen=True, slots=True)
class PublicCanineRecord:
    """One provenance-only image member; no field is a visual model input."""

    dataset_name: str
    dataset_version: str
    source_variant: str
    source_sample_id: str
    dataset_identity_id: str
    identity_semantics: IdentitySemantics
    region: CanineRegion
    original_split: str | None
    sequence_id: str | None
    camera_token: str | None
    camera_token_verified: bool
    filename_identity_token: str | None
    source_cluster_id: int | None
    in_no_mono_subset: bool | None
    paired_source_sample_id: str | None
    member_path: str
    member_crc32: int
    member_uncompressed_bytes: int
    source_archive_sha256: str
    source_archive_receipt_sha256: str
    container_member_path: str | None = None
    container_member_crc32: int | None = None
    container_member_uncompressed_bytes: int | None = None
    schema_version: str = "cvi.public_canine_record.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_record.v1":
            raise ValueError("unsupported public canine record")
        for name in (
            "dataset_name",
            "dataset_version",
            "source_variant",
            "source_sample_id",
            "dataset_identity_id",
        ):
            _require_token(getattr(self, name), name, maximum=1024)
        if not self.dataset_identity_id.startswith(f"{self.dataset_name}:"):
            raise ValueError("identity ID is not dataset-namespaced")
        if not self.source_sample_id.startswith(f"{self.dataset_name}:"):
            raise ValueError("sample ID is not dataset-namespaced")
        if not isinstance(self.identity_semantics, IdentitySemantics):
            raise TypeError("identity_semantics must be IdentitySemantics")
        if not isinstance(self.region, CanineRegion):
            raise TypeError("region must be CanineRegion")
        _require_member_path(self.member_path)
        _require_crc32(self.member_crc32, "member_crc32")
        _require_nonnegative_int(
            self.member_uncompressed_bytes, "member_uncompressed_bytes"
        )
        _require_sha256(self.source_archive_sha256, "source_archive_sha256")
        _require_sha256(
            self.source_archive_receipt_sha256,
            "source_archive_receipt_sha256",
        )
        if self.camera_token is not None:
            _require_token(self.camera_token, "camera_token")
        if not isinstance(self.camera_token_verified, bool):
            raise TypeError("camera_token_verified must be boolean")
        if self.camera_token_verified:
            raise ValueError("public filename camera tokens are unverified")
        for name in (
            "original_split",
            "sequence_id",
            "filename_identity_token",
            "paired_source_sample_id",
        ):
            value = getattr(self, name)
            if value is not None:
                _require_token(value, name, maximum=1024)
        if self.source_cluster_id is not None:
            _require_nonnegative_int(self.source_cluster_id, "source_cluster_id")
        if self.in_no_mono_subset is not None and not isinstance(
            self.in_no_mono_subset, bool
        ):
            raise TypeError("in_no_mono_subset must be boolean or null")
        nested = self.container_member_path is not None
        if nested:
            _require_member_path(self.container_member_path or "")
            _require_crc32(self.container_member_crc32, "container_member_crc32")
            _require_nonnegative_int(
                self.container_member_uncompressed_bytes,
                "container_member_uncompressed_bytes",
            )
        elif (
            self.container_member_crc32 is not None
            or self.container_member_uncompressed_bytes is not None
        ):
            raise ValueError("partial nested-container provenance")

    @property
    def visual_model_input_fields(self) -> tuple[()]:
        """Return no metadata fields: only separately decoded pixels may be input."""

        return ()


@dataclass(frozen=True, slots=True)
class PublicCanineManifest:
    dataset_name: str
    dataset_version: str
    source_archive_sha256: str
    source_archive_receipt_sha256: str
    records: tuple[PublicCanineRecord, ...]
    schema_version: str = "cvi.public_canine_manifest.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.public_canine_manifest.v1":
            raise ValueError("unsupported public canine manifest")
        _require_token(self.dataset_name, "dataset_name")
        _require_token(self.dataset_version, "dataset_version")
        _require_sha256(self.source_archive_sha256, "source_archive_sha256")
        _require_sha256(
            self.source_archive_receipt_sha256,
            "source_archive_receipt_sha256",
        )
        if not isinstance(self.records, tuple) or not self.records:
            raise ValueError("public canine manifest must not be empty")
        sample_ids: set[str] = set()
        for record in self.records:
            if not isinstance(record, PublicCanineRecord):
                raise TypeError("manifest records must be PublicCanineRecord")
            if (
                record.dataset_name != self.dataset_name
                or record.dataset_version != self.dataset_version
                or record.source_archive_sha256 != self.source_archive_sha256
                or record.source_archive_receipt_sha256
                != self.source_archive_receipt_sha256
            ):
                raise ValueError("record differs from manifest binding")
            if record.source_sample_id in sample_ids:
                raise ValueError("duplicate source sample ID")
            sample_ids.add(record.source_sample_id)

    @property
    def image_count(self) -> int:
        return len(self.records)

    @property
    def identity_count(self) -> int:
        return len({record.dataset_identity_id for record in self.records})


@dataclass(frozen=True, slots=True)
class DogFaceClassSplitReceipt:
    train_sha256: str
    train_md5: str
    train_lines: int
    train_identities: int
    test_sha256: str
    test_md5: str
    test_lines: int
    test_identities: int
    identity_intersection: int
    schema_version: str = "cvi.dogface_class_split_receipt.v1"

    def __post_init__(self) -> None:
        if self.schema_version != "cvi.dogface_class_split_receipt.v1":
            raise ValueError("unsupported DogFace class split receipt")
        _require_sha256(self.train_sha256, "train_sha256")
        _require_sha256(self.test_sha256, "test_sha256")
        _require_md5(self.train_md5, "train_md5")
        _require_md5(self.test_md5, "test_md5")
        for name in (
            "train_lines",
            "train_identities",
            "test_lines",
            "test_identities",
            "identity_intersection",
        ):
            _require_nonnegative_int(getattr(self, name), name)


@dataclass(frozen=True, slots=True)
class DogFaceManifestResult:
    manifest: PublicCanineManifest
    basename_identity_mismatches: int
    class_split_receipt: DogFaceClassSplitReceipt | None


@dataclass(frozen=True, slots=True)
class SibetanManifestResult:
    manifest: PublicCanineManifest
    cluster_count: int
    gt_identity_count: int
    no_mono_cluster_count: int
    no_mono_identity_count: int
    no_mono_image_count: int


@dataclass(frozen=True, slots=True)
class YtBbDogManifestResult:
    original: PublicCanineManifest
    random_background: PublicCanineManifest
    paired_test_images: int
    missing_random_background_images: int


DOGFACE_DATASET = "dogfacenet224"
MPDD_DATASET = "mpdd"
SIBETAN_DATASET = "sibetan"
YT_DATASET = "yt-bb-dog"

DOGFACE_TRAIN_SHA256 = (
    "ee37ec14c661a9dba49f4c45dd5d77618bcb0b8678af7a96a7ca7b75b79ba510"
)
DOGFACE_TRAIN_MD5 = "67f7bc3aecb967e43aa2585a1a4879d6"
DOGFACE_TEST_SHA256 = (
    "39132675b9b1371a09e55bf16d2fba733f489a9bec97312d7164e8fe384f6ad1"
)
DOGFACE_TEST_MD5 = "0afc77355916e658a00b050979534ad7"

_DOGFACE_RE = re.compile(
    r"^after_4_bis/(?P<folder>\d+)/(?P<filename_id>\d+)\.(?P<index>\d+)\.jpg$"
)
_MPDD_RE = re.compile(
    r"^MPDD/pytorch/(?P<split>train|val|query|gallery)/"
    r"(?P<identity>\d+)_c(?P<camera>\d+)_?s(?P<sequence>\d+)_"
    r"(?P<index>\d+)\.jpg$"
)
_SIBETAN_RE = re.compile(
    r"^Sibetan/(?P<cluster>\d+)/"
    r"(?P<site>Indonesia)_(?P<camera>C\d{2})(?:[-_])"
    r"(?P<date>SibetanSept2019|sept2019|sep2019)_"
    r"(?P<media>\d{3}MEDIA)(?P<volume>_?2)?_DSCF"
    r"(?P<clip>\d{4})(?P<frame>\d*)\.jpg$"
)
_YT_RE = re.compile(
    r"^YT-BB-Dog/(?P<split>train|test)/(?P<identity>\d+)/"
    r"(?P=identity)_(?P<frame>\d+)\.jpg$"
)
_YT_RANDOM_RE = re.compile(
    r"^YT-BB-Dog_random_bckg/YT-BB-Dog/test/(?P<identity>\d+)/"
    r"(?P=identity)_(?P<frame>\d+)\.jpg$"
)


def parse_dogfacenet224(
    *,
    archive_path: Path,
    binding: ArchiveReceiptBinding,
    classes_train_path: Path | None = None,
    classes_test_path: Path | None = None,
) -> DogFaceManifestResult:
    """Parse the audited 224-resized DogFaceNet web-folder archive."""

    _verify_archive_binding(archive_path, binding, DOGFACE_DATASET)
    if (classes_train_path is None) != (classes_test_path is None):
        raise ValueError("both DogFace class files are required together")
    with zipfile.ZipFile(archive_path) as bundle:
        infos = _regular_infos(bundle)
    image_infos = tuple(info for info in infos if info.filename.lower().endswith(".jpg"))
    if len(image_infos) != DOGFACE_POLICY.image_count:
        raise ValueError("DogFaceNet image count differs from audited policy")

    parsed: list[tuple[zipfile.ZipInfo, re.Match[str]]] = []
    folder_counts: Counter[int] = Counter()
    mismatches = 0
    for info in image_infos:
        match = _fullmatch(_DOGFACE_RE, info.filename, "DogFaceNet member")
        folder = int(match["folder"])
        folder_counts[folder] += 1
        mismatches += folder != int(match["filename_id"])
        parsed.append((info, match))
    if len(folder_counts) != DOGFACE_POLICY.identity_count or mismatches != 17:
        raise ValueError("DogFaceNet identity/mismatch policy differs")

    split_by_identity: dict[int, str] = {}
    split_receipt: DogFaceClassSplitReceipt | None = None
    if classes_train_path is not None and classes_test_path is not None:
        train_values, train_sha, train_md5 = _read_published_class_file(
            classes_train_path,
            expected_sha256=DOGFACE_TRAIN_SHA256,
            expected_md5=DOGFACE_TRAIN_MD5,
        )
        test_values, test_sha, test_md5 = _read_published_class_file(
            classes_test_path,
            expected_sha256=DOGFACE_TEST_SHA256,
            expected_md5=DOGFACE_TEST_MD5,
        )
        train_counts, test_counts = Counter(train_values), Counter(test_values)
        if (len(train_values), len(train_counts)) != (7_666, 1_254):
            raise ValueError("DogFace train class-file policy differs")
        if (len(test_values), len(test_counts)) != (697, 139):
            raise ValueError("DogFace test class-file policy differs")
        intersection = set(train_counts) & set(test_counts)
        if intersection or set(train_counts) | set(test_counts) != set(folder_counts):
            raise ValueError("DogFace class-file identity partition differs")
        if train_counts + test_counts != folder_counts:
            raise ValueError("DogFace class-file multiplicities differ from archive")
        split_by_identity.update((identity, "train") for identity in train_counts)
        split_by_identity.update((identity, "test") for identity in test_counts)
        split_receipt = DogFaceClassSplitReceipt(
            train_sha256=train_sha,
            train_md5=train_md5,
            train_lines=len(train_values),
            train_identities=len(train_counts),
            test_sha256=test_sha,
            test_md5=test_md5,
            test_lines=len(test_values),
            test_identities=len(test_counts),
            identity_intersection=0,
        )

    records = tuple(
        PublicCanineRecord(
            dataset_name=DOGFACE_DATASET,
            dataset_version="224resized-v1",
            source_variant="original",
            source_sample_id=(
                f"{DOGFACE_DATASET}:v1:web-folder:{match['folder']}:"
                f"image:{match['filename_id']}.{match['index']}"
            ),
            dataset_identity_id=(
                f"{DOGFACE_DATASET}:v1:web-folder:{match['folder']}"
            ),
            identity_semantics=IdentitySemantics.WEB_FOLDER,
            region=CanineRegion.FACE,
            original_split=split_by_identity.get(int(match["folder"])),
            sequence_id=None,
            camera_token=None,
            camera_token_verified=False,
            filename_identity_token=match["filename_id"],
            source_cluster_id=None,
            in_no_mono_subset=None,
            paired_source_sample_id=None,
            member_path=info.filename,
            member_crc32=info.CRC,
            member_uncompressed_bytes=info.file_size,
            source_archive_sha256=binding.archive_sha256,
            source_archive_receipt_sha256=binding.archive_receipt_sha256,
        )
        for info, match in parsed
    )
    return DogFaceManifestResult(
        manifest=PublicCanineManifest(
            DOGFACE_DATASET,
            "224resized-v1",
            binding.archive_sha256,
            binding.archive_receipt_sha256,
            records,
        ),
        basename_identity_mismatches=mismatches,
        class_split_receipt=split_receipt,
    )


def parse_mpdd(
    *, archive_path: Path, binding: ArchiveReceiptBinding
) -> PublicCanineManifest:
    """Parse MPDD while treating c/s filename fields as unverified tokens."""

    _verify_archive_binding(archive_path, binding, MPDD_DATASET)
    with zipfile.ZipFile(archive_path) as bundle:
        infos = _regular_infos(bundle)
    image_infos = tuple(info for info in infos if info.filename.lower().endswith(".jpg"))
    if len(image_infos) != MPDD_POLICY.image_count:
        raise ValueError("MPDD image count differs from audited policy")
    records: list[PublicCanineRecord] = []
    split_counts: Counter[str] = Counter()
    split_identities: dict[str, set[int]] = {
        split: set() for split in ("train", "val", "query", "gallery")
    }
    anomaly_seen = False
    for info in image_infos:
        match = _fullmatch(_MPDD_RE, info.filename, "MPDD member")
        split, identity = match["split"], int(match["identity"])
        split_counts[split] += 1
        split_identities[split].add(identity)
        anomaly_seen |= info.filename == "MPDD/pytorch/query/146_c1_s3_1.jpg"
        records.append(
            PublicCanineRecord(
                dataset_name=MPDD_DATASET,
                dataset_version="v1",
                source_variant="original",
                source_sample_id=(
                    f"{MPDD_DATASET}:v1:device-capture:{identity}:"
                    f"{split}:c{match['camera']}:s{match['sequence']}:"
                    f"image:{match['index']}"
                ),
                dataset_identity_id=(
                    f"{MPDD_DATASET}:v1:device-capture:{identity}"
                ),
                identity_semantics=IdentitySemantics.DEVICE_CAPTURE,
                region=CanineRegion.FACE,
                original_split=split,
                sequence_id=(
                    f"{MPDD_DATASET}:v1:filename-sequence-token:"
                    f"{match['sequence']}"
                ),
                camera_token=f"c{match['camera']}",
                camera_token_verified=False,
                filename_identity_token=match["identity"],
                source_cluster_id=None,
                in_no_mono_subset=None,
                paired_source_sample_id=None,
                member_path=info.filename,
                member_crc32=info.CRC,
                member_uncompressed_bytes=info.file_size,
                source_archive_sha256=binding.archive_sha256,
                source_archive_receipt_sha256=binding.archive_receipt_sha256,
            )
        )
    expected_counts = Counter(dict(MPDD_POLICY.split_image_counts))
    expected_ids = dict(MPDD_POLICY.split_identity_counts)
    if split_counts != expected_counts or {
        split: len(ids) for split, ids in split_identities.items()
    } != expected_ids:
        raise ValueError("MPDD original split policy differs")
    all_ids = set().union(*split_identities.values())
    if len(all_ids) != MPDD_POLICY.identity_count or all_ids != set(range(191)):
        raise ValueError("MPDD identity policy differs")
    if split_identities["train"] != split_identities["val"]:
        raise ValueError("MPDD train/val identities differ")
    if split_identities["query"] != split_identities["gallery"]:
        raise ValueError("MPDD query/gallery identities differ")
    if split_identities["train"] & split_identities["query"]:
        raise ValueError("MPDD train/test identity leakage")
    if not anomaly_seen:
        raise ValueError("MPDD audited underscore anomaly is absent")
    return PublicCanineManifest(
        MPDD_DATASET,
        "v1",
        binding.archive_sha256,
        binding.archive_receipt_sha256,
        tuple(records),
    )


def parse_sibetan(
    *, archive_path: Path, binding: ArchiveReceiptBinding
) -> SibetanManifestResult:
    """Parse Sibetan with JSON dog identities and folder sequence clusters."""

    _verify_archive_binding(archive_path, binding, SIBETAN_DATASET)
    with zipfile.ZipFile(archive_path) as bundle:
        infos = _regular_infos(bundle)
        image_infos = tuple(
            info for info in infos if info.filename.lower().endswith(".jpg")
        )
        full_gt = _read_small_json(bundle, "Sibetan/gt_sibetan.json")
        no_mono_gt = _read_small_json(
            bundle, "Sibetan/gt_sibetan_no_mono_cluster.json"
        )
    if len(image_infos) != SIBETAN_POLICY.image_count:
        raise ValueError("Sibetan image count differs from audited policy")
    cluster_to_identity = _parse_cluster_gt(full_gt, "full")
    no_mono_cluster_to_identity = _parse_cluster_gt(no_mono_gt, "no-mono")
    if len(full_gt) != 59 or len(cluster_to_identity) != 223:
        raise ValueError("Sibetan full GT policy differs")
    if len(no_mono_gt) != 39 or len(no_mono_cluster_to_identity) != 203:
        raise ValueError("Sibetan no-mono GT policy differs")
    if any(
        cluster_to_identity.get(cluster) != identity
        for cluster, identity in no_mono_cluster_to_identity.items()
    ):
        raise ValueError("Sibetan no-mono GT is not an exact full-GT subset")

    records: list[PublicCanineRecord] = []
    observed_clusters: set[int] = set()
    normalized_sequences: set[str] = set()
    no_mono_images = 0
    for info in image_infos:
        match = _fullmatch(_SIBETAN_RE, info.filename, "Sibetan member")
        cluster = int(match["cluster"])
        if cluster not in cluster_to_identity:
            raise ValueError("Sibetan image cluster is absent from full GT")
        observed_clusters.add(cluster)
        volume = (match["volume"] or "").replace("_", "")
        frame = match["frame"] or "0"
        sequence_key = (
            f"{match['site']}:{match['camera']}:sept2019:"
            f"{match['media']}:{volume or '1'}:{match['clip']}"
        )
        normalized_sequences.add(sequence_key)
        in_no_mono = cluster in no_mono_cluster_to_identity
        no_mono_images += in_no_mono
        dog_identity = cluster_to_identity[cluster]
        records.append(
            PublicCanineRecord(
                dataset_name=SIBETAN_DATASET,
                dataset_version="v1",
                source_variant="original",
                source_sample_id=(
                    f"{SIBETAN_DATASET}:v1:sequence:{cluster}:"
                    f"clip:{match['clip']}:frame:{frame}"
                ),
                dataset_identity_id=(
                    f"{SIBETAN_DATASET}:v1:gt-json:{dog_identity}"
                ),
                identity_semantics=IdentitySemantics.GT_JSON,
                region=CanineRegion.DOG_CROP,
                original_split=None,
                sequence_id=f"{SIBETAN_DATASET}:v1:sequence:{cluster}",
                camera_token=match["camera"],
                camera_token_verified=False,
                filename_identity_token=None,
                source_cluster_id=cluster,
                in_no_mono_subset=in_no_mono,
                paired_source_sample_id=None,
                member_path=info.filename,
                member_crc32=info.CRC,
                member_uncompressed_bytes=info.file_size,
                source_archive_sha256=binding.archive_sha256,
                source_archive_receipt_sha256=binding.archive_receipt_sha256,
            )
        )
    if observed_clusters != set(cluster_to_identity) or len(normalized_sequences) != 223:
        raise ValueError("Sibetan cluster/sequence coverage differs")
    if no_mono_images != 1_603:
        raise ValueError("Sibetan no-mono image policy differs")
    manifest = PublicCanineManifest(
        SIBETAN_DATASET,
        "v1",
        binding.archive_sha256,
        binding.archive_receipt_sha256,
        tuple(records),
    )
    return SibetanManifestResult(manifest, 223, 59, 203, 39, 1_603)


def parse_yt_bb_dog(
    *, archive_path: Path, binding: ArchiveReceiptBinding
) -> YtBbDogManifestResult:
    """Parse both parent-bound nested YT-BB-Dog ZIP variants without extraction."""

    _verify_archive_binding(archive_path, binding, YT_DATASET)
    with zipfile.ZipFile(archive_path) as outer:
        outer_infos = {info.filename: info for info in _regular_infos(outer)}
        original_name = "YT-BB-dog/YT-BB-Dog.zip"
        random_name = "YT-BB-dog/YT-BB-Dog_random_bckg.zip"
        if set(outer_infos) != {original_name, random_name}:
            raise ValueError("YT-BB-Dog outer member policy differs")
        original_records = _parse_nested_yt(
            outer,
            outer_infos[original_name],
            binding,
            random_background=False,
        )
        random_records = _parse_nested_yt(
            outer,
            outer_infos[random_name],
            binding,
            random_background=True,
        )

    original_by_pair_key = {
        _yt_pair_key(record.member_path, random_background=False): record
        for record in original_records
        if record.original_split == "test"
    }
    random_by_pair_key = {
        _yt_pair_key(record.member_path, random_background=True): record
        for record in random_records
    }
    if not set(random_by_pair_key) <= set(original_by_pair_key):
        raise ValueError("YT random-background images are not a test subset")
    paired_random = tuple(
        _replace_pair(record, original_by_pair_key[key].source_sample_id)
        for key, record in random_by_pair_key.items()
    )
    if len(paired_random) != 7_064 or len(original_by_pair_key) != 7_104:
        raise ValueError("YT paired-background policy differs")
    original = PublicCanineManifest(
        YT_DATASET,
        "v1",
        binding.archive_sha256,
        binding.archive_receipt_sha256,
        tuple(original_records),
    )
    random_manifest = PublicCanineManifest(
        YT_DATASET,
        "v1",
        binding.archive_sha256,
        binding.archive_receipt_sha256,
        paired_random,
    )
    return YtBbDogManifestResult(original, random_manifest, 7_064, 40)


def _parse_nested_yt(
    outer: zipfile.ZipFile,
    container_info: zipfile.ZipInfo,
    binding: ArchiveReceiptBinding,
    *,
    random_background: bool,
) -> tuple[PublicCanineRecord, ...]:
    regex = _YT_RANDOM_RE if random_background else _YT_RE
    variant = "random_background" if random_background else "original"
    with outer.open(container_info) as nested_stream:
        with zipfile.ZipFile(nested_stream) as nested:
            infos = _regular_infos(nested)
    image_infos = tuple(info for info in infos if info.filename.lower().endswith(".jpg"))
    policy = YT_RANDOM_BACKGROUND_POLICY if random_background else YT_ORIGINAL_POLICY
    expected_images = policy.image_count
    if len(image_infos) != expected_images:
        raise ValueError(f"YT {variant} image count differs")
    split_counts: Counter[str] = Counter()
    split_ids: dict[str, set[int]] = {"train": set(), "test": set()}
    records: list[PublicCanineRecord] = []
    for info in image_infos:
        match = _fullmatch(regex, info.filename, f"YT {variant} member")
        split = "test" if random_background else match["split"]
        identity = int(match["identity"])
        split_counts[split] += 1
        split_ids[split].add(identity)
        records.append(
            PublicCanineRecord(
                dataset_name=YT_DATASET,
                dataset_version="v1",
                source_variant=variant,
                source_sample_id=(
                    f"{YT_DATASET}:v1:{variant}:video-track:{identity}:"
                    f"frame:{match['frame']}"
                ),
                dataset_identity_id=f"{YT_DATASET}:v1:video-track:{identity}",
                identity_semantics=IdentitySemantics.VIDEO_TRACK,
                region=CanineRegion.DOG_CROP,
                original_split=split,
                sequence_id=f"{YT_DATASET}:v1:video-track:{identity}",
                camera_token=None,
                camera_token_verified=False,
                filename_identity_token=match["identity"],
                source_cluster_id=None,
                in_no_mono_subset=None,
                paired_source_sample_id=None,
                member_path=info.filename,
                member_crc32=info.CRC,
                member_uncompressed_bytes=info.file_size,
                source_archive_sha256=binding.archive_sha256,
                source_archive_receipt_sha256=binding.archive_receipt_sha256,
                container_member_path=container_info.filename,
                container_member_crc32=container_info.CRC,
                container_member_uncompressed_bytes=container_info.file_size,
            )
        )
    if random_background:
        if split_counts != Counter(test=7_064) or split_ids["test"] != set(
            range(2_000, 2_723)
        ):
            raise ValueError("YT random-background identity policy differs")
    else:
        if split_counts != Counter(train=19_932, test=7_104):
            raise ValueError("YT original split count differs")
        if split_ids["train"] != set(range(2_000)) or split_ids["test"] != set(
            range(2_000, 2_723)
        ):
            raise ValueError("YT original identity partition differs")
    return tuple(records)


def _replace_pair(
    record: PublicCanineRecord, paired_source_sample_id: str
) -> PublicCanineRecord:
    values = {
        field: getattr(record, field)
        for field in PublicCanineRecord.__dataclass_fields__
    }
    values["paired_source_sample_id"] = paired_source_sample_id
    return PublicCanineRecord(**values)


def _yt_pair_key(path: str, *, random_background: bool) -> str:
    prefix = (
        "YT-BB-Dog_random_bckg/YT-BB-Dog/test/"
        if random_background
        else "YT-BB-Dog/test/"
    )
    if not path.startswith(prefix):
        raise ValueError("YT pair path prefix differs")
    return path.removeprefix(prefix)


def _read_published_class_file(
    path: Path, *, expected_sha256: str, expected_md5: str
) -> tuple[tuple[int, ...], str, str]:
    payload = path.read_bytes()
    sha256 = hashlib.sha256(payload).hexdigest()
    md5 = hashlib.md5(payload, usedforsecurity=False).hexdigest()
    if sha256 != expected_sha256 or md5 != expected_md5:
        raise ValueError("DogFace published class-file hash differs")
    try:
        text = payload.decode("ascii")
        values = tuple(int(line) for line in text.splitlines())
    except (UnicodeDecodeError, ValueError) as error:
        raise ValueError("DogFace class file is not ASCII integer lines") from error
    if not values or any(value < 0 for value in values):
        raise ValueError("DogFace class file contains invalid identity")
    return values, sha256, md5


def _read_small_json(bundle: zipfile.ZipFile, member_path: str) -> object:
    try:
        info = bundle.getinfo(member_path)
    except KeyError as error:
        raise ValueError(f"required JSON member is absent: {member_path}") from error
    if info.file_size > 1_000_000:
        raise ValueError("GT JSON exceeds semantic-parser byte limit")
    try:
        return json.loads(bundle.read(info))
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError("GT JSON is invalid") from error


def _parse_cluster_gt(payload: object, label: str) -> dict[int, int]:
    if not isinstance(payload, dict):
        raise ValueError(f"Sibetan {label} GT must be an object")
    inverse: dict[int, int] = {}
    for raw_identity, raw_clusters in payload.items():
        if not isinstance(raw_identity, str) or not raw_identity.isdigit():
            raise ValueError(f"Sibetan {label} GT identity differs")
        identity = int(raw_identity)
        if str(identity) != raw_identity or not isinstance(raw_clusters, list):
            raise ValueError(f"Sibetan {label} GT representation differs")
        if not raw_clusters:
            raise ValueError(f"Sibetan {label} GT identity has no clusters")
        for cluster in raw_clusters:
            _require_nonnegative_int(cluster, f"Sibetan {label} cluster")
            if cluster in inverse:
                raise ValueError(f"Sibetan {label} GT cluster is duplicated")
            inverse[cluster] = identity
    return inverse


def _regular_infos(bundle: zipfile.ZipFile) -> tuple[zipfile.ZipInfo, ...]:
    infos = bundle.infolist()
    names: set[str] = set()
    regular: list[zipfile.ZipInfo] = []
    for info in infos:
        _require_member_path(info.filename)
        if info.filename in names:
            raise ValueError("duplicate ZIP member path")
        names.add(info.filename)
        mode = info.external_attr >> 16
        if stat.S_ISLNK(mode):
            raise ValueError("ZIP symlink member is forbidden")
        if info.flag_bits & 0x1:
            raise ValueError("encrypted ZIP member is forbidden")
        if not info.is_dir():
            regular.append(info)
    return tuple(regular)


def _verify_archive_binding(
    archive_path: Path, binding: ArchiveReceiptBinding, dataset_name: str
) -> None:
    if binding.dataset_name != dataset_name:
        raise ValueError("archive binding dataset differs")
    digest = hashlib.sha256()
    with archive_path.open("rb") as stream:
        while chunk := stream.read(1_048_576):
            digest.update(chunk)
    if digest.hexdigest() != binding.archive_sha256:
        raise ValueError("archive bytes differ from receipt binding")


def _fullmatch(
    pattern: re.Pattern[str], value: str, label: str
) -> re.Match[str]:
    match = pattern.fullmatch(value)
    if match is None:
        raise ValueError(f"{label} path schema differs: {value}")
    return match


def _require_member_path(value: str) -> None:
    if not isinstance(value, str) or not value or "\\" in value:
        raise ValueError("ZIP member path is invalid")
    path = PurePosixPath(value)
    if path.is_absolute() or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError("ZIP member path is unsafe")


def _require_token(value: object, name: str, *, maximum: int = 256) -> None:
    if not isinstance(value, str) or not value or len(value) > maximum:
        raise ValueError(f"{name} must be bounded non-empty text")
    if any(ord(character) < 32 for character in value):
        raise ValueError(f"{name} contains a control character")


def _require_nonnegative_int(value: object, name: str) -> None:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")


def _require_crc32(value: object, name: str) -> None:
    _require_nonnegative_int(value, name)
    if value > 0xFFFFFFFF:  # type: ignore[operator]
        raise ValueError(f"{name} exceeds CRC-32")


def _require_sha256(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise ValueError(f"{name} must be lowercase SHA-256")


def _require_md5(value: object, name: str) -> None:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{32}", value) is None:
        raise ValueError(f"{name} must be lowercase MD5")


# These constants are constructed after validators are defined so their own
# strict dataclass invariants are checked at import time.
DOGFACE_POLICY = PublicCanineSchemaPolicy(DOGFACE_DATASET, 8_363, 1_393)
MPDD_POLICY = PublicCanineSchemaPolicy(
    MPDD_DATASET,
    1_657,
    191,
    split_image_counts=(
        ("gallery", 521),
        ("query", 104),
        ("train", 921),
        ("val", 111),
    ),
    split_identity_counts=(
        ("gallery", 96),
        ("query", 96),
        ("train", 95),
        ("val", 95),
    ),
)
SIBETAN_POLICY = PublicCanineSchemaPolicy(SIBETAN_DATASET, 1_755, 59)
YT_ORIGINAL_POLICY = PublicCanineSchemaPolicy(
    YT_DATASET,
    27_036,
    2_723,
    split_image_counts=(("test", 7_104), ("train", 19_932)),
    split_identity_counts=(("test", 723), ("train", 2_000)),
)
YT_RANDOM_BACKGROUND_POLICY = PublicCanineSchemaPolicy(
    YT_DATASET,
    7_064,
    723,
    split_image_counts=(("test", 7_064),),
    split_identity_counts=(("test", 723),),
)
