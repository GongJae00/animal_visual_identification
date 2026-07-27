"""Export oracle crops from public dataset ZIP archives for the protected split.

Reads the protected split assignment + evaluator binding, maps each sample
token back to its source image via source_sample_id, extracts from the
appropriate ZIP archive, and writes organized crops for each protocol role.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import zipfile
from pathlib import Path

from cvi.source_provenance import build_offline_tool_provenance


class _ZipCache:
    def __init__(self, archive_config: dict[str, str]) -> None:
        self._outer: dict[Path, zipfile.ZipFile] = {}
        self._inner: dict[Path, dict[str, zipfile.ZipFile]] = {}
        for dataset, path_str in archive_config.items():
            path = Path(path_str)
            outer = zipfile.ZipFile(path)
            self._outer[path] = outer
            inners: dict[str, zipfile.ZipFile] = {}
            for name in outer.namelist():
                if name.endswith(".zip"):
                    inners[name] = zipfile.ZipFile(outer.open(name))
            self._inner[path] = inners

    def read(self, archive_path: Path, member_path: str) -> bytes:
        outer = self._outer.get(archive_path)
        if outer is None:
            raise KeyError(f"archive not cached: {archive_path}")
        try:
            return outer.read(member_path)
        except KeyError:
            pass
        for inner in self._inner.get(archive_path, {}).values():
            try:
                return inner.read(member_path)
            except KeyError:
                continue
        raise KeyError(f"member not found: {member_path}")

    def close(self) -> None:
        for inners in self._inner.values():
            for zf in inners.values():
                zf.close()
        for zf in self._outer.values():
            zf.close()


def _sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        while chunk := f.read(1 << 20):
            h.update(chunk)
    return h.hexdigest()


YT_MEMBER_RE = re.compile(
    r"^YT-BB-Dog/(?P<split>train|test)/(?P<identity>\d+)/"
    r"(?P=identity)_(?P<frame>\d+)\.jpg$"
)
YT_RANDOM_MEMBER_RE = re.compile(
    r"^YT-BB-Dog_random_bckg/YT-BB-Dog/test/(?P<identity>\d+)/"
    r"(?P=identity)_(?P<frame>\d+)\.jpg$"
)


def _locate_yt_path(source_sample_id: str) -> str:
    m = re.search(r"video-track:(\d+)", source_sample_id)
    identity = m.group(1) if m else "0"
    mf = re.search(r"frame:(\d+)", source_sample_id)
    frame = mf.group(1) if mf else "000000"
    if "random_background" in source_sample_id:
        return f"YT-BB-Dog_random_bckg/YT-BB-Dog/test/{identity}/{identity}_{frame}.jpg"
    split = "test" if int(identity) >= 2000 else "train"
    return f"YT-BB-Dog/{split}/{identity}/{identity}_{frame}.jpg"


def _locate_dogface_path(source_sample_id: str) -> str:
    m = re.search(r"web-folder:(\d+):image:(\d+)\.(\d+)", source_sample_id)
    if m:
        return f"after_4_bis/{m.group(1)}/{m.group(2)}.{m.group(3)}.jpg"
    raise ValueError(f"cannot locate DogFaceNet path: {source_sample_id}")


_MPDD_MEMBER_RE = re.compile(
    r"^MPDD/pytorch/(?P<split>train|val|query|gallery)/"
    r"(?P<identity>\d+)_c(?P<camera>\d+)_?s(?P<sequence>\d+)_"
    r"(?P<index>\d+)\.jpg$"
)


def _build_mpdd_lookup(archive_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            m = _MPDD_MEMBER_RE.match(name)
            if m:
                sid = (
                    f"mpdd:v1:device-capture:{m.group('identity')}:"
                    f"{m.group('split')}:c{m.group('camera')}:"
                    f"s{m.group('sequence')}:image:{m.group('index')}"
                )
                lookup[sid] = name
    return lookup


def _locate_mpdd_path(source_sample_id: str, lookup: dict[str, str] | None = None) -> str:
    if lookup is not None:
        path = lookup.get(source_sample_id)
        if path is not None:
            return path
    m = re.search(r"device-capture:(\d+):(\w+):c(\d+):s(\d+):image:(\d+)", source_sample_id)
    if m:
        identity, split, cam, seq, idx = m.group(1), m.group(2), m.group(3), m.group(4), m.group(5)
        return f"MPDD/pytorch/{split}/{identity}_c{cam}s{seq}_{idx}.jpg"
    raise ValueError(f"cannot locate MPDD path: {source_sample_id}")


_SIBETAN_MEMBER_RE = re.compile(
    r"^Sibetan/(?P<cluster>\d+)/"
    r"(?P<site>Indonesia)_(?P<camera>C\d{2})(?:[-_])"
    r"(?P<date>SibetanSept2019|sept2019|sep2019)_"
    r"(?P<media>\d{3}MEDIA)(?P<volume>_?2)?_DSCF"
    r"(?P<clip>\d{4})(?P<frame>\d*)\.jpg$"
)


def _build_sibetan_lookup(archive_path: Path) -> dict[str, str]:
    lookup: dict[str, str] = {}
    with zipfile.ZipFile(archive_path) as zf:
        for name in zf.namelist():
            m = _SIBETAN_MEMBER_RE.match(name)
            if m:
                cluster = int(m.group("cluster"))
                clip = m.group("clip")
                frame = m.group("frame") or "0"
                sid = f"sibetan:v1:sequence:{cluster}:clip:{clip}:frame:{frame}"
                lookup[sid] = name
    return lookup


def _locate_sibetan_path(source_sample_id: str, lookup: dict[str, str] | None = None) -> str:
    if lookup is not None:
        path = lookup.get(source_sample_id)
        if path is not None:
            return path
    m = re.search(r"sequence:(\d+):clip:(\d+):frame:(\d+)", source_sample_id)
    if m:
        return f"Sibetan/{m.group(1)}/Indonesia_C01_sept2019_100MEDIA_DSCF{m.group(2)}{m.group(3)}.jpg"
    raise ValueError(f"cannot locate Sibetan path: {source_sample_id}")


def _find_member_path(
    dataset_name: str, source_sample_id: str,
    sibetan_lookup: dict[str, str] | None = None,
    mpdd_lookup: dict[str, str] | None = None,
) -> str:
    if dataset_name == "yt-bb-dog":
        return _locate_yt_path(source_sample_id)
    elif dataset_name == "dogfacenet224":
        return _locate_dogface_path(source_sample_id)
    elif dataset_name == "mpdd":
        return _locate_mpdd_path(source_sample_id, mpdd_lookup)
    elif dataset_name == "sibetan":
        return _locate_sibetan_path(source_sample_id, sibetan_lookup)
    raise ValueError(f"unknown dataset: {dataset_name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--assignment", required=True, type=Path)
    parser.add_argument("--labels", required=True, type=Path)
    parser.add_argument("--dataset-archives", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    parser.add_argument("--manifest-output", required=True, type=Path)
    args = parser.parse_args()

    parser.exit(
        status=2,
        message=(
            "Protected public crop export is disabled until the split receipt, "
            "evaluator-binding digest, and every exported crop content hash are "
            "verified and atomically published.\n"
        ),
    )

    assignment = json.loads(args.assignment.read_text())
    labels = json.loads(args.labels.read_text())
    archive_config = json.loads(args.dataset_archives.read_text())

    labels_by_token = {r["sample_token"]: r for r in labels["records"]}

    sibetan_archive = archive_config.get("sibetan")
    sibetan_lookup: dict[str, str] | None = None
    if sibetan_archive:
        sibetan_lookup = _build_sibetan_lookup(Path(sibetan_archive))
    mpdd_archive = archive_config.get("mpdd")
    mpdd_lookup: dict[str, str] | None = None
    if mpdd_archive:
        mpdd_lookup = _build_mpdd_lookup(Path(mpdd_archive))

    zip_cache = _ZipCache(archive_config)
    out_dir = Path(args.output_directory)
    out_dir.mkdir(parents=True, exist_ok=True)

    records: list[dict] = []
    try:
        for rec in assignment.get("records", []):
            token = rec["sample_token"]
            label = labels_by_token.get(token)
            if label is None:
                continue
            source_id = label.get("source_sample_id", "")
            dataset_name = label.get("dataset_identity_id", "").split(":")[0]
            if dataset_name not in archive_config:
                continue

            archive_path = Path(archive_config[dataset_name])
            uses = rec.get("uses", [])

            for use in uses:
                protocol = use["protocol"]
                shot = use["shot"]
                use_role = use["role"]
                gallery_size = use.get("gallery_size")

                rel_parts = [f"protocol={protocol}", f"shot={shot}", f"role={use_role}"]
                if gallery_size:
                    rel_parts.append(f"gallery_size={gallery_size}")
                rel_dir = "/".join(rel_parts)
                sample_rel = f"{rel_dir}/{token}.jpg"
                dest = out_dir / sample_rel
                if dest.exists():
                    records.append({
                        "sample_token": token,
                        "source_sample_id": source_id,
                        "dataset_name": dataset_name,
                        "protocol": protocol,
                        "shot": shot,
                        "role": use_role,
                        "gallery_size": gallery_size,
                        "relative_path": sample_rel,
                        "status": "EXISTS",
                    })
                    continue

                dest.parent.mkdir(parents=True, exist_ok=True)

                member_path = _find_member_path(dataset_name, source_id, sibetan_lookup, mpdd_lookup)
                try:
                    image_bytes = zip_cache.read(archive_path, member_path)
                except KeyError as exc:
                    records.append({
                        "sample_token": token,
                        "source_sample_id": source_id,
                        "dataset_name": dataset_name,
                        "protocol": protocol,
                        "shot": shot,
                        "role": use_role,
                        "gallery_size": gallery_size,
                        "relative_path": sample_rel,
                        "status": "MISSING",
                        "error": str(exc),
                    })
                    continue

                digest = hashlib.sha256(image_bytes).hexdigest()
                dest.write_bytes(image_bytes)

                records.append({
                    "sample_token": token,
                    "source_sample_id": source_id,
                    "dataset_name": dataset_name,
                    "protocol": protocol,
                    "shot": shot,
                    "role": use_role,
                    "gallery_size": gallery_size,
                    "relative_path": sample_rel,
                    "source_sha256": digest,
                    "member_path": member_path,
                    "archive_path": str(archive_path),
                    "status": "EXPORTED",
                })
    finally:
        zip_cache.close()

    manifest = {
        "schema_version": "cvi.public_frozen_experiment_crop_manifest.v1",
        "assignment_sha256": assignment.get("assignment_sha256", ""),
        "record_count": len(records),
        "exported_count": sum(1 for r in records if r["status"] == "EXPORTED"),
        "exists_count": sum(1 for r in records if r["status"] == "EXISTS"),
        "missing_count": sum(1 for r in records if r["status"] == "MISSING"),
        "records": records,
        "tool_provenance": build_offline_tool_provenance(Path(__file__)),
    }
    args.manifest_output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest_output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )

    print(
        json.dumps(
            {
                "status": "EXPORTED",
                "record_count": len(records),
                "exported_count": manifest["exported_count"],
                "exists_count": manifest["exists_count"],
                "missing_count": manifest["missing_count"],
                "output_directory": str(args.output_directory),
                "manifest_output": str(args.manifest_output),
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
