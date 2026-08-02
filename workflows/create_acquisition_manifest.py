"""Create a G0 acquisition manifest from camera specs and raw video records."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from data_pipeline.acquisition import (
    AcquisitionManifest,
    CameraSpecification,
    RawVideoRecord,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera-spec", required=True, type=Path, action="append")
    parser.add_argument("--raw-video-record", required=True, type=Path, action="append")
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    cameras = []
    for spec_path in args.camera_spec:
        payload = json.loads(spec_path.resolve(strict=True).read_text(encoding="utf-8"))
        cameras.append(CameraSpecification.from_dict(payload))

    videos = []
    for record_path in args.raw_video_record:
        payload = json.loads(record_path.resolve(strict=True).read_text(encoding="utf-8"))
        videos.append(RawVideoRecord.from_dict(payload))

    manifest = AcquisitionManifest(
        cameras=tuple(cameras),
        videos=tuple(videos),
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest.to_dict(), ensure_ascii=False, sort_keys=True, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    main()