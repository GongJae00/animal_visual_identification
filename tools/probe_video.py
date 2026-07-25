"""Create a G0 raw-video record without copying or decoding the source."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from cvi.acquisition import (
    ModalityInterval,
    ModalityState,
    RawVideoRecord,
    probe_video_file,
    sha256_file,
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--source-id", required=True)
    parser.add_argument("--camera-id", required=True)
    parser.add_argument("--cage-id", required=True)
    parser.add_argument("--camera-setting-version", required=True)
    parser.add_argument("--recording-start-ns", required=True, type=int)
    parser.add_argument(
        "--modality",
        choices=[state.value for state in ModalityState],
        default=ModalityState.UNKNOWN.value,
    )
    args = parser.parse_args()

    source = args.source.resolve(strict=True)
    probe = probe_video_file(source)
    duration_ns = round(probe.duration_seconds * 1_000_000_000)
    recording_end_ns = args.recording_start_ns + duration_ns
    record = RawVideoRecord(
        source_id=args.source_id,
        source_uri=str(source),
        source_sha256=sha256_file(source),
        byte_size=source.stat().st_size,
        camera_id=args.camera_id,
        cage_id=args.cage_id,
        camera_setting_version=args.camera_setting_version,
        recording_start_ns=args.recording_start_ns,
        recording_end_ns=recording_end_ns,
        probe=probe,
        modality_intervals=(
            ModalityInterval(
                args.recording_start_ns,
                recording_end_ns,
                ModalityState(args.modality),
            ),
        ),
    )
    print(json.dumps(record.to_dict(), ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
