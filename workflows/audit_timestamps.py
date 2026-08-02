"""Stream a source timestamp audit without materializing per-frame metadata."""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from data_pipeline.acquisition import audit_video_timestamps


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    parser.add_argument("--expected-fps", required=True, type=float)
    parser.add_argument("--level", choices=("frame", "packet"), default="frame")
    args = parser.parse_args()

    result = audit_video_timestamps(
        args.source.resolve(strict=True),
        expected_fps=args.expected_fps,
        level=args.level,
    )
    print(json.dumps(asdict(result), sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
