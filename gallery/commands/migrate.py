"""Migrate an immutable v3 gallery into a separate v4 directory."""

from __future__ import annotations

import argparse
from pathlib import Path

from gallery.migration.v3_to_v4 import migrate_gallery


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    arguments = parser.parse_args()
    migrate_gallery(arguments.source, arguments.output)


if __name__ == "__main__":
    main()
