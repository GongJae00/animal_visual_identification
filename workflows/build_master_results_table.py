"""Generate a deterministic long-form master table only from sealed reports."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from evaluation.common_reporting import build_master_results_table
from foundation.protected_io import read_strict_json_object, write_private_json_bundle


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--report", action="append", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    table = build_master_results_table(
        [read_strict_json_object(path) for path in args.report]
    )
    write_private_json_bundle(((args.output, table),))
    print(
        json.dumps(
            {
                "event": "master_results_table_built",
                "output": str(args.output),
                "report_count": len(table["source_report_sha256s"]),
                "row_count": len(table["rows"]),
                "table_sha256": table["table_sha256"],
            },
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
