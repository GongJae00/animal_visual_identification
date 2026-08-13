"""Inspect all admitted canid datasets and print per-dataset statistics."""

from __future__ import annotations

import json
from pathlib import Path

from data.adapters import ADAPTERS
from data.duplicates import find_exact_duplicates, summarize_duplicates
from data.report import compute_dataset_statistics
from data.source_lock import SOURCE_REGISTRY, admitted_records


def main() -> None:
    output: dict[str, dict] = {}
    for record in admitted_records():
        name = record.canonical_name
        adapter = ADAPTERS.get(name)
        if adapter is None:
            output[name] = {"error": "no adapter", "admission": record.admission.value}
            continue
        root = Path(record.data_root)
        if not root.is_dir():
            output[name] = {
                "error": "data root not found",
                "root": record.data_root,
                "admission": record.admission.value,
            }
            continue
        samples = adapter(root)
        stats = compute_dataset_statistics(samples)
        dup_summary = summarize_duplicates(samples, root)
        output[name] = {
            "admission": record.admission.value,
            "capture_kind": record.capture_group_kind.value,
            "license": record.license_id,
            "statistics": stats,
            "duplicates": dup_summary,
        }

    print(json.dumps(output, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
