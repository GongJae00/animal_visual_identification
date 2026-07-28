"""Audit per-pixel duplicates within and across all admitted canid datasets."""

from __future__ import annotations

import json
from pathlib import Path

from cvi.canid_data.adapters import ADAPTERS
from cvi.canid_data.duplicates import find_cross_dataset_duplicates, find_exact_duplicates
from cvi.canid_data.source_lock import admitted_records


def main() -> None:
    samples_by_dataset: dict[str, tuple] = {}
    roots: dict[str, Path] = {}
    for record in admitted_records():
        adapter = ADAPTERS.get(record.canonical_name)
        if adapter is None:
            continue
        root = Path(record.data_root)
        if not root.is_dir():
            continue
        samples_by_dataset[record.canonical_name] = adapter(root)
        roots[record.canonical_name] = root

    report: dict = {"within_dataset": {}, "cross_dataset": {}}
    for name, samples in samples_by_dataset.items():
        duplicates = find_exact_duplicates(samples, roots[name])
        report["within_dataset"][name] = {
            "groups": len(duplicates),
            "total_duplicate_samples": sum(len(g) for g in duplicates.values()),
        }

    cross = find_cross_dataset_duplicates(samples_by_dataset, roots)
    report["cross_dataset"] = {
        "groups": len(cross),
        "datasets_involved": sorted(
            {entry[0] for entries in cross.values() for entry in entries}
        ),
    }
    print(json.dumps(report, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()
