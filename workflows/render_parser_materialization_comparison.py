"""Render deterministic paired v5/v6 parser materialization contact sheets."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
from collections import defaultdict
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image, ImageDraw, ImageFont

from foundation.protected_io import read_strict_json_document, write_private_json_bundle
from foundation.protected_publication import fsync_directory, rename_directory_noreplace
from foundation.provenance import content_sha256
from identity_methods.full_segment.materialization import read_route_plan_bundle
from identity_methods.full_segment.route_plan import validate_full128_route_plan_bundle
from workflows.render_parser_failure_review import (
    _FONT,
    _FONT_BOLD,
    _annotated_tile,
    _load_review_sample,
)

_CATEGORIES = ("TERMINAL_TO_BODY_PARSING", "BODY_PARSING_CROP_CHANGED")
_PAIR_SIZE = (300, 230)
_COLUMNS = 2
_ROWS = 4
_PAGE_SIZE = _COLUMNS * _ROWS


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--comparison", type=Path, required=True)
    parser.add_argument("--baseline-route-plan", type=Path, required=True)
    parser.add_argument("--candidate-route-plan", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--samples-per-category", type=int, default=24)
    args = parser.parse_args()
    manifest = render(
        comparison_path=args.comparison,
        baseline_route_plan=args.baseline_route_plan,
        candidate_route_plan=args.candidate_route_plan,
        output_dir=args.output_dir,
        samples_per_category=args.samples_per_category,
    )
    print(
        json.dumps(
            {
                "status": "CREATED_PARSER_MATERIALIZATION_COMPARISON_REVIEW",
                "output": str(args.output_dir),
                "review_sha256": manifest["review_sha256"],
                "page_count": len(manifest["pages"]),
                "selected_sample_count": len(manifest["selected"]),
            },
            sort_keys=True,
        )
    )
    return 0


def render(
    *,
    comparison_path: Path,
    baseline_route_plan: Path,
    candidate_route_plan: Path,
    output_dir: Path,
    samples_per_category: int,
) -> dict[str, Any]:
    if output_dir.exists() or output_dir.is_symlink():
        raise FileExistsError(output_dir)
    if (
        isinstance(samples_per_category, bool)
        or not isinstance(samples_per_category, int)
        or samples_per_category <= 0
    ):
        raise ValueError("samples_per_category must be positive")
    comparison_document = read_strict_json_document(
        comparison_path, maximum_bytes=134_217_728
    )
    comparison = comparison_document.payload
    comparison_payload = {
        key: value for key, value in comparison.items() if key != "comparison_sha256"
    }
    if (
        comparison.get("schema_version") != "cvi.parser_materialization_comparison.v1"
        or comparison.get("comparison_sha256") != content_sha256(comparison_payload)
    ):
        raise ValueError("parser comparison report differs")
    baseline_bundle = _plan(baseline_route_plan)
    candidate_bundle = _plan(candidate_route_plan)
    _validate_binding(comparison["baseline"], baseline_bundle)
    _validate_binding(comparison["candidate"], candidate_bundle)
    baseline_rows = {row["sample_token"]: row for row in baseline_bundle["plan"]["records"]}
    candidate_rows = {row["sample_token"]: row for row in candidate_bundle["plan"]["records"]}
    baseline_root = Path(comparison["baseline"]["materialization_root"]).resolve(strict=True)
    candidate_root = Path(comparison["candidate"]["materialization_root"]).resolve(strict=True)
    selected = {}
    for category in _CATEGORIES:
        tokens = comparison["changed_sample_tokens"][category]
        selected[category] = _balanced_tokens(
            tokens,
            baseline_rows,
            min(samples_per_category, len(tokens)),
            category,
        )
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(
        tempfile.mkdtemp(prefix=f".{output_dir.name}.staging-", dir=output_dir.parent)
    )
    try:
        pages = []
        manifest_rows = []
        for category in _CATEGORIES:
            tokens = selected[category]
            for page_index, start in enumerate(range(0, len(tokens), _PAGE_SIZE), start=1):
                page_tokens = tokens[start : start + _PAGE_SIZE]
                pairs = []
                for token in page_tokens:
                    baseline = _load_review_sample(
                        baseline_rows[token],
                        baseline_root,
                        parser_policy_sha256=baseline_bundle["parser_policy_sha256"],
                    )
                    candidate = _load_review_sample(
                        candidate_rows[token],
                        candidate_root,
                        parser_policy_sha256=candidate_bundle["parser_policy_sha256"],
                    )
                    pairs.append((baseline, candidate))
                    manifest_rows.append(_manifest_row(category, baseline, candidate))
                filename = f"{category.lower()}_{page_index:02d}.png"
                path = staging / filename
                _render_page(path, category, page_index, pairs)
                payload = path.read_bytes()
                pages.append(
                    {
                        "relative_path": filename,
                        "category": category,
                        "page_index": page_index,
                        "sample_tokens": list(page_tokens),
                        "byte_size": len(payload),
                        "sha256": hashlib.sha256(payload).hexdigest(),
                    }
                )
        body = {
            "schema_version": "cvi.parser_materialization_comparison_review.v1",
            "comparison_sha256": comparison["comparison_sha256"],
            "baseline_route_plan_sha256": baseline_bundle["plan_sha256"],
            "candidate_route_plan_sha256": candidate_bundle["plan_sha256"],
            "selection_method": (
                "BALANCED_BY_DATASET_THEN_SHA256_OF_CATEGORY_DATASET_AND_SAMPLE_TOKEN"
            ),
            "samples_per_category": samples_per_category,
            "selected": manifest_rows,
            "pages": pages,
            "interpretation": (
                "PAIRED_MODEL_OUTPUT_REVIEW_ONLY;LEFT_BASELINE_RIGHT_CANDIDATE;"
                "NOT_GROUND_TRUTH"
            ),
        }
        manifest = {**body, "review_sha256": content_sha256(body)}
        write_private_json_bundle(((staging / "review-manifest.json", manifest),))
        fsync_directory(staging)
        rename_directory_noreplace(staging, output_dir)
        fsync_directory(output_dir.parent)
    except BaseException:
        shutil.rmtree(staging, ignore_errors=True)
        raise
    return manifest


def _plan(path: Path) -> dict[str, Any]:
    return validate_full128_route_plan_bundle(
        read_route_plan_bundle(path.absolute()), verify_files=False
    )


def _validate_binding(binding: object, bundle: dict[str, Any]) -> None:
    if not isinstance(binding, dict) or any(
        binding.get(binding_field) != bundle[bundle_field]
        for binding_field, bundle_field in (
            ("route_plan_sha256", "plan_sha256"),
            ("route_policy_sha256", "route_policy_sha256"),
            ("parser_runtime_manifest_sha256", "parser_runtime_manifest_sha256"),
            ("parser_policy_sha256", "parser_policy_sha256"),
        )
    ):
        raise ValueError("parser comparison route-plan binding differs")


def _balanced_tokens(
    tokens: Sequence[str],
    rows: Mapping[str, Mapping[str, Any]],
    count: int,
    category: str,
) -> list[str]:
    by_dataset: defaultdict[str, list[str]] = defaultdict(list)
    for token in tokens:
        by_dataset[rows[token]["dataset_name"]].append(token)
    for dataset, values in by_dataset.items():
        values.sort(
            key=lambda token: hashlib.sha256(
                f"{category}\0{dataset}\0{token}".encode("ascii")
            ).hexdigest()
        )
    selected = []
    while len(selected) < count:
        progressed = False
        for dataset in sorted(by_dataset):
            if by_dataset[dataset]:
                selected.append(by_dataset[dataset].pop(0))
                progressed = True
                if len(selected) == count:
                    break
        if not progressed:
            break
    return selected


def _render_page(
    path: Path,
    category: str,
    page_index: int,
    pairs: Sequence[tuple[Mapping[str, Any], Mapping[str, Any]]],
) -> None:
    pair_width = _PAIR_SIZE[0] * 2 + 28
    pair_height = _PAIR_SIZE[1] + 92
    canvas = Image.new(
        "RGB",
        (24 + _COLUMNS * pair_width, 100 + _ROWS * pair_height),
        (10, 16, 28),
    )
    draw = ImageDraw.Draw(canvas)
    title_font = ImageFont.truetype(_FONT_BOLD, 24)
    label_font = ImageFont.truetype(_FONT_BOLD, 13)
    small_font = ImageFont.truetype(_FONT, 12)
    draw.text((24, 16), category, font=title_font, fill=(244, 247, 252))
    draw.text(
        (24, 52),
        f"page {page_index} | left=v5 route-v2 | right=v6 route-v3",
        font=small_font,
        fill=(148, 163, 184),
    )
    for index, (baseline, candidate) in enumerate(pairs):
        row_index, column = divmod(index, _COLUMNS)
        x = 24 + column * pair_width
        y = 92 + row_index * pair_height
        left = _annotated_tile(
            baseline["source"], baseline["prediction"], baseline["row"]["route_evidence"]
        )
        right = _annotated_tile(
            candidate["source"], candidate["prediction"], candidate["row"]["route_evidence"]
        )
        canvas.paste(left, (x, y))
        canvas.paste(right, (x + _PAIR_SIZE[0] + 8, y))
        token = baseline["row"]["sample_token"]
        dataset = baseline["row"]["dataset_name"]
        draw.text(
            (x, y + _PAIR_SIZE[1] + 7),
            f"{dataset} | {token[:12]}",
            font=label_font,
            fill=(226, 232, 240),
        )
        draw.text(
            (x, y + _PAIR_SIZE[1] + 28),
            f"v5 {_receipt_state(baseline['receipt'])}",
            font=small_font,
            fill=(248, 113, 113),
        )
        draw.text(
            (x, y + _PAIR_SIZE[1] + 47),
            f"v6 {_receipt_state(candidate['receipt'])}",
            font=small_font,
            fill=(74, 222, 128),
        )
    canvas.save(path, format="PNG", compress_level=9, optimize=False)


def _receipt_state(receipt: Mapping[str, Any]) -> str:
    return (
        "BODY_PARSING"
        if receipt["actual_route"] == "BODY_PARSING"
        else str(receipt["terminal_reason"])
    )


def _manifest_row(
    category: str,
    baseline: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> dict[str, Any]:
    payload = {
        "category": category,
        "sample_token": baseline["row"]["sample_token"],
        "dataset_name": baseline["row"]["dataset_name"],
        "source_sha256": baseline["row"]["source_sha256"],
        "baseline_state": _receipt_state(baseline["receipt"]),
        "candidate_state": _receipt_state(candidate["receipt"]),
        "baseline_prediction_sha256": baseline["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
        "candidate_prediction_sha256": candidate["receipt"]["parser_lineage"][
            "prediction_sha256"
        ],
    }
    return {**payload, "record_sha256": content_sha256(payload)}


if __name__ == "__main__":
    raise SystemExit(main())
