"""Ablation study for multi-head identity search.

Measures per-evidence and combined retrieval accuracy on a gallery/query
split.  Produces a structured JSON report suitable for comparison across
training runs and fusion strategies.

Usage:
  uv run python tools/ablate_identity_search.py \\
      --model model.onnx \\
      --index-dir ./idx \\
      --queries queries.json \\
      --output report.json

  queries.json format:
  [
      {"image": "path/to/crop.jpg", "registered_dog_id": "uuid"},
      ...
  ]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from cvi.search_engine import SearchEngine


_EVIDENCE_KEYS = ["visual", "texture", "structural"]


def _eval_split(engine: SearchEngine, queries: list[dict],
                top_k: int, weights: tuple[float, float, float] | None,
                label: str) -> dict:
    ranks: list[int] = []
    top1_hits = 0
    top5_hits = 0
    top10_hits = 0
    total = len(queries)

    by_evidence = {k: {"top1": 0, "total": total} for k in _EVIDENCE_KEYS}
    per_evidence_ranks: dict[str, list[int]] = {k: [] for k in _EVIDENCE_KEYS}

    for q in queries:
        rid = q["registered_dog_id"]
        img = Image.open(q["image"]).convert("RGB")
        resp = engine.search(img, top_k=top_k, fusion_weights=weights)
        found = False
        for rank, m in enumerate(resp.matches):
            if m.registered_dog_id == rid:
                ranks.append(rank)
                if rank == 0:
                    top1_hits += 1
                if rank < 5:
                    top5_hits += 1
                if rank < 10:
                    top10_hits += 1
                found = True
                for ek in _EVIDENCE_KEYS:
                    ev = getattr(m.evidence, ek)
                    if rank == 0 and ev == max(
                        getattr(m.evidence, ek2) for ek2 in _EVIDENCE_KEYS
                    ):
                        by_evidence[ek]["top1"] += 1
                break
        if not found:
            ranks.append(top_k)

    for ek in _EVIDENCE_KEYS:
        by_evidence[ek]["rate"] = (
            by_evidence[ek]["top1"] / max(by_evidence[ek]["total"], 1)
        )

    result = {
        "label": label,
        "weights": list(weights) if weights else None,
        "top_k": top_k,
        "total_queries": total,
        "top1_accuracy": top1_hits / max(total, 1),
        "top5_accuracy": top5_hits / max(total, 1),
        "top10_accuracy": top10_hits / max(total, 1),
        "mean_rank": (sum(ranks) / max(len(ranks), 1)) if ranks else None,
        "per_evidence_win_rate": by_evidence,
    }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--index-dir", required=True)
    parser.add_argument("--queries", required=True, type=Path)
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=50)
    args = parser.parse_args()

    queries = json.loads(args.queries.read_text())

    engine = SearchEngine(Path(args.model), Path(args.index_dir))
    print(json.dumps({"event": "ablation_start", "index_size": engine.size,
                       "queries": len(queries)}))

    results = []

    for ek in _EVIDENCE_KEYS:
        w = [0.0, 0.0, 0.0]
        w[_EVIDENCE_KEYS.index(ek)] = 1.0
        r = _eval_split(engine, queries, args.top_k, tuple(w), f"{ek}_only")
        results.append(r)

    for pair in [("visual+texture", 0.5, 0.5, 0.0),
                 ("visual+structural", 0.5, 0.0, 0.5),
                 ("texture+structural", 0.0, 0.5, 0.5)]:
        w = (pair[1], pair[2], pair[3])
        r = _eval_split(engine, queries, args.top_k, w, pair[0])
        results.append(r)

    r = _eval_split(engine, queries, args.top_k, None, "all_fused")
    results.append(r)

    report = {
        "model": args.model,
        "index_size": engine.size,
        "query_count": len(queries),
        "results": results,
    }
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    print(json.dumps({"event": "ablation_done", "output": str(args.output)}))
    engine.close()


if __name__ == "__main__":
    main()
