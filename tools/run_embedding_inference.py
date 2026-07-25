"""Run ONNX-based embedding inference against a gallery.

No PyTorch required — uses ONNX Runtime, numpy, and PIL.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

from PIL import Image

from cvi.inference import EmbeddingInferencePipeline, InferenceConfig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--gallery-embeddings", required=True, type=Path)
    parser.add_argument("--gallery-labels", type=Path, default=None)
    parser.add_argument("--registry-db", required=True, type=Path)
    parser.add_argument("--query", required=True, type=Path)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()

    if args.query.is_dir():
        query_paths = sorted(args.query.glob("*.jpg")) + sorted(args.query.glob("*.png"))
    elif args.query.is_file():
        query_paths = [args.query]
    else:
        print(json.dumps({"status": "ERROR", "error": f"query not found: {args.query}"}))
        raise SystemExit(1)

    config = InferenceConfig(
        model_path=str(args.model),
        gallery_embeddings_path=str(args.gallery_embeddings),
        registry_db_path=str(args.registry_db),
        gallery_labels_path=str(args.gallery_labels) if args.gallery_labels else None,
        top_k=args.top_k,
        similarity_threshold=args.threshold,
    )
    pipeline = EmbeddingInferencePipeline(config)

    images: list[Image.Image] = []
    tokens: list[str] = []
    for p in query_paths:
        images.append(Image.open(p))
        tokens.append(p.stem)

    t0 = time.time()
    results = pipeline.identify_batch(images, tokens)
    elapsed = time.time() - t0
    pipeline.close()

    manifest = {
        "schema_version": "cvi.embedding_inference_results.v1",
        "config": config.to_dict(),
        "query_count": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "results": [r.to_dict() for r in results],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2) + "\n"
    )
    print(json.dumps({
        "status": "DONE",
        "queries": len(results),
        "elapsed_seconds": round(elapsed, 3),
        "output": str(args.output),
    }, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
