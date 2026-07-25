"""Identity Search Engine CLI — enroll, search, explain, export-index.

Usage:
  # Enroll a new identity
  search_identity.py --model model.onnx --index-dir ./idx enroll \\
      --image crop.jpg --registered-dog-id uuid --dataset-name yt-bb-dog

  # Search top-k
  search_identity.py --model model.onnx --index-dir ./idx search \\
      --image crop.jpg --top-k 5

  # Explain (decompose score for a specific identity)
  search_identity.py --model model.onnx --index-dir ./idx explain \\
      --image crop.jpg --registered-dog-id uuid

  # Export index artifacts (for deployment)
  search_identity.py --model model.onnx --index-dir ./idx export \\
      --output-dir ./deploy
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from PIL import Image

from cvi.search_engine import SearchEngine


def _open_image(path: str) -> Image.Image:
    img = Image.open(path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    return img


def cmd_enroll(args: argparse.Namespace) -> None:
    engine = SearchEngine(Path(args.model), Path(args.index_dir))
    img = _open_image(args.image)
    idx = engine.enroll(
        img, args.registered_dog_id, args.dataset_name,
        json.loads(args.metadata) if args.metadata else {},
    )
    print(json.dumps({
        "event": "enrolled",
        "index_position": idx,
        "registered_dog_id": args.registered_dog_id,
        "total": engine.size,
    }, sort_keys=True))
    engine.close()


def cmd_search(args: argparse.Namespace) -> None:
    engine = SearchEngine(Path(args.model), Path(args.index_dir))
    img = _open_image(args.image)
    resp = engine.search(img, top_k=args.top_k,
                         fusion_weights=(args.w_v, args.w_t, args.w_s))
    print(json.dumps({
        "event": "search_result",
        "matches": [m.to_dict() for m in resp.matches],
        "elapsed_ms": resp.elapsed_ms,
    }, indent=2, sort_keys=True))
    engine.close()


def cmd_explain(args: argparse.Namespace) -> None:
    engine = SearchEngine(Path(args.model), Path(args.index_dir))
    img = _open_image(args.image)
    breakdown = engine.explain(img, args.registered_dog_id)
    print(json.dumps({
        "event": "explain",
        "registered_dog_id": args.registered_dog_id,
        "breakdown": breakdown,
    }, indent=2, sort_keys=True))
    engine.close()


def cmd_export(args: argparse.Namespace) -> None:
    import shutil
    src_dir = Path(args.index_dir)
    dst_dir = Path(args.output_dir)
    dst_dir.mkdir(parents=True, exist_ok=True)
    for fname in ["identities.idx", "identities.json"]:
        src = src_dir / fname
        if src.exists():
            shutil.copy2(src, dst_dir / fname)
    model_src = Path(args.model)
    if model_src.exists():
        shutil.copy2(model_src, dst_dir / model_src.name)
    print(json.dumps({
        "event": "exported",
        "output_dir": str(dst_dir),
        "files": [str(p) for p in dst_dir.iterdir() if p.is_file()],
    }, indent=2, sort_keys=True))


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True, type=str)
    parser.add_argument("--index-dir", required=True, type=str)

    sub = parser.add_subparsers(dest="command")

    p_enroll = sub.add_parser("enroll")
    p_enroll.add_argument("--image", required=True)
    p_enroll.add_argument("--registered-dog-id", required=True)
    p_enroll.add_argument("--dataset-name", default=None)
    p_enroll.add_argument("--metadata", default=None)

    p_search = sub.add_parser("search")
    p_search.add_argument("--image", required=True)
    p_search.add_argument("--top-k", type=int, default=5)
    p_search.add_argument("--w-v", type=float, default=1.0)
    p_search.add_argument("--w-t", type=float, default=0.5)
    p_search.add_argument("--w-s", type=float, default=0.5)

    p_explain = sub.add_parser("explain")
    p_explain.add_argument("--image", required=True)
    p_explain.add_argument("--registered-dog-id", required=True)

    p_export = sub.add_parser("export")
    p_export.add_argument("--output-dir", required=True)

    args = parser.parse_args()
    if args.command == "enroll":
        cmd_enroll(args)
    elif args.command == "search":
        cmd_search(args)
    elif args.command == "explain":
        cmd_explain(args)
    elif args.command == "export":
        cmd_export(args)
    else:
        parser.print_help()
        sys.exit(1)


if __name__ == "__main__":
    main()
