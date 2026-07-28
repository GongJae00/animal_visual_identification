"""Run zero-shot dog detection across canid datasets for qualitative audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from PIL import Image

from cvi.canid_data.adapters import ADAPTERS
from cvi.canid_data.source_lock import admitted_records, get_record
from cvi.localization.benchmark import build_contact_sheet, run_benchmark
from cvi.localization.types import LocalizationResult


def _yolo_detect(image: Image.Image, *, image_id: str) -> LocalizationResult:
    """Lazy YOLO detection — only imported on first call."""
    from cvi.detection import DogDetector, Detection

    detector = _yolo_detect._detector  # type: ignore[attr-defined]
    raw = detector.detect(image)
    boxes = [
        Detection(
            x1=int(d.x1), y1=int(d.y1), x2=int(d.x2), y2=int(d.y2),
            confidence=d.confidence, class_id=d.class_id, class_name=d.class_name,
        )
        for d in raw
    ]
    return LocalizationResult(
        image_id=image_id,
        dog_boxes=tuple(
            DetectionBox(
                x1=float(b.x1), y1=float(b.y1), x2=float(b.x2), y2=float(b.y2),
                confidence=b.confidence, class_name=b.class_name,
            )
            for b in boxes
        ),
        face_boxes=(),
        nose_boxes=(),
        body_keypoints=(),
        face_landmarks=(),
        model_name="yolo-legacy",
        model_family="ultralytics-dog",
        inference_ms=0.0,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("reports/localization/zero-shot"))
    parser.add_argument("--max-images", type=int, default=64)
    parser.add_argument("--dataset", type=str, default="yt-bb-dog")
    args = parser.parse_args()

    record = get_record(args.dataset)
    adapter = ADAPTERS[args.dataset]
    root = Path(record.data_root)
    samples = adapter(root)[: args.max_images]

    import torch
    try:
        from cvi.detection import DogDetector as DetectorClass
        _yolo_detect._detector = DetectorClass(
            model_path=None, device="cuda" if torch.cuda.is_available() else "cpu"
        )
    except Exception as exc:
        print(json.dumps({"error": f"YOLO detector failed: {exc}"}))
        return

    results: list[LocalizationResult] = []
    for sample in samples:
        image_path = root / sample.image_path
        if not image_path.is_file():
            continue
        image = Image.open(image_path).convert("RGB")
        result = _yolo_detect(image, image_id=sample.sample_id)
        results.append(result)

    output = args.output_dir / args.dataset
    build_contact_sheet(
        tuple(results), samples, root, output, grid_size=8
    )

    stats = {
        "dataset": args.dataset,
        "images": len(results),
        "dog_detections": sum(len(r.dog_boxes) for r in results),
        "detection_rate": (
            sum(1 for r in results if r.dog_boxes) / max(len(results), 1)
        ),
    }
    output.mkdir(parents=True, exist_ok=True)
    (output / "summary.json").write_text(json.dumps(stats, indent=2))
    print(json.dumps(stats))


if __name__ == "__main__":
    main()
