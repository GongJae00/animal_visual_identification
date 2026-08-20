# Parsing

In: source images, optional frozen detections/masks, receipt-bound model artifacts.

Out: boxes, instance/foreground masks, region candidates, quality scores, and crops.

`training/` holds student/teacher loops (region decoders, SAM2 teacher masks, nose-mask student, nose-localizer datasets). Identification embedding trainers live under `identification/training/`.

`export/` is the runtime path in order: detection → segmentation → regions → quality → crops. Export must not import training. That order is backbone-independent.

Stage × data × metric × extraction catalog: `evaluation.parsing_protocol` (JSON, not a figure).

Commands: `uv run python -m parsing.commands.parse --help`
(`materialize`, `manifest`, `panel`, `compare`, `three-region`).
Parser batch benchmarks live under `archive.full128.evaluation.parsing_batch_benchmark`.
Localizer benchmarks, Oxford-Pet, oracle crops, parsing protocol, and the parsing optimization catalog: `evaluation.commands.evaluate`.
Unified manifests: `data.commands.audit unified-manifest`.
