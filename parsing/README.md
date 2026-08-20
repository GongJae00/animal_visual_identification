# Parsing

In: source images, optional frozen detections/masks, receipt-bound model artifacts.

Out: boxes, instance/foreground masks, region candidates, quality scores, and crops.

`training/` holds student/teacher loops (region decoders, SAM2 teacher masks, nose-mask student, nose-localizer datasets). Identity embedding trainers are not owned here; they live under `identification/training/nose/`.

`export/` is the runtime path in order: detection → segmentation → regions → quality → crops. Export must not import training.

Commands: `uv run python -m parsing.commands.parse --help`
(`materialize`, `manifest`, `panel`, `compare`, `three-region`,
`benchmark-batches`, `benchmark-localizers`, `unified-manifest`,
`oracle-crops`).
