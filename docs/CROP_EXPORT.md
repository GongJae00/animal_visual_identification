# Token-keyed Oracle Crop Export

## Scope

This stage sanitizes already admitted still-image frames and integer oracle
bounding boxes. It does not detect dogs, choose frames, resize evidence,
perform super-resolution, infer masks, or remove backgrounds.

For each protected sample binding it:

1. verifies the source SHA-256;
2. probes one still-image video stream;
3. rejects a crop outside source bounds;
4. applies `crop=w:h:x:y:exact=1` and `setsar=1`;
5. disables metadata/chapter copying;
6. writes exactly one no-overwrite PNG frame;
7. fixes RGB to `rgb24` and IR to `gray`;
8. reprobes dimensions, pixel format, PNG format, and tags;
9. rechecks source stat and SHA-256 after export;
10. emits token-only filenames and a rehashed artifact manifest.

FFmpeg may substitute an unsupported requested pixel format. Therefore command
success is insufficient; the output probe must exactly match the declared
format.

## Safety and limitations

The output directory must exist and be empty. A temporary sibling directory is
fully verified before files are hard-linked into the destination; partial
links created by a failed export or final verification are removed. Existing
files are never overwritten. Source and destination symlinks are rejected.

The hashed export policy bounds source bytes, source pixels, crop pixels,
artifact count, per-artifact bytes, total output bytes, and per-artifact wall
time. These are fail-closed resource ceilings, not evidence-quality thresholds;
raising one requires a new policy hash but does not alter accepted crop pixels.
Only still-image PNG and JPEG sources and explicit RGB or IR modalities are
admitted.

Metadata-tag removal does not prove absence of identity shortcuts in pixels.
Watermarks, cage background, collars, clothing, and source overlays remain
visual-control concerns. PNG encoding changes representation but cannot create
new biometric detail; these artifacts are oracle evaluation inputs, not
protected raw acquisition replacements.

## Protected CLI

`tools/export_oracle_crops.py` reconstructs the four separated pair artifacts,
requires their common hashes and exact content-addressed result to agree, reads
strict duplicate-key-free JSON without symlinks, exports the crops, and writes
a mode-0600 no-overwrite receipt. Receipt publication is preflighted before
export; if the final receipt write fails, only the crops created by that run
are removed.

```bash
uv run python tools/export_oracle_crops.py \
  --scoring-requests /protected/pairs/scoring.json \
  --artifact-bindings /protected/pairs/bindings.json \
  --ground-truth /protected/pairs/ground-truth.json \
  --pair-summary /protected/pairs/summary.json \
  --crop-sources /protected/crops/sources.json \
  --export-policy configs/data/crop_export_policy.example.json \
  --output-directory /protected/crops/token-images \
  --receipt-output /protected/crops/export-receipt.json
```
