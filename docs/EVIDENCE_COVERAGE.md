# G1 Evidence Coverage

## Purpose

Coverage asks whether the camera produces potentially usable visual evidence.
It does not estimate identity accuracy. Initial thresholds are versioned
engineering definitions and may be replaced only before the corresponding
manifest/report is frozen.

## Duration weighting

Frame counts are biased by variable sampling and duplicate adjacent frames.
For consecutive observations at \(t_i,t_{i+1}\), observation \(i\) receives:

\[
w_i=\min(t_{i+1}-t_i,\;h_{\max}),
\]

where \(h_{\max}\) is a declared hold limit. Any excess gap is accumulated as
unobserved duration rather than assigned to the last frame. Final observations
receive one expected sampling period.

The accumulator keeps the previous observation, constant-size modality
counters, and fixed pixel histograms. Its memory does not grow with video
duration.

Usable-tracklet opportunities additionally require a complete
camera/session/track namespace, one modality, continuous usable evidence, and a
predeclared minimum duration. Long sampling gaps close the active opportunity;
missing track keys are reported rather than inferred.

## Initial usable-evidence definition

The versioned policy specifies:

- minimum dog crop height;
- minimum head long edge and face minimum edge;
- minimum visible fraction;
- maximum total, cage-bar, motion-blur, and defocus scores;
- minimum detector/mask confidence;
- exposure validity;
- maximum IR saturation;
- expected sampling period and maximum hold multiple.

Full-body, head, and face coverage are reported separately. Missing quality
fields are counted and treated as not usable, never silently imputed.

## JSON Lines workflow

```bash
uv run python workflows/summarize_evidence_coverage.py \
  --policy canine_identity/configs/evidence/evidence_coverage.example.json \
  --observations /protected/audits/camera-01.jsonl \
  --timeline-start-ns 1784516400000000000 \
  --timeline-end-ns 1784602800000000000
```

The report includes aggregate and per-modality observed duration, unobserved
gap duration, occupancy duration, usable-region coverage, missing-quality
duration, region availability, exposure failure, fixed pixel-size histograms,
and fixed 10-bin visibility/occlusion/blur/cage-bar/IR-saturation histograms. A
synthetic or placeholder report is not camera evidence.

Timeline bounds are mandatory in the CLI so the interval before the first
observation and after the final observation cannot disappear from the coverage
denominator.
