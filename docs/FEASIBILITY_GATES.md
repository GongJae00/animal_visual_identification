# Feasibility Gates

Stages are sequential evidence gates. Infrastructure may be prepared early, but
no downstream performance claim may bypass a failed upstream gate.

## G0 — Acquisition contract

Required:

- camera inventory and RGB/IR mechanism;
- codec, stored FPS, bitrate, GOP, timestamp behavior, and drop rate;
- three or more 24-hour source samples when available;
- checksum manifest and protected raw-data location;
- day, night, and transition interval extraction.

Stop or redesign if modality state and time alignment cannot be reconstructed.

## G1 — Evidence coverage

Measure dog/body/head/face pixel size, visibility, blur, occlusion, exposure,
IR saturation, and usable tracklet opportunities per dog-active hour.

No recognition model is promoted if the camera does not produce adequate
evidence. The remedy is acquisition redesign, not generative reconstruction.

## G2 — Identity and leakage contract

Required:

- externally verified dog/session/episode/track linkage;
- namespace and timestamp rules;
- video-, session-, time-, camera-, cage-, and open-set split checks;
- dog–cage and dog–camera association audit;
- background-, cage-, and accessory-only controls.

No random frame split is permitted.

## G3 — Oracle recognition bound

Evaluate manually admitted body/head crops before automated front-end effects.
Mode 1 verification is primary; closed-set retrieval is secondary.

Stop or narrow the ODD if same-breed verification under independent sessions
cannot meet a predeclared operating region.

## G4 — Automated front end

Add detection, segmentation where justified, tracking, region extraction, and
quality selection. Attribute the delta from the oracle bound to each component.

Promotion requires usable-tracklet recall and track purity in addition to
recognition metrics.

## G5 — Tracklet identity

Compare single frame, quality Top-K, quality-diversity Top-K, weighted
aggregation, and multiple prototypes under the same eligible evidence horizon.

Temporal aggregation must outperform or safely reject more often than the
single-frame reference; forced coverage is not an improvement.

## G6 — RGB/IR and open set

Evaluate all four query/gallery modality directions, modality-specific
calibration, unknown identities, gallery scaling, score margin, and temporal
consensus.

Known/unknown thresholds are frozen before independent testing.

## G7 — Deployment Pareto optimization

Profile the end-to-end pipeline before applying:

- adaptive scheduling and redundant-call removal;
- hardware decode and transfer reduction;
- FP16 compilation;
- INT8 PTQ, selective QAT, or distillation;
- batching and concurrent-stream scheduling;
- exact-to-approximate search transition if required.

Each change follows `OPTIMIZATION_CONTRACT.md`.

## G8 — Longitudinal templates

Static protected templates are the reference. Dynamic prototypes remain
quarantined, versioned, reviewable, and reversible.

No automatic update is enabled without an independent contamination-rate bound.

## G9 — Independent feasibility decision

Required:

- frozen code, model, gallery, thresholds, split, and hardware profile;
- unseen identities/sessions and an allowed camera/cage transfer condition;
- visual-only and expected-ID-assisted reports;
- false assignment per dog-day and per camera-hour;
- decision latency, review burden, evidence coverage, and resource budget;
- explicit unsupported claims and scale-up requirements.

The decision is `GO`, `NARROW_ODD`, `REDESIGN_ACQUISITION`, or `NO_GO`; a
successful software demonstration alone is insufficient.

