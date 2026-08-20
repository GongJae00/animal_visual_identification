# Identity Metadata and Split Contract

## Assignment unit

The smallest assignable unit is a tracklet, not a frame. Every record carries
the source, camera, cage, site, session, occupancy episode, track, time interval,
modality, externally verified dog identity, and verification source.

The checker always forbids the same:

- raw source;
- camera-namespaced session;
- camera/session/cage-namespaced occupancy episode; or
- camera/session-namespaced track

from crossing split roles. Multiple tracklets from one track therefore cannot
be used to recreate a random-frame split.

## Open-set roles

Calibration and final test have separate identity banks:

```text
TRAIN
DEVELOPMENT
CALIBRATION_GALLERY
CALIBRATION_KNOWN_QUERY
CALIBRATION_UNKNOWN_QUERY
TEST_GALLERY
TEST_KNOWN_QUERY
TEST_UNKNOWN_QUERY
```

Known-query identities must exist in the corresponding gallery. Unknown-query
identities must be absent from that gallery and its known queries. By default,
training, calibration, and test identity sets are mutually disjoint. This
measures unseen-identity generalization and prevents final-test identities from
shaping thresholds.

## Separate transfer protocols

Camera-, cage-, site-, and chronological transfer answer different questions.
They are declared in `SplitPolicy`; they are not silently combined into one
maximally sparse split.

- `stage_disjoint_keys=["camera_id"]`: training, calibration, and test cameras
  are disjoint.
- `stage_disjoint_keys=["cage_id"]`: stage-level cage transfer.
- `stage_disjoint_keys=["site_id"]`: stage-level facility transfer.
- `require_chronological_test=true`: every final-test interval starts after all
  training and calibration intervals end.

Each protocol receives a separate manifest hash and report. RGB→IR and IR→RGB
directional evaluation is defined by gallery/query role metadata, not by
removing one modality globally.

`modality_rules` fixes the modalities allowed in each role before evaluation.
`require_known_query_accessory_change` requires a resolved collar/harness/
clothing signature and at least one changed gallery/query pairing per known
identity. `minimum_gallery_query_gap_seconds` fixes the longitudinal gap before
scores are observed.

## Comparable transfer protocol

Frozen comparison panel for prior-work-adjacent closed-set retrieval. This is
not a random-frame split and not a BIFOR sequence-mean reproduction.

- Train: official `yt-bb-dog` train identities only.
- Eval: Sibetan identities held out of training (identity-disjoint).
- Gallery and query: same frozen lists; sequences do not overlap; seed `0`.
- Crops: parser policy v6 materialization, byte-bound when scored.
- Comparison variable: backbone only.
- Metrics: Rank-1, Rank-5, mAP.

```bash
uv run python -m evaluation.commands.evaluate comparable-transfer freeze --help
```

## Validation

```bash
uv run python -m evaluation.commands.evaluate split-check \
  /protected/manifests/split.json
```

A structural pass does not prove that the number of identities, hard negatives,
or time gaps is statistically sufficient. Those are experiment-design gates.
