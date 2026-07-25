# System Contract

## Scope

CVI accumulates identity evidence from continuous cage RGB or IR video. It must
detect absence, multiplicity, insufficient evidence, unknown identities, and
operational conflicts without forcing a registered identity.

The first operational target is Mode 1: verify an externally supplied expected
dog ID. Mode 2 gallery retrieval, Mode 3 open-set identification, cross-modal
matching, and longitudinal updates remain gated extensions.

## Orthogonal internal state

A single status field cannot preserve all causes. Internal results therefore
separate five state axes:

| Axis | Values at the initial boundary |
|---|---|
| Stream | `STREAM_OK`, `STREAM_DEGRADED`, `STREAM_FAILED` |
| Occupancy | `NOT_EVALUATED`, `NO_DOG`, `SINGLE_DOG`, `MULTIPLE_DOGS`, `OCCUPANCY_UNCERTAIN` |
| Evidence | `NOT_EVALUATED`, `USABLE`, `NO_USABLE_EVIDENCE` |
| Visual identity | `NOT_EVALUATED`, `PENDING`, `KNOWN`, `UNKNOWN`, `REVIEW_REQUIRED` |
| Operational conflict | `NONE`, `IDENTITY_CONFLICT` |

The external API may expose a representative status, but it must retain the
orthogonal fields and visual-only scores. `IDENTITY_CONFLICT` must not overwrite
the visual prediction that caused the conflict.

`STREAM_FAILED` and genuinely uncertain occupancy have no identity decision
status. They retain their explicit state axes instead of being mislabeled as
`NO_DOG` or `NO_USABLE_EVIDENCE`.

## Identity namespace

`frame_detection_id`, `track_id`, `visual_entity_id`, and
`registered_dog_id` are distinct. A `track_id` is unique only inside a declared
camera/session namespace. No tracker output is accepted as registered identity
evidence by construction.

## Evidence boundary

An identity decision may consume:

- dog, head, face, muzzle, body-side, and motion regions derived from the dog;
- frame and region quality;
- RGB/IR modality;
- temporal agreement among eligible tracklets;
- registered visual templates and their provenance.

It must not consume as identity evidence:

- cage, room, site, timestamp bucket, camera identity, or camera watermark;
- bedding, bowl, wall, handler, or background appearance;
- collar, harness, or clothing appearance;
- expected dog ID or occupancy records;
- evaluation labels, future frames outside the declared decision horizon, or
  cached outputs from another split.

Camera and cage metadata may condition calibration or domain diagnostics only
when the visual-only score is retained and the identity candidate set is not
hard-filtered by that metadata.

## Decision ordering

The initial deterministic order is:

1. `STREAM_FAILED` blocks occupancy and identity inference.
2. Reliable no-dog evidence maps to `NO_DOG`.
3. Reliable multi-dog evidence maps to `MULTIPLE_DOGS`.
4. A single dog with insufficient eligible evidence maps to
   `NO_USABLE_EVIDENCE`.
5. Eligible evidence is evaluated visually.
6. A known decision requires score, margin, uncertainty, and temporal gates.
7. Low match evidence maps to `UNKNOWN`; ambiguous evidence maps to
   `REVIEW_REQUIRED`.
8. Expected-ID comparison is applied after the visual-only result and may add
   `IDENTITY_CONFLICT`.

Exact thresholds remain unfrozen until calibration data and a gallery-size
contract exist.

## Structural separation

The visual result object contains track provenance, modality, quality,
candidates, evidence references, and model/gallery versions. It cannot contain
`cage_id` or `expected_dog_id`. Those fields live in a separate operational
context object and are joined only when the final decision record is created.
This type boundary is not proof of invariance, but it prevents the primary API
from silently introducing operational priors into the visual score.

## Protected evidence

Protected enrollment references are immutable through automated operation.
Dynamic templates are separately versioned, reversible, and disabled until the
static open-set system passes an independent contamination audit.

## Forbidden claims

Until supported by admitted evidence, CVI must not claim:

- real-time capacity for a specified camera count;
- cross-camera or cross-cage generalization;
- RGB-to-IR identity invariance;
- commercial open-set accuracy;
- long-term or lifelong identity stability;
- safe automatic template updating;
- superiority from quantization, pruning, distillation, or adaptive sampling.
