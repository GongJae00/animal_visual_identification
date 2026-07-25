# Public Dataset Feasibility Gate

The animal hospital requires a quantitative public-data feasibility result
before producing CVI-specific RGB/IR cage video. Public data feasibility is
therefore an upstream evidence gate, not a substitute for camera intake.

## Claims this gate can support

Subject to a frozen, leakage-audited protocol, public dog identity data can
measure:

- oracle-crop RGB verification and retrieval;
- identity-disjoint generalization;
- open-set rejection with calibration identities separated from test identities;
- gallery-size, latency, memory, storage, and energy scaling;
- shortcut sensitivity where background or sequence metadata is available.

It cannot establish real canine RGB-to-IR matching, cage-video detector or
tracker performance, camera-hour error rates, or longitudinal hospital-domain
stability. Grayscale or synthetic pseudo-IR is an augmentation experiment only
and must never be reported as real IR evidence.

## Intake lanes

`RESEARCH_ONLY` permits feasibility use but cannot supply deployment-model data
lineage. `DEPLOYMENT_ELIGIBLE_CANDIDATE` means only that the published license
and source integrity are candidates for later legal and provenance review. It
does not automatically admit a dataset or a derived model for deployment.

Every archive is bound to an official page, exact URL, version, license
snapshot, byte size, and published checksum authority. If the provider did not
publish a checksum, the archive remains research-only and the locally observed
SHA-256 is not misrepresented as publisher-authenticated.

## Fail-closed archive intake

`tools/audit_public_dataset_archive.py` performs no extraction. It rejects
source or terms mismatches, path traversal, Unicode/casefold path collisions,
symbolic links, encrypted or unsupported members, disallowed suffixes,
oversized archives or members, excessive expansion ratios, CRC failures, and
files that change during verification. Its receipt means only that archive
intake passed; it does not admit labels, decoded pixels, duplicates, or splits.

`tools/audit_public_canine_semantics.py` then binds dataset-specific identity
meaning and exact cardinalities to the admitted archive receipt. The protected
semantic receipts distinguish web folders, device-capture identities, JSON
ground truth, and video-track labels; filename camera/sequence tokens remain
unverified provenance and are never visual-model inputs. This semantic gate has
passed for 38,811 independent source images and 4,366 dataset-namespaced source
identities. It is not image-decode, duplicate, split, or model admission.

The subsequent receipt-bound image gate decoded all 45,875 processed inputs
(the 38,811 independent sources plus 7,064 paired YT background controls) and
computed canonical RGB pixel hashes without exposing labels. It found two
same-identity, same-sequence Sibetan exact frame pairs and no cross-dataset
exact collision. MPDD also contains four opaque PNG/RGBA payloads stored under
`.jpg` names. These facts are recorded rather than silently normalized. Near-
duplicate/crop detection and label-conflict quarantine still block split
admission.

`tools/audit_public_canine_phash.py` is the protected next-stage runner. It
rederives all four semantic manifests, authenticates the image-content
receipts, reopens the exact source ZIPs, verifies decoded pixels again, and
publishes label-blind pHash evidence separately from the opaque-to-source join.
Its radius-10 candidates remain review inputs only; they do not adjudicate a
duplicate or admit a split.

Nested packaging archives require a second audit after the publisher-bound
outer archive is admitted. Allowing `.zip` in a dataset-specific outer policy
does not admit the nested contents.

## Frozen experiment sequence

1. Audit and safely extract into an immutable, content-addressed data root.
2. Inventory identities, images, sequence/camera fields, corrupt decodes, exact
   duplicates, and near duplicates.
3. Freeze identity/session/camera-aware train, calibration, validation, and test
   partitions before model selection.
4. Establish simple non-learned and pretrained oracle-crop references, with
   pretraining-overlap and license lineage recorded.
5. Train or adapt only on the frozen training identities; use calibration data
   for thresholds and leave test identities sealed.
6. Report verification, closed-set retrieval, open-set identification,
   coverage-risk, shortcut controls, and resource measurements together.
7. Promote an optimization only through `docs/OPTIMIZATION_CONTRACT.md`.

Actual gallery sizes must be reported as actual. Repeated or synthetic
distractors may test compute scaling but must not be called a 10,000-identity
biometric evaluation.
