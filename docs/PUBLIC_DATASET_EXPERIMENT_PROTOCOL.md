# Public RGB Oracle-Crop Feasibility Protocol

This protocol is the quantitative prerequisite for requesting hospital RGB/IR
cage video. It is pre-hospital evidence, not a replacement for G0 camera intake.

## Frozen evidence scope

Independent source images comprise 38,811 visible-light/web-color crops and
4,366 dataset-namespaced source identities:

| Dataset | Images | Source identities | Primary role |
|---|---:|---:|---|
| YT-BB-Dog original | 27,036 | 2,723 video-track identities | training and held-out in-domain retrieval |
| Sibetan | 1,755 | 59 JSON-mapped dogs / 223 sequences | external cross-sequence test |
| MPDD | 1,657 | 191 archive identities | external cross-pose test |
| DogFaceNet 224 | 8,363 | 1,393 web-folder identities | separate face-crop test |

YT-BB-Dog random-background contains 7,064 paired counterfactual versions of
the original 7,104-image test set. It is not counted as independent data or an
additional identity set. The missing 40 pairs are excluded from paired deltas.

None of these datasets contains verified canine IR. Grayscale and pseudo-IR
results may be reported only as visible-image robustness ablations.

## Semantic and leakage gates

Before model work:

1. Verify every archive and nested archive against the protected v3 receipts.
2. Parse DogFaceNet identity from its parent folder, never basename prefix.
3. Treat MPDD filename `c` and `s` fields as unverified nuisance tokens, not
   physical camera or session IDs.
4. Map Sibetan's 223 cluster folders to 59 dog identities through
   `gt_sibetan.json`; a folder is a sequence, not an identity.
5. Treat each YT identity as a temporary video-track visual entity, not a
   registered lifelong dog identity.
6. Decode every image, record dimensions/mode, and quarantine corrupt or
   unsupported images before splitting.
7. Build exact SHA-256 duplicate components and perceptual near-duplicate
   components. A component crossing identities is a label conflict; a component
   crossing roles is removed from the protected result.
8. Freeze all train, development, calibration, gallery, known-query, and
   unknown-query roles before model or threshold selection.

Paths, folder IDs, split names, camera tokens, and sequence IDs are provenance
only. They must not reach the visual model or scorer.

### Near-duplicate precommitment

Candidate construction is label-blind and receives only an opaque audit token
and canonical RGB pixels. Exact-pixel components are collapsed to one
representative before near-duplicate search and expanded afterward.

The deterministic first candidate channel uses a 64-bit pHash: 32x32 grayscale
DCT, a fixed low-frequency zigzag of 64 AC coefficients with DC excluded,
bit `1` only when a coefficient is strictly above the median, and both original
and horizontal-flip fingerprints. Candidate radius is Hamming distance at most
10. Four 16-bit blocks are indexed; querying every block key within distance 2
is complete for global radius 10 because at least one of four blocks must have
distance at most `floor(10/4)=2`. Candidate generation is checked against
brute-force search on synthetic/random fixtures.

MIH bucket collisions are deduplicated only within the current query. An
unordered pair is inspected only while its lexicographically larger opaque ID
is the query, so retaining every provisional pair globally is unnecessary. For
45,875 random-like two-orientation hashes, the conservative approximation
`n(n-1)/2 * [1-(1-137/65536)^16]` is about 35 million unique pair inspections;
the frozen default therefore permits 100 million integer-counted inspections
while retaining at most one million accepted/expanded candidates. This keeps
exact radius-10 completeness without a tens-of-millions-entry provisional set.

A second channel uses weight-free BSD-3-Clause Meta PDQ: 256-bit signatures,
all eight dihedral transforms retained separately, and calibration-frozen
quality/distance rules. The upstream quality 49 and Hamming distance 31 values
mean that the upstream initialization discards quality at most 49 and retains
quality at least 50 while accepting distance at most 31; they are not
automatically valid CVI thresholds. PDQ quality is a 64x64 gradient heuristic,
not canine recognition quality, and must never enter the identity quality
selector. PDQ receives the receipt-verified full-resolution canonical RGB
pixels; the pHash 32x32 raster must not be reused. Store all eight hashes and
the winning orientation pair rather than canonicalizing them to one minimum
hash. At distance at most 31, use sixteen 16-bit slots and enumerate the exact
key plus every one-bit flip per slot. The resulting 272 bucket lookups per
orientation are complete because one slot must have distance at most one.
Low-quality samples are `PDQ_INELIGIBLE_LOW_QUALITY`, not evidence of no
duplicate, and remain in the pHash/verifier/review path. The v1 caps are 400,000
orientation hashes, 1.5 billion raw posting visits, 800 million unique
orientation-pair inspections, and one million accepted sample pairs. Cap
exhaustion is `AUDIT_CAP_EXCEEDED`, never silent truncation. Crop and letterbox
transform hashes are added only after synthetic-transform recall is measured
and frozen under a separate policy version.

An optional third channel uses a separately license-admitted frozen visual
embedding. It contributes exact blocked top-100 cosine neighbours plus every
pair above a calibration-frozen threshold. SSCD/RDCD weights are research-only
because their weight and DISC-derived lineage is not deployment-admissible;
they cannot enter the deployment result. At 45,875 inputs and 512 dimensions,
an FP32 matrix is about 89.6 MiB; blocked exact search is preferred to ANN at
this scale. More than ten million union candidates is `AUDIT_CAP_EXCEEDED`, not
a reason to truncate silently.

Neither pHash nor embedding similarity confirms a duplicate. Automatic edges
require a frozen geometric/photometric verifier; crop-like cases require local
inliers, spatial coverage, bounded reprojection error, overlap, and overlap
SSIM/gradient consistency. Border/background-only support cannot confirm an
edge. Unresolved candidates enter a label-blind review graph and remain excluded
from protected splits.

Only after the candidate and confirmed-edge receipts are frozen are labels and
roles joined. A confirmed component crossing identity becomes
`LABEL_CONFLICT`; a component crossing train/calibration/test or gallery/query
roles is removed from protected evaluation. YT original/random-background pairs
are typed dependencies, not automatic duplicate edges. Same-dog different-pose
images are not duplicates merely because an identity model considers them
similar.

## Dataset roles

### YT-BB-Dog

Preserve the official 2,000-train/723-test identity boundary. After duplicate
closure and quarantine, rank eligible official-training identities with a
domain-separated HMAC-SHA256 over the protected evidence bundle and target
1,200 fit while preserving exactly 200 development, 300 calibration-known, and
300 calibration-unknown identities. Architecture, preprocessing, score fusion,
and the margin rule are selected without either calibration subset. The
protected final model remains trained on fit identities only: development
identities are not merged into training after margin selection because
retraining would change the score distribution governed by that margin. The
600 calibration identities never enter training or model selection.
If quarantine leaves 1,800--1,999 eligible identities, reduce only fit to
`eligible_count - 800` (minimum 1,000) and record requested and actual counts.
If fewer than 1,800 remain, or development and the two calibration roles cannot
retain 200/300/300 identities, emit `SPLIT_CAPACITY_FAILED` rather than silently
shrinking the safety evidence.

The official 723 test identities remain sealed until the model and all decision
rules are frozen. Use all 723 for closed-set retrieval. For a separate protected
open-set protocol, assign 300 known and 423 unknown test identities by another
domain-separated HMAC ranking. With zero false accepts, 300 independent
calibration-unknown events give a one-sided 95% FPIR upper bound of 0.994%, and
423 protected unknown events give 0.706%; smaller frame-level event counts must
not be substituted for these identity-level events.

For each primary `k in {1,3}` temporal proxy, collapse confirmed duplicate components,
order the remaining components by source frame index, choose the first `k` as
gallery, the next component as an unused guard, and the last component as query.
Require both component ordinal and raw frame-index gaps of at least two between
the final gallery item and query. All 723 protected-test identities must remain
eligible at `k=3`; otherwise construction fails before any score is read. The
5-shot result is a separate closed-set diagnostic over its duplicate-closed
eligible subset, with the subset size reported explicitly and no replacement
after scores are seen. This remains
same-video-track evidence and must not be called cross-session, longitudinal,
or a measured time gap.

The split seed is a protected 32-byte secret with a public commitment. Derive a
master HMAC key from the seed and exact semantic, decoded-pixel, duplicate-graph,
and protocol hashes; derive separate keys for identity roles, sequence roles,
frame roles, and bootstrap draws. Never use Python `hash()`, PRNG shuffle, file
system order, labels, or model scores for protected role assignment.

The 200 development identities are divided by another protected HMAC domain
into A and B panels of 100 identities. Episode A uses A as known gallery/query
and B as unknown query; episode B reverses those roles. These crossed episodes
select the margin/scalar construction only and are not claimed as 400
independent identities. The 300 calibration-known identities form nested
gallery panels at `N in {39, 64, 100, 300}`. The separate 300
calibration-unknown identities never enter any gallery and contribute exactly
one primary unknown query per identity at each registered gallery size. Every
development and calibration query uses the same first-k, unused-guard,
last-query, component-gap, and raw-frame-gap rules as the protected test for
the primary `k in {1,3}` panels. A
shot/gallery panel that cannot preserve all preassigned identities fails with
an explicit capacity status; it is never backfilled after scores are seen.

Each random-background image is a typed dependency of its matching original
image, inherits the original role, and is used only for an identity-level paired
robustness delta. It is never an independent sample, gallery item, or unknown.
The 40 missing pairs are excluded only from that paired delta.

### Sibetan

Use the 39 identities with at least two sequence clusters for cross-sequence
closed-set retrieval. The 20 singleton identities form a natural unknown set
for an externally calibrated open-set diagnostic. Only seven identities span
different camera tokens, so strict cross-camera results are a small diagnostic,
not a primary KPI.

Choose one gallery sequence and nested 1/3/5-shot frames for closed-set
diagnostics and 1/3-shot frames for the `N=39` open-set diagnostic with the
protected sequence/frame HMAC keys; use other sequences as queries, with one primary query
event per sequence and identity-clustered secondary analysis. The 20 unknown
identities provide only exploratory rejection evidence: zero false accepts
still has a one-sided 95% FPIR upper bound of 13.91%.

### MPDD

Preserve train/validation identities (95) and disjoint query/gallery identities
(96). The primary result is zero-shot query-to-gallery retrieval with a model
selected without MPDD test identities. An open-set derivative freezes 64 known
and 32 unknown test identities by keyed hash, uses exact `N=64`,
`k in {1,3}` manifests, and must not alter the original query/gallery manifest.

Keep MPDD's publisher query/gallery roles for the 96-ID closed-set result and
choose nested gallery shots plus one primary query per identity by protected
HMAC. The 32-unknown derivative is exploratory because even zero false accepts
has a one-sided 95% FPIR upper bound of 8.94%. Filename `c` and `s` tokens do
not establish physical camera or session separation.

### DogFaceNet

Keep face-only results separate from dog-body results. The official ancillary
class lists contain 1,254 train identities and 139 test identities and must be
checksum-admitted before use. Exact and perceptual duplicate conflicts are
reported because the data were web collected.

After duplicate closure, assign the 1,254 training identities to 1,004 fit, 125
development, and 125 calibration roles by protected HMAC. Report fixed 1- and
3-shot face retrieval on the 139 official test identities, not full-body,
temporal, or session-generalization performance. DogFace data must not silently
enter the primary full-body model.

### External-result boundary

`STRICT_EXTERNAL_DOMAIN_ZERO_SHOT` means model weights, preprocessing, score
fusion, and margin use only YT training, development, and calibration roles.
The only automatic safety boundary currently admitted by code is the
precommitted YT `N=300`, 3-shot point. MPDD `N=64` and Sibetan `N=39`
open-set results remain descriptive diagnostics until a family-wise calibration
allocation is frozen before score access; no interpolated or borrowed automatic
threshold is applied. MPDD and Sibetan protected scores are then
revealed once. Dataset-specific adaptation or threshold calibration is a
separate `WITHIN_DATASET_UNSEEN_IDENTITY` diagnostic and cannot replace the
strict result. Known and unknown evidence in every open-set protocol must share
dataset, source variant, region, preprocessing, evidence count, and temporal
position rules; cross-dataset, face-versus-body, or random-background unknowns
are prohibited.

## Metrics and uncertainty

Closed-set reports Rank-1/5/10 and mAP for fixed 1- and 3-shot enrollment over
all 723 identities. The 5-shot result is reported only for its pre-score fixed
eligible subset and is labeled diagnostic. Verification reports ROC, AUC, EER,
and FNMR at FMR 1e-2; 1e-3 is
exploratory unless enough independent negative events exist.

For open-set query `q` and gallery identity `j`:

\[
S_1 = \max_j S(q,j), \qquad \Delta = S_1-S_2.
\]

Automatic `KNOWN` requires both `S_1 >= tau_N` and `Delta >= delta_N`.
Freeze the margin or scalar-confidence construction on development identities,
then use calibration-unknown identities only to select the scalar threshold by
an exact order-statistic/tolerance rule. Jointly grid-searching threshold and
margin on calibration coverage is prohibited. The primary `N=300`, 3-shot
threshold is derived on YT calibration. Secondary N/shot thresholds require a
pre-score family-wise allocation receipt; without it they are not frozen or
transferred to strict external zero-shot tests. Report TPIR/FNIR at observed FPIR 1e-2,
coverage-risk curves, `REVIEW_REQUIRED` coverage, and the one-sided 95% upper
bound on wrong automatic assignments.

After the development-frozen margin, define an unknown-query effective score as
`X_i=S_1` only when the margin passes and the top identity is unique; otherwise
use negative infinity. For unknown count `n`, target FPIR `p_0`, and one-sided
error `alpha`, admit at most

\[
c^*=\max\left\{c:\sum_{j=0}^{c}{n\choose j}p_0^j(1-p_0)^{n-j}\le\alpha\right\}.
\]

If `X_(1) <= ... <= X_(n)`, freeze
`tau=nextafter(X_(n-c^*), +infinity)` so all boundary ties are rejected. For
`n=300`, `p_0=0.01`, and `alpha=0.05`, `c^*=0`; the threshold must therefore be
strictly greater than the maximum calibration-unknown effective score. If that
value is the maximum representable model score, use an explicit
`AUTO_ACCEPT_DISABLED` state rather than weakening the threshold. Equal top
scores for two distinct gallery identities always produce `REVIEW_REQUIRED`;
opaque tokens may break serialization order but never identity decisions.
The primary 1% safety claim is the precommitted `N=300`, 3-shot operating point.
Other `N` and shot panels are secondary unless their family-wise calibration
error is allocated before score access. Threshold interpolation to an
unregistered gallery size is prohibited and returns `RECALIBRATION_REQUIRED`.

Use 10,000 paired hierarchical bootstrap draws with identity as the outer
cluster; Sibetan additionally resamples sequence within identity. Frame-level
bootstrap is prohibited. Multiple comparisons use Holm correction.
Bootstrap repetitions narrow Monte Carlo error only and do not increase the
number of independent identities or rare-event trials.

## Baselines

The minimum frozen comparison contains:

1. HSV/LBP/HOG and background-only/border-only shortcut controls;
2. one generic frozen visual encoder;
3. compact and larger animal-ReID encoders with independently audited weights;
4. the official BIFOR reproduction anchor where its license and train overlap
   are recorded;
5. one CVI compact candidate selected before final-test access.

Restricted or noncommercial datasets and weights stay in a research-only lane.
No research-only weight may be relabeled as the deployment candidate.

## Resource table

Every model row includes accuracy and cost together: parameters, artifact size,
embedding dimension/gallery bytes, batch-1 p50/p95 latency, batch-32 throughput,
cold start, index-build time, peak allocated/reserved/process VRAM, peak RSS/USS,
cache/index storage, average/p95 power, and baseline-subtracted energy per crop.

Real biometric gallery sizes are reported only up to actual identities. Synthetic
50K/100K unit-vector galleries may measure ANN memory and latency, never
biometric accuracy.

## Optimization admission

FP32 is the portable reference. FP16/INT8/packed-cache/ANN candidates are
promoted only under `docs/OPTIMIZATION_CONTRACT.md`. For quantization, require:

- Rank-1 and mAP paired-CI lower bound no worse than -0.5 percentage point;
- FNMR increase CI upper bound no more than +1.0 percentage point;
- no increase in wrong automatic identity assignments;
- at least 99.9% Top-1 agreement and 100% status agreement;
- at least 20% latency/energy reduction or 25% storage reduction, with a
  conservative improvement lower bound of at least 10%.

## Preregistered feasibility GO boundary

The initial hospital-request package targets:

- YT paired random-background Rank-1 at least 60% and mAP at least 42%;
- Sibetan zero-shot Rank-1 at least 80% and mAP at least 65%, with bootstrap
  lower bounds at least 70% and 55%;
- MPDD zero-shot Rank-1 at least 75% and mAP at least 60%;
- protected full-body TPIR at observed FPIR no greater than 1e-2 of at least
  60%, automatic coverage at least 50%, and accepted wrong-ID risk at most 2%;
- compact RTX 5080 path batch-1 p95 no more than 50 ms, batch-32 throughput at
  least 100 crops/s, peak VRAM no more than 4 GiB, and process RAM no more than
  8 GiB.

These are preregistered engineering targets, not achieved results. If duplicate
or label conflicts remain, BIFOR reproduction differs by more than 2 percentage
points in two clean runs, external mAP remains below 50 after three strong
models, or open-set TPIR at FPIR 1e-2 remains below 40%, the corresponding claim
is frozen and the method/data protocol is redesigned.

The accepted wrong-ID target is evaluated by its one-sided 95% upper bound, not
only the point estimate. Zero accepted known queries yields undefined
conditional risk rather than zero risk. At 50% coverage over 300 known test
identities, zero wrong accepted identities gives an upper bound of about 1.98%,
illustrating why risk and coverage must be reported together.
