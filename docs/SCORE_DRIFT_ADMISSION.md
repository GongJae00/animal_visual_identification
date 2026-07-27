# Score, Rank, and Frozen-Boundary Drift Admission

## Purpose

`cvi.score_drift_admission` is the second label-blind safety gate after
canonical embedding numerical admission. It applies the same opaque
query×candidate workload to content-addressed reference and candidate caches and
measures whether embedding drift changes retrieval behavior.

The gate receives no dog ID, session ID, identity label, positive/negative
pair label, filename, camera ID, cage ID, or expected dog ID. Query groups,
candidate slots, and request IDs are lowercase SHA-256 tokens. Artifact tokens
are used only to locate rehashed cache vectors. Hashing supplies content
addressing and tamper detection; it is not proof of author identity or a
cryptographic signature.

Passing is not biometric non-inferiority and never promotes an optimization.
It only shows that the declared score, rank, Top-K, and frozen score/margin
limits passed on one exact workload.

## Frozen inputs

An admissible run binds all of the following:

- canonical, sorted query×candidate requests and their exact workload hash;
- gallery, RGB/IR pairing policy, retrieval plan, split manifest, and workload
  construction receipt hashes;
- fixed `OPTIMIZATION_DEV` role and an explicit operator attestation that
  workload selection did not inspect candidate outputs;
- exact label-blind artifact inventory;
- reference and candidate committed fresh-worker production bundles, their
  independently archived outer receipt and completed-attempt-ledger hashes,
  producer configurations, cache manifests, cache verifications, and closed
  cache directories;
- a passing numerical-admission bundle and policy, recomputed from those exact
  cache bytes during this gate rather than trusted as a claimed PASS;
- a pre-existing score threshold, Top-1/Top-2 margin threshold, Top-K list,
  reference model, calibration manifest, gallery, and pairing policy;
- the exact normalized-vector score accumulation and deterministic ranking
  semantics used when that boundary was calibrated;
- a candidate-output-blind precommitment, created before candidate production,
  that freezes workload, inventory, reference receipt, candidate configuration,
  numerical/score/cache policies, boundary, prior attempt-ledger head, unique
  attempt token, and monotone attempt sequence;
- a post-production admission plan that embeds that precommitment and binds the
  candidate production receipt, numerical receipt, and advanced attempt-ledger
  head;
- externally archived expected precommitment and plan hashes supplied again at
  comparison time.

The code does not search, tune, or repair thresholds. The frozen boundary uses
the exact rules

\[
\operatorname{accept}(q) =
[s_1(q) \geq \tau_s]\,[s_1(q)-s_2(q) \geq \tau_m].
\]

This is a score/margin stability diagnostic, not the complete quality-aware,
uncertainty-aware, temporal open-set policy.
The margin threshold must be strictly positive, so an exact Top-1 tie cannot
be accepted through an opaque-slot tie break.
Because the score is the dot product of unit vectors, the score threshold is
restricted to `[-1, 1]` and the margin threshold to `(0, 2]`; this blocks
vacuous impossible boundaries but does not validate the referenced calibration
receipt's statistical correctness.

## Measures

For every opaque request, reference score \(s_r\) and candidate score \(s_c\)
are float64-compensated dot products of rehashed L2-normalized float32
vectors. The receipt reports maximum and compensated-mean
\(|s_r-s_c|\).

Ranking is deterministic: descending score followed by ascending opaque
candidate-slot token. Per query, the gate computes:

- exact Top-1 changes;
- exact Top-K set changes and symmetric-difference item count;
- Top-1/Top-2 margin drift;
- full-ranking query changes and maximum rank displacement;
- exact inversion count using merge-sort counting;
- exact score and margin threshold-decision flips, separated into reject→accept
  and accept→reject directions;
- exact-tie pair counts and distance to each frozen boundary.

Rank inversion work is \(O(G\log G)\) per query rather than \(O(G^2)\), where
\(G\) is the query gallery size. Dense Top-K inspection uses one incremental
symmetric-difference sweep through \(\max K\), not repeated set construction.
Total score/rank work is
\(O(PD+\sum_q G_q\log G_q+\sum_q\max K_q)\). Requests must be grouped canonically, so the
implementation retains only the current query's score rows and inversion
buffers: \(O(G_{max})\) additional rank memory. The current strict JSON parser
materializes the immutable request tuple, so total controller memory is
\(O(P+N+G_{max}+D)\) for \(P\) requests and \(N\) inventory/cache entries; a
future streaming manifest format is
required before claiming \(O(G_{max}+D)\) total memory at very large gallery
stress scale. Raw vector payload memory is three vectors, \(O(D)\), because
each reference/candidate query is loaded once and candidates are streamed one
backend at a time.

Cache directories are fully reverified before and after scoring, and numerical
admission is recomputed once. Every vector
payload is also size-, stat-, and hash-verified when read. The receipt separates
cache-verification, numerical-recomputation, and scoring bytes, scalar products,
and raw vector-payload peak. Each and their total have preflight caps.
Python/runtime object overhead is not claimed as measured peak RSS.

## Decision

The policy independently bounds maximum/mean score drift, maximum/mean margin
drift, inversion count, rank-changed queries, maximum displacement, Top-1
changes, aggregate and per-K set/item drift, aggregate decision flips, and each
decision-flip direction. Any exceeded bound produces
`SCORE_RANK_THRESHOLD_FAIL`; otherwise the result is
`SCORE_RANK_THRESHOLD_PASS_ON_FROZEN_WORKLOAD`.

In both cases the optimization promotion decision is fixed to `INCONCLUSIVE`.
The next protected gate still requires identity labels, session-disjoint test
data, frozen operating points, subgroup results, statistical non-inferiority,
and a resource improvement with uncertainty.

## Protected CLI

Before candidate production, create and externally archive a precommitment. The
attempt token must be unique and the prior ledger head/sequence must come from
the append-only optimization-attempt controller:

```bash
uv run python tools/create_score_drift_precommitment.py \
  --workload RETRIEVAL_WORKLOAD.json \
  --inventory INVENTORY.json \
  --reference-production REFERENCE_PRODUCTION_BUNDLE.json \
  --expected-reference-production-receipt-sha256 ARCHIVED_REFERENCE_RECEIPT_HASH \
  --expected-reference-completed-attempt-ledger-head-sha256 ARCHIVED_REFERENCE_LEDGER_HEAD \
  --reference-producer-config REFERENCE_CONFIG.json \
  --candidate-producer-config CANDIDATE_CONFIG.json \
  --numerical-policy NUMERICAL_POLICY.json \
  --frozen-boundary FROZEN_BOUNDARY.json \
  --score-drift-policy configs/research/contracts/score_drift_policy.example.json \
  --cache-policy CACHE_POLICY.json \
  --prior-attempt-ledger-sha256 PRIOR_LEDGER_HEAD \
  --candidate-attempt-token UNIQUE_ATTEMPT_TOKEN \
  --precommitment-sequence 1 \
  --precommitment SCORE_DRIFT_PRECOMMITMENT.json
```

Only after candidate production and numerical comparison, advance the external
attempt ledger and bind the resulting artifacts in a post-production plan:

```bash
uv run python tools/create_score_drift_plan.py \
  --workload RETRIEVAL_WORKLOAD.json \
  --inventory INVENTORY.json \
  --reference-production REFERENCE_PRODUCTION_BUNDLE.json \
  --candidate-production CANDIDATE_PRODUCTION_BUNDLE.json \
  --expected-reference-production-receipt-sha256 ARCHIVED_REFERENCE_RECEIPT_HASH \
  --expected-reference-completed-attempt-ledger-head-sha256 ARCHIVED_REFERENCE_LEDGER_HEAD \
  --expected-candidate-production-receipt-sha256 ARCHIVED_CANDIDATE_RECEIPT_HASH \
  --expected-candidate-completed-attempt-ledger-head-sha256 ARCHIVED_CANDIDATE_LEDGER_HEAD \
  --reference-producer-config REFERENCE_CONFIG.json \
  --candidate-producer-config CANDIDATE_CONFIG.json \
  --numerical-admission NUMERICAL_BUNDLE.json \
  --numerical-policy NUMERICAL_POLICY.json \
  --frozen-boundary FROZEN_BOUNDARY.json \
  --score-drift-policy configs/research/contracts/score_drift_policy.example.json \
  --cache-policy CACHE_POLICY.json \
  --precommitment SCORE_DRIFT_PRECOMMITMENT.json \
  --plan SCORE_DRIFT_PLAN.json
```

The plan derives the next ledger head as a hash-chain entry over the prior
head, precommitment hash, attempt token, sequence, candidate-production receipt,
and numerical-admission receipt. Supplying an arbitrary different head is not
accepted. The external controller must still persist that exact entry; the plan
cannot prove external storage by itself.

Then execute only that frozen plan:

```bash
uv run python tools/compare_score_drift.py \
  --workload RETRIEVAL_WORKLOAD.json \
  --inventory INVENTORY.json \
  --reference-cache-directory REFERENCE_CACHE \
  --candidate-cache-directory CANDIDATE_CACHE \
  --reference-cache-manifest REFERENCE_MANIFEST.json \
  --candidate-cache-manifest CANDIDATE_MANIFEST.json \
  --reference-cache-verification REFERENCE_VERIFY.json \
  --candidate-cache-verification CANDIDATE_VERIFY.json \
  --reference-production REFERENCE_PRODUCTION_BUNDLE.json \
  --candidate-production CANDIDATE_PRODUCTION_BUNDLE.json \
  --expected-reference-production-receipt-sha256 ARCHIVED_REFERENCE_RECEIPT_HASH \
  --expected-reference-completed-attempt-ledger-head-sha256 ARCHIVED_REFERENCE_LEDGER_HEAD \
  --expected-candidate-production-receipt-sha256 ARCHIVED_CANDIDATE_RECEIPT_HASH \
  --expected-candidate-completed-attempt-ledger-head-sha256 ARCHIVED_CANDIDATE_LEDGER_HEAD \
  --reference-producer-config REFERENCE_CONFIG.json \
  --candidate-producer-config CANDIDATE_CONFIG.json \
  --numerical-admission NUMERICAL_BUNDLE.json \
  --numerical-policy NUMERICAL_POLICY.json \
  --frozen-boundary FROZEN_BOUNDARY.json \
  --score-drift-policy configs/research/contracts/score_drift_policy.example.json \
  --cache-policy CACHE_POLICY.json \
  --admission-plan SCORE_DRIFT_PLAN.json \
  --expected-precommitment-sha256 ARCHIVED_PRECOMMITMENT_HASH \
  --expected-admission-plan-sha256 ARCHIVED_PLAN_HASH \
  --receipt SCORE_DRIFT_RECEIPT.json
```

Immediately after the trusted comparison process returns, archive the printed
receipt hash outside the candidate process. Downstream consumers verify all
three archived values:

```bash
uv run python tools/verify_score_drift_receipt.py \
  --receipt SCORE_DRIFT_RECEIPT.json \
  --expected-precommitment-sha256 ARCHIVED_PRECOMMITMENT_HASH \
  --expected-admission-plan-sha256 ARCHIVED_PLAN_HASH \
  --expected-receipt-sha256 ARCHIVED_RECEIPT_HASH
```

All three protected outputs are mode-0600, no-overwrite, content-hashed bundles.
The expected hashes must be archived by an orchestrator outside the candidate
process before the corresponding phase. `from_dict()` proves only structural
and internal consistency; callers must additionally invoke
`verify_score_drift_receipt_external_anchors()` with all three expected hashes,
or use the protected verifier CLI. Precommitment and plan anchors do not
authenticate the computed score summary; the independently archived receipt
hash is required to detect a summary-only PASS forgery.
These SHA-256 anchors detect substitution relative to a trusted expected value;
they are not signatures, trusted timestamps, or proof that an append-only ledger
was honestly maintained.

The example
policy is an exact-equality implementation default, not a scientifically
justified FP16/INT8 tolerance. Real tolerances must be frozen before observing
candidate results.

## Current boundary

The workload's construction, gallery, pairing, and split hashes remain
external anchors: this module checks their exact frozen values but cannot prove
that their source artifacts were constructed without labels or candidate-output
selection. Opaque hashes are pseudonyms, not proof of label absence. A future
workload-construction CLI must join the admitted split/gallery manifests and
emit the frozen construction receipt; until then this is an explicit P1 risk.
The implementation does reject repeated query artifacts, duplicate source
content within query or gallery roles, and query/gallery source-content
self-matches. Modality, split, and label-blind construction semantics still
require the future source construction receipt.

Candidate slots currently represent one gallery vector each. The gate does not
yet validate multi-prototype-to-registered-identity fusion or identity-level
ranking. It is therefore a vector retrieval stability gate only.

The present numerical receipt requires the same model semantics and model bytes
while allowing backend, runtime dependency lock, and precision declaration to
differ. A weight-changing quantization or distillation candidate needs an
additional externally anchored model-transformation lineage contract before it can
enter this gate. No such candidate, canine checkpoint, real RGB/IR workload,
deployment scorer, or biometric result currently exists. Final-test data are
forbidden from this diagnostic workload.
