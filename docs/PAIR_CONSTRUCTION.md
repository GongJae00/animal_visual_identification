# Bounded Oracle Pair Construction

## Evaluation-only boundary

Breed, coat, size, cage history, and registered identity are used only to
construct and audit oracle evaluation pairs. They are forbidden inputs to
visual embedding, similarity, threshold, and inference code. Pair construction
therefore lives outside the visual model path.

The generated artifacts are split:

- scoring requests contain only opaque `pair_id` and opaque query/reference
  artifact tokens;
- protected artifact bindings map tokens to original sample IDs for crop
  export;
- sealed ground truth contains dog IDs, session IDs, and hard-negative stratum.

The scorer receives only the first artifact and token-keyed image/embedding
artifacts. It does not receive original filenames or paths. Joining scores to
labels by `pair_id` happens in the evaluator after scoring is frozen. Pair IDs
do not hash the stratum, so the small stratum vocabulary cannot be brute-forced
from the request.

## Sampling unit and bound

The generator first selects query sessions, then negative identities, then one
reference template per selected negative identity. It never enumerates every
query/reference template pair.

For \(Q_d\) selected query sessions per dog, positive quota \(P\), and negative
stratum quotas \(K_s\),

\[
N_{\mathrm{pairs}}
\leq
\sum_d Q_d\left(P+\sum_s K_s\right).
\]

Both \(Q_d\) and the total pairs per query are hard policy caps. Multiple
tracklets from one session contribute at most one query representative.
Multiple gallery templates for one negative dog do not increase that dog's
sampling probability.

Attribute and cage inverted indices require \(O(D+R)\) storage for \(D\)
identities and \(R\) admitted records. Gallery records are reduced to one
deterministic template per dog/session before query pairing. Pair output is
linear in the declared quota rather than the Cartesian product. Broad pools are
traversed from a content-hash-derived rotation over sorted identity IDs,
avoiding process-dependent randomness.

Each stratum has a frozen candidate-scan cap \(L\). Ignoring the bounded index
construction, query selection is \(O(QSL)\) for \(Q\) queries and \(S\)
strata. The receipt records candidates scanned and whether the limit caused a
shortfall. This trades silent worst-case work for explicit evaluation coverage;
raising \(L\) is a policy change, not a hidden retry.

## Ordered hard-negative strata

Typical order:

1. `SAME_BREED`, only for non-mixed labels above a frozen confidence;
2. `SAME_COAT`, using an exact normalized color/pattern signature;
3. `SAME_SIZE`;
4. `SHARED_CAGE_HISTORY`;
5. `RANDOM`, if declared, always last.

A negative identity may appear only once per query. Every candidate belongs to
its earliest matching declared stratum, even if that stratum's quota is already
full; it cannot be relabeled as an easier later stratum. If a hard stratum
cannot fill its quota, the shortfall is reported; it is never silently
backfilled and relabeled as hard. Unknown or mixed breed metadata cannot create
same-breed pairs.

Every pair crosses sessions. The split manifest must pass before construction,
and the result binds split, policy, and attribute hashes. These structural
checks do not prove that the resulting number of dogs or hard errors is
statistically sufficient.

The construction tool writes four new files in one protected directory and
refuses overwrite:

```bash
uv run python tools/construct_verification_pairs.py \
  --split-manifest /protected/split.json \
  --dog-attributes /protected/dog_attributes.json \
  --pairing-policy /protected/pairing_policy.json \
  --scoring-output /protected/pairs/scoring.json \
  --binding-output /protected/pairs/bindings.json \
  --ground-truth-output /protected/pairs/ground_truth.json \
  --summary-output /protected/pairs/summary.json
```
