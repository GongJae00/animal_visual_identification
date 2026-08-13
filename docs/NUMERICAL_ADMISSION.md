# Embedding Numerical Admission

## Purpose

`cvi.numerical_admission` is a label-blind diagnostic gate between two
canonical float32 embedding caches. It is intended for CPU/CUDA, runtime,
precision, compiler, or engine candidates before expensive biometric
evaluation. It cannot promote an optimization.

An admissible comparison requires identical model and lineage, artifact token
and content bindings, preprocessing file and structured semantics, code
revision, batch size, input shape/dtype width, vector dimension, L2 rule,
normalization tolerance, warm-up count, and canonical output format. Backend
identity and dependency lock may differ and are retained separately in the two
producer configs. Any other difference invalidates the comparison before cache
bytes are read.

## Drift measures

For reference value (a) and candidate value (b), the elementwise diagnostic
records

\[
e_{abs}=|a-b|,\qquad
e_{rel}=\frac{|a-b|}{\max(|a|,|b|,\rho)}.
\]

The hard elementwise gate is

\[
|a-b| \leq a_{tol}+r_{tol}\max(|a|,|b|).
\]

For each embedding it additionally computes float64-accumulated L2 drift and

\[
d_{cos}=1-\operatorname{clip}
\left(\frac{a^Tb}{\lVert a\rVert_2\lVert b\rVert_2},-1,1\right).
\]

Float32 ULP distance is reported as a diagnostic after mapping signed IEEE-754
bit patterns to monotonic integer keys and treating positive and negative zero
as equal. ULP is not a standalone hard gate because near-zero values can have
large ULP differences with negligible geometric effect.

The implementation reads one reference and one candidate vector at a time,
uses compensated running sums, and retains no all-vector error table. Extra
memory is therefore (O(D)) bytes for dimension (D), while work is (O(UD))
for (U) unique input contents.

## Decision boundary

The only successful state is
`NUMERICAL_PASS_ON_FROZEN_WORKLOAD`. Elementwise, L2, or cosine failure returns
`NUMERICAL_FAIL`; lineage, binding, resource-cap, directory-closure, hash,
finite-value, or normalization errors fail by exception and publish no receipt.

Every receipt carries the fixed interpretation
`NUMERICAL_ADMISSION_ONLY_NOT_OPTIMIZATION_PROMOTION`. Passing this gate does
not establish threshold stability, rank stability, CPU/CUDA equivalence outside
the frozen workload, biometric non-inferiority, memory fit, latency improvement,
or energy improvement. Promotion still requires the protected operating-point,
subgroup, coverage, and resource evidence in `docs/OPTIMIZATION_CONTRACT.md`.

The protected CLI creates a content-hashed, mode-0600, no-overwrite bundle:

```bash
uv run python workflows/compare_embedding_caches.py \
  --reference-cache-directory REFERENCE_CACHE \
  --candidate-cache-directory CANDIDATE_CACHE \
  --reference-cache-manifest REFERENCE_MANIFEST.json \
  --candidate-cache-manifest CANDIDATE_MANIFEST.json \
  --reference-producer-config REFERENCE_CONFIG.json \
  --candidate-producer-config CANDIDATE_CONFIG.json \
  --policy NUMERICAL_POLICY.json \
  --receipt NUMERICAL_BUNDLE.json
```

Downstream protected CLIs accept this authenticated bundle, not an unwrapped
receipt object. A modified decision or summary therefore changes and fails the
joined receipt digest.

## Downstream gates

- connect the exact post-preprocessing tensor hash now emitted by the
  fresh-worker benchmark to the cache admission receipt itself;
- run the implemented `docs/SCORE_DRIFT_ADMISSION.md` gate on the exact opaque
  query×candidate workload;
- compare the separate supervised cold/session/first-run, warm tensor-API, and
  end-to-end receipt only on a frozen real canine workload.
