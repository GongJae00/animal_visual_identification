# Paired CPU↔CUDA Inference Comparison

`cvi.measurement_comparison` admits a strict CPU reference and full-graph CUDA
candidate for descriptive systems comparison. It is intentionally downstream
of both fresh-worker measurement and label-blind canonical-cache numerical
admission.

This research tool is not connected to the public `cvi.CVI` runtime and does
not establish a supported CUDA deployment path or canine identity performance.

The comparison requires exact agreement on:

- host identity and Linux boot identifier;
- model, preprocessing semantics and preprocessing file;
- dependency lock and code revision;
- strict runtime-library `PASS` in both dependency lanes;
- ordered artifact contents;
- post-preprocessing tensor hash, shape, and byte count;
- output dimension;
- fresh-process count, warm-up/repeat protocol, caps, supervisor behavior, and
  general system-work declaration.

CUDA-only telemetry fields are removed only when comparing measurement
protocols. The CUDA receipt must still attest that unrelated GPU work was
excluded. Both CPU and CUDA policies must attest that unrelated system work was
excluded. These are operator declarations, not independently verified facts.

The paired join also requires the exact reference/candidate producer configs,
canonical cache manifests, and a `NUMERICAL_PASS_ON_FROZEN_WORKLOAD` receipt.
Every measured artifact must have a vector in the numerical-admission cache.
Producer model, preprocessing, dependency, code, backend identity, input
tensor, batch, and vector-dimension bindings are rechecked.

The output reports descriptive candidate/reference ratios and deltas for cold
process/session phases, first and warm preprocessed-tensor calls, end-to-end
calls, and worker-process `ru_maxrss`. It deliberately excludes CUDA
device-wide memory, utilization, and board energy from CPU comparison because
those values are neither process-attributed nor scope-compatible.

Every receipt fixes:

```text
decision = MEASUREMENT_COMPARABLE_NOT_PROMOTED
promotion_decision = INCONCLUSIVE
```

No confidence interval or protected biometric metric exists at this stage.
Therefore even a large synthetic speedup cannot establish non-inferiority or
optimization promotion. Real promotion still requires the frozen operating
points, subgroup evidence, coverage, and uncertainty-aware strict improvement
defined in `docs/OPTIMIZATION_CONTRACT.md`.

The protected join CLI is:

```bash
uv run --extra cuda python tools/compare_onnx_measurements.py \
  --reference-benchmark CPU_BENCHMARK.json \
  --candidate-benchmark CUDA_BENCHMARK.json \
  --reference-producer-config CPU_PRODUCER.json \
  --candidate-producer-config CUDA_PRODUCER.json \
  --reference-cache-manifest CPU_CACHE_MANIFEST.json \
  --candidate-cache-manifest CUDA_CACHE_MANIFEST.json \
  --numerical-admission NUMERICAL_BUNDLE.json \
  --receipt PAIRED_RECEIPT.json
```

The output is mode 0600 and refuses overwrite.
