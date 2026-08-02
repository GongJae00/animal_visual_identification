# Batch-Composition Invariance Admission

This label-blind gate tests whether an artifact embedding changes because of
batch size, slot, neighbors, duplicates, a non-full tail, or repeated execution.
It is separate from cross-backend numerical admission and performance timing.

For artifact \(x_i\), singleton output \(a_i=f([x_i])_0\) is the star
reference. Fixed scenarios are `FULL_BATCH`, `PERMUTED_NEIGHBORS`,
`DUPLICATE_PACKED`, `TAIL_SIZE`, and `REPEATED_SAME_COMPOSITION`. Every raw and
L2-normalized output is compared with \(a_i\); repeated full composition must
also match its first raw float32 digest exactly. Padding is forbidden. A future
static/padded engine requires a separate logical-to-physical mapping contract
without padding-specific tolerance relaxation.

The gate reports elementwise absolute/relative error, ULP distance, raw L2 and
norm drift, normalized L2 drift, and cosine drift. For unit gallery vector
\(g\), Cauchy--Schwarz gives

\[
|g^T u-g^T v| \leq \|u-v\|_2,
\]

so normalized L2 drift upper-bounds per-template score drift. It does not prove
rank, margin, open-set, or biometric invariance.

Fixed scenarios use linear-in-\(N\) evaluations and \(O(ND)\) comparison work.
Singleton raw/normalized anchors are streamed to temporary float32 files, so
RAM is the backend batch tensor/output plus \(O(D)\) comparison state rather
than \(O(ND)\). Temporary storage is exactly \(8ND\) bytes and policy-capped.
The precommitment stores only \(O(N)\) content metadata, never embeddings.

Before inference, freeze the inventory and ordering, artifact bytes, producer
and backend identity, model/preprocessing/lock provenance, policy, exact
schedule, supervised-process policy, exact Python executable, and complete
allowlisted worker environment identity. The coordinator validates the external
precommitment before process launch and never imports ONNX Runtime. The worker
uses `-I -B`, validates its environment and every input binding before importing
ONNX Runtime, and writes only below coordinator-owned mode-0700 scratch. The
prior attempt-ledger hash, monotonically increasing sequence, and candidate
attempt token prevent an unrecorded retry from silently replacing the declared
attempt. Archive the printed precommitment hash outside the candidate result
directory.

Runtime policy creation is a separate two-pass workflow. First create a distinct
precommitment for each of at least two discovery attempts using the
discovery-only runtime policy, different attempt tokens, and increasing
sequences. Run each with a distinct `--runtime-discovery-output`. Discovery
completes the same batch workload in supervised fresh workers but cannot publish
an admission receipt. Then freeze a candidate policy from the supervised
discovery bundles:

```bash
uv run python workflows/freeze_batch_runtime_library_policy.py \
  --discovery-policy RUNTIME_DISCOVERY_POLICY.json \
  --discovery-manifest BATCH_RUNTIME_DISCOVERY_1.json \
  --discovery-manifest BATCH_RUNTIME_DISCOVERY_2.json \
  --policy STRICT_RUNTIME_LIBRARY_POLICY.json \
  --freeze-receipt BATCH_RUNTIME_POLICY_FREEZE.json
```

The binary sets, worker environment, execution policy, and exact ONNX Runtime
distribution/version must agree. Review every resolved binary path before the
strict rerun; the freeze receipt explicitly does not authorize its own
discovery run. Create a new precommitment for the strict policy after review.

```bash
uv run python workflows/create_batch_invariance_precommitment.py \
  --inventory INVENTORY.json \
  --artifact-paths ARTIFACT_PATHS.json \
  --producer-config PRODUCER_CONFIG.json \
  --model MODEL.onnx \
  --model-lineage MODEL_LINEAGE.json \
  --preprocessing-config PREPROCESSING.json \
  --dependency-lock uv.lock \
  --policy experiments/configs/benchmarks/batch_invariance_policy.example.json \
  --runtime-library-policy STRICT_RUNTIME_LIBRARY_POLICY.json \
  --worker-execution-policy operations/configs/batch_worker_execution_policy.example.json \
  --python-executable .venv-cpu/bin/python \
  --prior-attempt-ledger-sha256 PRIOR_LEDGER_SHA256 \
  --candidate-attempt-token CANDIDATE_ATTEMPT_SHA256 \
  --precommitment-sequence 1 \
  --output BATCH_PRECOMMITMENT.json
```

```bash
uv run python workflows/evaluate_batch_invariance.py \
  --backend cpu \
  --inventory INVENTORY.json \
  --artifact-paths ARTIFACT_PATHS.json \
  --producer-config PRODUCER_CONFIG.json \
  --onnx-config ONNX_CONFIG.json \
  --preprocessing-config PREPROCESSING.json \
  --model MODEL.onnx \
  --model-lineage MODEL_LINEAGE.json \
  --dependency-lock uv.lock \
  --policy experiments/configs/benchmarks/batch_invariance_policy.example.json \
  --precommitment BATCH_PRECOMMITMENT.json \
  --expected-precommitment-sha256 ARCHIVED_PRECOMMITMENT_SHA256 \
  --runtime-library-policy STRICT_RUNTIME_LIBRARY_POLICY.json \
  --worker-execution-policy operations/configs/batch_worker_execution_policy.example.json \
  --python-executable .venv-cpu/bin/python \
  --receipt BATCH_INVARIANCE_RECEIPT.json
```

After evaluation, archive the printed receipt hash separately. Verification
requires both independent anchors; editing a failure into a structurally valid
success changes the final hash and is rejected.

```bash
uv run python workflows/verify_batch_invariance_receipt.py \
  --receipt BATCH_INVARIANCE_RECEIPT.json \
  --expected-precommitment-sha256 ARCHIVED_PRECOMMITMENT_SHA256 \
  --expected-receipt-sha256 ARCHIVED_RECEIPT_SHA256
```

Use `.venv-cuda/bin/python` together with `--backend cuda` for the isolated CUDA
lane; CPU and CUDA precommitments and runtime-library policies are not
interchangeable. The final protected object is the outer supervised fresh-worker
receipt (`cvi.batch_invariance_bundle.v4`), not the inner mathematical batch
receipt. It binds the exact command, Python binary, allowlisted environment,
supervisor policy/result, ONNX Runtime distribution, worker request, runtime
manifest, and inner receipt. Legacy inner-only bundles are rejected.

The artifact-path payload accepts only schema plus token/path entries. Identity,
dog, session, cage, camera, and pair labels are not accepted or emitted. Output
is mode-0600 and no-overwrite. The example zero-drift policy is an implementation
default, not a tolerance justified for a real canine model. The fixed outcome
is non-promoting; synthetic receipts are not biometric or optimization results.
Precommitment proves only that experiment selection preceded candidate outputs;
the external final anchor proves only immutability relative to that archived
hash. Evaluation additionally requires the strict, precommitted runtime-library
policy and captures executable mappings at dependency-import, session-ready,
first-output, and final-output boundaries. The complete manifest and binary-set
hash are embedded in the receipt. Worker stdout/stderr must be empty and the
process group must contain no surviving descendants; otherwise no receipt is
issued. These contracts still do not substitute for biometric non-inferiority
or the optimization promotion contract.
