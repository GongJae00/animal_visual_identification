# Matched Visual Shortcut Controls

## Purpose

Shortcut controls test whether a frozen recognizer can recover identity from
background, cage, silhouette, or accessories instead of the intended dog
evidence. They are diagnostic interventions, not training augmentation and not
proof of invariance by themselves.

For base artifact \(x_i\), verified masks \(m_i\), and deterministic control
operator \(T_k\), a panel compares

\[
\{s(T_k(x_i),T_k(x_j)) : (i,j)\in P_{\mathrm{panel}}\}
\]

on one identical pair set \(P_{\mathrm{panel}}\) for every \(k\) in that
panel. No score, dog ID, session ID, stratum, or expected cage assignment may
affect membership in \(P_{\mathrm{panel}}\).

## Supported planning semantics

The planning contract supports:

- `ORIGINAL`;
- `DOG_ONLY`;
- `BACKGROUND_ONLY`;
- `BODY_BLURRED`;
- `MASK_ONLY`;
- `ACCESSORY_ONLY`;
- `ACCESSORY_MASKED`.

Every non-original operator is bound to a transform-config hash and a semantics
version. The planner now also loads the executable config manifest and refuses
recipes whose kind, config hash, or semantics version differs from the pixel
executor. Random-background replacement is intentionally deferred until a
donor policy can prove that donor selection is identity, session, cage, split,
and score blind.

Planning requires both the base-artifact verification receipt and the mask-file
verification receipt. Their manifest hashes, file counts, and total bytes must
agree exactly; the plan hashes both receipts rather than trusting unverified
manifest strings.

Before pixel transforms, `cvi.mask_semantics` additionally requires each
reviewed mask to decode as metadata-free grayscale PNG with dimensions exactly
matching its base crop and declaration. FFmpeg emits one `gray` rawvideo frame;
the verifier scans it in fixed-size chunks, accepts only byte values 0 and 255,
requires nonempty support, and checks that every verified accessory pixel lies
inside the verified dog mask up to a predeclared tolerance.

Peak additional RAM is \(O(b)\) for configured chunk size \(b\). Raw masks are
decoded sequentially to a temporary directory and deleted per artifact, so
temporary storage is bounded by two one-byte-per-pixel masks rather than the
whole dataset.

## Executable pixel equations

Let \(x\in\{0,\ldots,255\}^{H\times W\times C}\), binary dog mask
\(m\in\{0,1\}^{H\times W}\), binary accessory mask \(a\), and neutral image
\(\mathbf{0}\). The fixed `cvi.visual_control_transform.v1` operators are

\[
\begin{aligned}
T_{\mathrm{dog}}(x,m) &= m\odot x,\\
T_{\mathrm{background}}(x,m) &= (1-m)\odot x,\\
T_{\mathrm{blur}}(x,m) &= (1-m)\odot x+
  m\odot G_{\sigma}(x),\\
T_{\mathrm{mask}}(m) &= 255m,\\
T_{\mathrm{accessory}}(x,a) &= a\odot x,\\
T_{\mathrm{accessory\ masked}}(x,a) &= (1-a)\odot x.
\end{aligned}
\]

For RGB, a scalar mask value is repeated over all three channels. For IR, the
single `gray` channel is preserved. Geometry and pixel format must equal the
base artifact exactly. Neutral value zero, metadata removal, one-frame output,
single-thread filtering, and PNG prediction mode are fixed rather than
silently inherited from a workstation.

Blur strength is resolution-normalized:

\[
\sigma(H,W)=\operatorname{clip}
\left(\rho\min(H,W),\sigma_{\min},\sigma_{\max}\right).
\]

The normalized fraction, bounds, and FFmpeg approximation steps are all in the
config hash. This prevents a fixed-pixel blur from becoming weak on large crops
and destructive on small crops. The example value is provisional and must be
frozen before scoring.

The executor decodes base and masks once per base-artifact group, generates
each unique content-addressed control once, then independently decodes the
result and checks every pixel against the equation above. `BODY_BLURRED` uses
a separately decoded Gaussian reference for that comparison. The verifier
scans fixed pixel chunks; it does not load a full image into Python memory.

For \(P=HW\), channel count \(C\), distinct required mask roles \(R\), and
\(I_b\in\{0,1\}\) indicating a blur control in the group, peak raw temporary
storage is preflighted as

\[
B_{\mathrm{raw,group}} =
P(2C+R+C I_b).
\]

RAM remains \(O(Cb)\) for configured chunk size \(b\); total task-pixels,
per-file bytes, total output bytes, raw temporary bytes, process time, and task
count each have independent fail-closed ceilings. Generated PNGs accumulate
only in a sibling temporary directory up to the total-output cap. Inputs are
rehashed again before commit, and outputs are hard-linked only after every
task and receipt invariant passes. Any publication error removes partial
links.

## Matched panels and missing masks

Each panel starts with `ORIGINAL` and declares its required mask roles. A pair
is eligible only when both artifacts have reviewed `VERIFIED` masks for every
required role. Missing, unverified, and rejected masks are never treated as
empty masks. Exclusion-reason counts and eligible/selected counts by hard-pair
stratum are sealed with the evaluator so selection bias remains visible.

Panels are matched internally rather than forced into one global intersection.
For example, the background panel can use every verified dog mask, while an
accessory panel legitimately covers only samples with verified accessory
masks. Cross-panel metric differences therefore are not paired causal
comparisons.

## Leakage boundary

The scorer payload contains only opaque request and artifact tokens. Protected
transform tasks contain base-token and mask bindings. Panel/control/base-pair
bindings remain sealed until evaluation, where they join the already sealed
identity truth. Cage ID, expected dog ID, original sample paths, dog IDs, and
session IDs never enter the scoring request.

## Computational contract

Transforms and embeddings are keyed by unique content-derived artifact token:

\[
C_{\mathrm{embed}} =
U_{\mathrm{control\ artifacts}}\,c_{\mathrm{embed}},
\qquad
C_{\mathrm{score}} =
K_{\mathrm{matched\ requests}}\,c_{\mathrm{score}}.
\]

This replaces the naive \(2K\) embedding calls with one call per unique
artifact. The plan records naive calls, unique artifacts, reusable calls saved,
and transform-task count. Cache reuse is valid only under the same model and
inference-config hashes; the plan alone does not certify a cache implementation.

## Planning CLI

`workflows/plan_visual_shortcut_controls.py` reconstructs the separated pair
bundle, reparses the crop receipt, rehashes the current base-crop directory,
rehashes the closed mask directory, validates the transform config manifest
against every non-original recipe, and then creates six mode-0600,
no-overwrite outputs:

1. label-blind scoring requests;
2. protected transform tasks;
3. sealed evaluation bindings and panel summaries;
4. the fresh mask verification receipt;
5. the bounded mask-pixel semantic verification receipt;
6. the plan/cost/gate summary.

All outputs share one protected directory and are published as one
all-or-cleaned bundle. If any panel misses its declared minimum matched-pair
count, no output is written.

`workflows/execute_visual_control_transforms.py` then consumes the protected task
file, crop receipt, closed base and mask directories, both mask-verification
receipts, the same config manifest, and an execution policy. It writes a
closed control-artifact directory plus one mode-0600, no-overwrite receipt:

```bash
uv run python workflows/execute_visual_control_transforms.py \
  --transform-tasks PROTECTED/control-transform.json \
  --crop-export-receipt PROTECTED/crop-export-receipt.json \
  --base-artifact-directory DATA/oracle-crops \
  --mask-manifest PROTECTED/control-mask-manifest.json \
  --mask-directory DATA/control-masks \
  --mask-verification PROTECTED/mask-verification.json \
  --mask-semantic-verification PROTECTED/mask-semantic-verification.json \
  --transform-config-manifest PROTECTED/control-transform-configs.json \
  --execution-policy PROTECTED/control-transform-execution-policy.json \
  --output-directory DATA/control-artifacts-empty \
  --receipt-output PROTECTED/control-transform-receipt.json
```

The output directory must already exist, be empty, and not be a symlink.

## Label-blind embedding reuse and reference scoring

Control scoring must not recompute a neural embedding for every pair side.
For \(K\) requests and \(U\) distinct artifact contents, the required neural
calls are \(U\), not \(2K\). Cache identity is not an artifact token:

\[
k_{\mathrm{cache}} =
H(h_{\mathrm{pixels}},h_{\mathrm{model}},h_{\mathrm{inference}},
  h_{\mathrm{lock}},\mathrm{code\ revision},\mathrm{precision},
  D,\mathrm{vector\ format}).
\]

Thus two opaque tokens with byte-identical pixels may share one vector, while
the same pixels under a different model, preprocessing, code revision,
dependency lock, precision, dimension, or representation cannot silently
reuse it.

`cvi.control_scoring` builds an exact inventory from the scoring requests,
freshly rehashes both the original-crop and transformed-control directories,
and rejects missing tokens, base/control token collisions, or unrequested
transformed outputs. The planner's complete scoring-payload hash is carried
through the protected transform task and transform receipt, so recombining the
same artifacts into different pairs is also rejected. Cache bindings must
cover this inventory exactly. Unique
vectors are header-free little-endian float32 files named by cache key. The
admission pass combines SHA-256, finite-value checking, and L2-norm checking in
one streaming file pass. It rejects symlinks, extra files, stale content,
overly loose normalization, excess dimension/count/bytes, and trailing data.

The portable reference scorer computes cosine similarity as a dot product of
admitted L2-normalized vectors. Each pair costs \(D\) products and reads
\(8D\) bytes. Chunk-level `fsum` with Neumaier accumulation gives a
deterministic CPU reference without loading a full cache or vector set.
Scoring policy separately caps request count, scalar products, dot-product
bytes, and chunk size. The receipt reports:

- dot-product scalar products and bytes;
- the two before/after cache-verification passes and their square terms/bytes;
- total file bytes read;
- unique artifact and unique vector counts;
- neural embedding calls saved relative to naive pair-side execution;
- peak raw vector chunk bytes.

`workflows/score_visual_controls.py` accepts only opaque request IDs and artifact
tokens plus authenticated artifact/cache receipts. It has no ground-truth,
dog, session, cage, camera, stratum, panel, or control-kind input. It atomically
writes a mode-0600 inventory, cache verification, and blind score receipt.
The embedding producer is intentionally outside this reference scorer and must
create a manifest bound to the actual frozen model/inference lineage.

```bash
uv run python workflows/score_visual_controls.py \
  --scoring-requests PROTECTED/control-scoring.json \
  --crop-export-receipt PROTECTED/crop-export-receipt.json \
  --base-artifact-directory DATA/oracle-crops \
  --control-transform-receipt PROTECTED/control-transform-receipt.json \
  --control-artifact-directory DATA/control-artifacts \
  --embedding-cache-manifest PROTECTED/embedding-cache-manifest.json \
  --embedding-cache-directory DATA/control-embedding-cache \
  --embedding-cache-policy PROTECTED/embedding-cache-policy.json \
  --score-policy PROTECTED/control-score-policy.json \
  --gallery-sha256 FROZEN_GALLERY_PROTOCOL_SHA256 \
  --inventory-output PROTECTED/control-scoring-inventory.json \
  --cache-verification-output PROTECTED/cache-verification.json \
  --score-receipt-output PROTECTED/control-blind-scores.json
```

## Sealed matched-panel evaluation

Only `workflows/evaluate_visual_controls.py` receives both blind scores and sealed
registered-dog/session truth. The evaluator requires:

- the pair-set hash in the sealed binding payload to equal the reconstructed
  pair bundle;
- the supplied pairing policy hash to equal the policy frozen into that pair
  bundle, and its RGB/IR direction to equal the frozen threshold direction;
- score request IDs and sealed binding IDs to match exactly;
- every panel/control/base-pair binding to be unique;
- `ORIGINAL` plus at least one intervention per panel;
- every control in a panel to have the identical base-pair set;
- selected-pair and selected-stratum counts to match sealed truth;
- the score receipt, embedding-cache manifest, frozen model, and frozen gallery
  hashes to agree;
- the panel to meet its predeclared minimum.

For each control it applies the already frozen `score >= threshold` rule and
reuses the query-dog or query-session whole-cluster bootstrap for FMR/FNMR.
It also reports a Mann–Whitney midrank AUC point estimate and positive/negative
mean score changes relative to the matched `ORIGINAL` pair. Those latter
statistics are explicitly descriptive: they have no equivalence or causal
interpretation and no confidence interval in the current contract.

AUC computation is \(O(M\log M)\) for \(M\) rows per control and is capped by
total sort items. Threshold counting and paired deltas are \(O(M)\). The
receipt records joined bindings, metric rows, sort items, paired terms, and
maximum rows per control.

```bash
uv run python workflows/evaluate_visual_controls.py \
  --pair-scoring-requests PROTECTED/pair-scoring.json \
  --pair-artifact-bindings PROTECTED/pair-bindings.json \
  --pair-ground-truth PROTECTED/pair-ground-truth.json \
  --pair-summary PROTECTED/pair-summary.json \
  --pairing-policy PROTECTED/pairing-policy.json \
  --control-evaluation-bindings PROTECTED/control-evaluation.json \
  --blind-score-receipt PROTECTED/control-blind-scores.json \
  --embedding-cache-manifest PROTECTED/embedding-cache-manifest.json \
  --frozen-threshold PROTECTED/frozen-threshold.json \
  --bootstrap-config PROTECTED/bootstrap-config.json \
  --evaluation-policy PROTECTED/control-evaluation-policy.json \
  --evaluation-output PROTECTED/control-evaluation-receipt.json
```

## Interpretation limits

- `BACKGROUND_ONLY` retains the dog-mask hole and can expose silhouette or
  occupancy leakage; low performance does not prove general background
  invariance.
- `BODY_BLURRED` retains low-frequency color, outline, and background; it tests
  dependence on fine detail, not pure background use.
- `ACCESSORY_ONLY` includes accessory shape and position; it diagnoses an
  available shortcut but does not identify its causal subfeature.
- Passing pixel semantics still does not prove anatomical annotation quality;
  it proves decode, range, dimensions, nonempty support, and declared
  accessory containment only.
- Passing transform-equation verification proves the declared intervention was
  executed, not that its causal interpretation is complete. Mask boundaries,
  Gaussian boundary mixing, silhouette holes, and low-frequency cues remain
  part of the diagnostic.
- Passing contract tests proves matching and provenance behavior only, not
  recognizer robustness.
- Cache deduplication proves fewer neural calls for identical admitted pixels;
  it does not prove that the external embedding producer is fast, numerically
  portable across backends, or biologically valid.
- A nuisance-only AUC or threshold error rate diagnoses recoverable signal in
  the declared intervention. It does not by itself identify which pixel cue
  caused that signal or prove the original model is invariant when the metric
  is low.
