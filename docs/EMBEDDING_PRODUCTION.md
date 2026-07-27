# Frozen Embedding Production

## Boundary

`cvi.embedding_producer` is the label-blind boundary between admitted image
artifacts and the float32 embedding cache used by the reference scorer. It does
not choose a model, train a model, calibrate a biometric threshold, or prove
that an optimized backend is safe.

This research production path is not connected to the public `cvi.CVI` runtime
and is not a supported CPU or CUDA deployment facade.

The backend receives only an ordered tuple of opaque artifact paths. The
producer binds and verifies:

- exact scoring-inventory content;
- model bytes declared by the producer and independently reported by the
  backend as the exact bytes loaded into the runtime;
- model license/lineage manifest bytes;
- preprocessing contract bytes;
- dependency-lock bytes and code revision;
- backend, runtime, execution provider, device, precision, and determinism
  declaration;
- fixed input tensor shape/value width, batch size, vector dimension,
  normalization rule, and vector format;
- production and cache resource policies.

All four provenance files and every input artifact are hashed before inference
and again before output publication. Symlinked inputs and nonempty output
directories fail closed.

## Exact-content reuse

Let \(N\) be artifact tokens, \(U\) unique pixel-file contents, \(D\) embedding
dimension, and \(B\) batch size. The producer performs \(U\), not \(N\), cache
production evaluations:

\[
C_{\text{saved}} = N-U.
\]

Aliases are authorized only by matching SHA-256 file content. Token equality,
dog identity, cage, camera, session, or control label never authorizes reuse.
Each token remains in the cache bindings, while byte-identical artifacts point
to the same content-derived vector key.

The canonical cache storage is

\[
S_{\text{cache}} = 4UD
\]

bytes for little-endian float32 output, excluding filesystem and manifest
overhead. Lower-precision model execution does not silently change this
reference storage/score format.

## Numerical contract

For every backend output \(v\), the producer requires:

1. exactly one row per input in the original order;
2. exactly \(D\) finite real values;
3. a stable scaled L2 norm greater than the frozen epsilon;
4. canonical L2 normalization;
5. float32 rounding followed by a second normalization correction;
6. independent cache rehash, finite scan, and norm verification.

The stable norm avoids intermediate overflow for extreme finite values. This
is an interchange contract, not evidence that FP16 or INT8 preserves
biometric geometry.

## Memory and work bounds

For fixed input shape \(H\times W\times C\), input value width \(w\), and
canonical vector width four bytes, the nominal batch tensors are bounded by

\[
M_{\text{input,nominal}} = BHWCw,\qquad
M_{\text{output,canonical}} = 4BD.
\]

The policy separately caps compressed artifact bytes per batch, nominal input
tensor bytes, canonical output bytes, total input/output bytes, model and
provenance file sizes, batch count, and wall time. The receipt reports:

- warm-up and production backend evaluations separately;
- warm-up, production, and total backend wall time;
- input/provenance integrity read passes and bytes;
- deduplicated calls saved;
- nominal peak batch input/output sizes;
- measured CPU peak RSS, device peak memory, and energy when the backend can
  attribute them.

Unavailable resource telemetry remains `UNAVAILABLE`; it is never represented
as zero. Backend-internal decode buffers, allocator reserve, engine workspace,
and temporary activations require actual runtime measurement and are not
proven by the nominal formulas.

The strict optional ONNX CPU reference additionally fixes the decoder version,
source mode/format/dimensions, RGB or grayscale conversion, resize operation
order, interpolation, channel order, normalization, tensor layout and dtype.
It rejects alpha/transparency, ICC profiles, external-data ONNX sidecars,
custom-op libraries, model-embedded ORT configuration, undeclared I/O metadata,
and provider fallback. It passes the in-memory bytes whose digest it reports
directly to ONNX Runtime.

The guarded CUDA reference uses the `cuda` extra, which must not be installed
with the mutually exclusive `cpu` extra. The dependency lock and runtime
manifest bind the exact runtime libraries. The adapter preloads those declared
libraries, sets `enable_fallback=0`, calls `disable_fallback()`, and also sets
`session.disable_cpu_ep_fallback=1`. These controls are distinct:
the first pair prevents Python session recreation after provider failure, while
the session entry rejects graph nodes that CUDA EP cannot execute. Session
creation must still report the expected CUDA-plus-registered-CPU provider list
and the complete frozen CUDA option map. A CUDA provider advertised by
`get_available_providers()` is not treated as ready until this session smoke
passes.

The initial CUDA contract uses host input and requested CPU output through
`session.run`; therefore its observed inference scope includes H2D, execution,
and D2H completion. It disables TF32, CUDA Graphs, external streams/allocators,
NHWC preference, maximum cuDNN workspace, tunable-op tuning, and separate copy
streams. I/O binding, TF32, CUDA Graph capture, exhaustive algorithm search,
and larger workspaces are separate optimization candidates, never silent
defaults.

The protected production path runs in a sanitized, isolated Python worker under
the bounded process supervisor. The coordinator validates the external
precommitment, exact worker environment, execution policy, input/provenance
files, and every top-level `cvi/*.py` source before launch; it never imports
ONNX Runtime. It copies the complete Python package and every unique input
content into private, byte-capped snapshots. A fixed, hash-bound `-c` bootstrap
places only that package snapshot ahead of site packages, so the child does not
execute the editable worktree. The worker rederives the source manifest and
precommitment from the snapshot before importing ONNX Runtime and creating a
session. Timeout, output overflow, crash, provider mismatch, surviving
descendants, or input/source mutation produces no approved receipt.

Each discovery or strict attempt advances a content-hashed attempt ledger. Two
distinct, increasing, ledger-chained discovery receipts with independently
archived hashes are required to freeze the executable-library set. Discovery
never publishes a cache. The strict rerun writes into coordinator-owned
same-filesystem staging and publishes the complete directory with Linux
`renameat2(RENAME_NOREPLACE)` where available. WSL DrvFS uses an atomic empty-
directory reservation followed by same-filesystem rename because it rejects
`RENAME_NOREPLACE`; the receipt records which strategy ran. Both paths are
followed by directory and parent `fsync` and a full cache re-verification, and
neither replaces an existing or populated output path. See
`docs/STORAGE_SECURITY.md` for the crash/orphan and ACL boundary.

The externally admissible artifact is
`cvi.embedding_production_bundle.v2`. Its outer receipt binds the inner
mathematical receipt, runtime-library manifest, exact ORT lane and provider
options, worker environment, supervisor result, snapshot accounting,
atomic-publication status, and completed-attempt ledger head. Protected
downstream tools reject legacy inner-only v1 bundles and require the archived
outer receipt hash and ledger-head hash again. A self-consistent rewritten JSON
file is therefore insufficient.

This is an execution-integrity boundary for ordinary concurrent worktree and
data-pipeline changes, not an OS security sandbox against a hostile process
running as the same Unix user. The latter can inspect or alter peer-process
state and would require a separate privilege boundary or Linux sealed-memfd
package lane.

Batch timings in `cvi.embedding_production_receipt.v1` describe the one-pass
cache-production run. They are explicitly
`OBSERVATIONAL_CACHE_PRODUCTION_ONLY_NOT_PROMOTION_EVIDENCE`.

The canonical cache still stores one float32 file per unique content. Replacing
it with a contiguous vector pack and canonical offset index is a separate
storage/metadata optimization candidate; it must pass exact cache semantics,
score/rank drift, crash consistency, and protected non-inferiority before
promotion.

## Optimization promotion

The order of attack is:

1. exact-content cache reuse and removal of redundant calls;
2. matched batching and input/ROI reuse;
3. guarded FP16 or mixed-precision execution;
4. compiled/optimized runtime on the same frozen model and preprocessing;
5. component-wise INT8 PTQ with representative RGB, IR, transition, quality,
   coat, and hard-negative calibration data;
6. selective higher-precision fallback or QAT only where PTQ fails;
7. distillation or architecture changes only after the stronger reference is
   established.

Normalization, similarity accumulation, thresholding, margin computation, and
calibration remain FP32/reference precision until separately admitted.

A candidate backend is not promoted from embedding cosine drift alone. It must
use the same protected test evidence and pass the non-inferiority rule for
frozen operating-point safety metrics, cross-modal directions, hard-negative
subgroups, and evidence coverage. It must also show a strict uncertainty-aware
improvement in at least one measured resource metric using
`docs/OPTIMIZATION_CONTRACT.md`. Otherwise the decision is `INCONCLUSIVE` or
`REJECT`.

## Current limitation

Optional ONNX Runtime CPU and guarded CUDA adapters are admitted only as
contract references. Tiny synthetic-graph integration smokes exercise strict
model, preprocessing, provider, ordering, normalization, deduplication,
fallback rejection, immutable input snapshotting, supervised execution,
runtime discovery, attempt chaining, and atomic publication on those graphs.
They do
not establish a canine recognizer, biometric accuracy, memory fit, throughput,
energy use, general CPU/CUDA equivalence, or optimization safety. Those require
a frozen, license-admissible oracle model and actual RGB/IR workload.
