# Architecture

## Current Implementation

The canonical runtime is `cvi.CVI`, defined in `src/cvi/api.py`. It accepts a
user-provided `PIL.Image` crop and performs local, crop-level closed-set
retrieval. It does not invoke video decoding, detection, tracking, or temporal
aggregation.

```text
strict config v2
       |
       v
configured evidence extractors
       |
       v
EvidenceObservation per channel
       |
       +--> required channel unavailable: fail closed
       +--> optional channel unavailable: record absence
       |
       v
versioned gallery
  required vectors: dense FAISS IndexFlatIP storage
  optional vectors: sparse sidecar storage
       |
       v
exact available-intersection weighted cosine scoring
       |
       v
maximum template score per registered UUIDv5 identity
       |
       v
ordered Match candidates
```

### Configuration And Construction

`CVI` accepts a Python dictionary, strict JSON text, or a JSON file path. It
rejects unknown keys, duplicate JSON keys, non-finite values, unsupported modes,
implicit optional evidence, and enabled open-set behavior. Config v2 requires
`optional_channels`, and at least one configured channel must remain required.

Each channel is constructed from its exact channel schema. `CVI` rejects the
legacy `dinov2` and `appearance` types and has no public branch that executes an
unpinned Torch Hub loader. The retained `dinov2_local` channel validates
receipt-bound local model and preprocessing artifacts, loads local files only,
and disables remote-code trust. Other ONNX, landmark, nose, and MiewID paths
likewise require their corresponding local files and manifests. The repository
does not bundle those artifacts.

### Enrollment

`CVI.enroll` requires a canonical UUIDv5 registered identity. The helper
`compute_registered_dog_id` deterministically derives it from a stable source
identity using the CVI namespace. The pipeline hashes image mode, dimensions,
and pixels, extracts each configured channel, validates finite non-zero vectors,
and rejects conflicting template content or idempotency keys.

Multiple templates may be enrolled for one identity. A single image payload
cannot be bound to different identities in one gallery.

### Search And Scoring

Search extracts the same evidence contract used to create the gallery. Required
channels must be available for every query and template. Optional channels are
scored only when present on both sides. Configured weights are renormalized over
that intersection, and each channel contributes cosine similarity.

The current gallery implementation evaluates stored templates exactly and then
keeps the maximum-scoring template for each identity. `Match` exposes the
candidate identity, similarity, channel evidence, availability, scorer hash,
and exactness marker. No acceptance threshold is applied by `CVI`.

### Persistence And Concurrency

`src/cvi/index/hierarchical.py` persists gallery manifest v4 plus
content-addressed index and sidecar files. Loading validates the embedding
contract, dimensions, scorer, cardinalities, hashes, normalized vectors, and
template metadata. Publishing uses temporary files followed by `os.replace`.

The writable gallery uses a non-blocking `fcntl` lock and permits one writer.
This makes Linux/POSIX filesystem semantics part of the current support
boundary. The public API does not expose a read-only multi-process service
contract.

### Implemented But Non-Canonical Components

The repository also contains detection, evaluation, training, temporal,
open-set, ONNX measurement, and protected-evaluation components. Their presence
does not mean they are connected to `CVI` or admitted for production use.
Notably, `CVIDeploymentCPU` and `CVIDeploymentCUDA` intentionally fail closed,
and `CVI` rejects enabled open-set configuration.

## Module Map

| Module | Current role |
|---|---|
| `src/cvi/api.py` | Public configuration, enrollment, search, explanation, and persistence |
| `src/cvi/pipeline/` | Evidence orchestration for crop enrollment and search |
| `src/cvi/evidence/` | Extractors, availability, model manifests, and parity contracts |
| `src/cvi/fusion/` | Weight representation and research calibration/aggregation utilities |
| `src/cvi/index/` | Versioned gallery storage and exact candidate scoring |
| `src/cvi/identity_registry.py` | Deterministic UUIDv5 identity registry |
| `src/cvi/evaluation/` | Metric implementations outside the canonical decision path |
| `src/cvi/deployment/` | Disabled deployment facades reserved for future integration |

## Roadmap

Future integration is gate-driven rather than a statement of current
capability. See [Roadmap](ROADMAP.md) for the required evidence, calibration,
video, and deployment gates, and [Known Limitations](KNOWN_LIMITATIONS.md) for
the present boundary.
