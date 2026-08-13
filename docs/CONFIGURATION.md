# Configuration

## Retrieval Config V2

The canonical configuration schema is `cvi.retrieval_config.v2`. The top-level
[README](../README.md) shows only the public API shape; exact field and channel
requirements are maintained here. A ready-to-run retrieval JSON file is
intentionally not shipped because artifact-backed channels require
user-specific, locally verified paths.

`IdentityEngine` accepts a dictionary, JSON text, `pathlib.Path`, or string file path. JSON
is parsed strictly: duplicate keys, non-finite numbers, excessive structure,
unknown top-level fields, and non-object roots are rejected.

| Field | Requirement | Meaning |
|---|---|---|
| `schema_version` | Required | Must be `cvi.retrieval_config.v2` for new configurations |
| `mode` | Required | Must be `closed_set_retrieval` |
| `index_dir` | Required | Explicit non-empty path to the local gallery directory |
| `channels` | Required | Non-empty object of named evidence channel specifications |
| `optional_channels` | Required in v2 | Explicit list of configured channel names allowed to be unavailable |
| `fusion_weights` | Optional | Non-negative positional weights with positive total |
| `fused_dim` | Optional | Must equal the sum of all configured channel dimensions |
| `open_set` | Optional | Omit it or set disabled; enabled open-set is rejected |

The compatibility-facing names `index_dir`, `fusion_weights`, and `fused_dim`
remain part of config v2. Internally they configure the K/V gallery directory,
the exact availability-aware QK channel weights, and the total channel embedding
dimension. They do not imply a fused query vector or an attention mechanism.

At least one channel must be required. A required channel that cannot produce
evidence aborts enrollment or search. An optional channel records availability
and contributes only when present for both query and template.

Configuration objects are canonicalized with sorted JSON keys. Consequently,
`fusion_weights` correspond to lexicographically sorted channel names. The
normalized weights and channel contract are bound into the gallery scorer hash;
changing them requires a separate gallery or an explicit migration.

## Channel Types

| Type | Boundary |
|---|---|
| `dinov2_local` | Source-bound local Hugging Face DINOv2 files plus weight and preprocessor intake bundles; remote loading and remote-code trust are disabled |
| `miewid`, `miewid_reid`, `wildlife_reid` | ONNX model, exact manifest, parity receipt, and CPU/CUDA device |
| `dogfacenet_onnx` | ONNX model plus exact DogFaceNet manifest |
| `convnext_onnx` | ONNX model plus exact ConvNeXt manifest |
| `petreid_nose_onnx` | ONNX model plus exact Pet-ReID manifest |
| `landmark_onnx` | Keypoint and graph models with both manifests and explicit device |
| `nose_print_onnx` | Detector, embedding, ROI policy, manifests, device, and optional mask bundle |

The unbound `dinov2` and `appearance` type names are rejected by `IdentityEngine`; there is
no public opt-in for the unpinned Torch Hub loader. For DINOv2, only the exact
`dinov2_local` schema is accepted. Its `model_dir`, `weight_intake_bundle`, and
`preprocessor_intake_bundle` must refer to admitted local artifacts, and
`device` must be explicitly `cpu` or `cuda`.

The exact accepted keys are enforced in `canine_identity/engine.py`; extra keys are errors.
Artifact presence alone is not performance or deployment admission. See
[Data and Models](DATA_AND_MODELS.md) and
[Known Limitations](KNOWN_LIMITATIONS.md).

## Gallery Compatibility

The gallery binds channel names, dimensions, optional status, channel
configuration, artifact identifiers, fusion algorithm, and normalized weights.
Opening an existing `index_dir` with a different contract fails rather than
silently rebuilding or mixing embeddings.

Use a new gallery directory when experimenting with a different model,
preprocessor, channel set, optional-channel policy, or fusion weight. The
directory in the README example is generated runtime state and should not be
committed.

## Other Config Files

Tracked configs live with their owning packages: public examples under
`canine_identity/configs/`, operational policies under `operations/configs/`,
data policies under `data_pipeline/configs/`, schemas under
`contracts/configs/`, and research definitions under
`experiments/configs/`. Their independent schemas are not interchangeable with
retrieval config v2 and must not be passed to `IdentityEngine` unless they
explicitly declare that schema.
