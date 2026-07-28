# Canine Video Identity (CVI)

CVI is a research-oriented Python package for canine identity retrieval. The
current executable recognizer accepts user-supplied dog image crops, enrolls
them in a local gallery, and returns closed-set candidates for another crop.

## Current Scope

Implemented in the public `CVI` API:

- Explicit `PIL.Image` crop enrollment and search.
- Canonical UUIDv5 registered identities.
- Strict retrieval configuration and artifact contracts.
- Required and optional evidence-channel handling.
- Exact weighted cosine scoring with identity-level template aggregation.
- Versioned, integrity-checked local gallery persistence.

Not implemented as an end-to-end product:

- Video decoding, dog detection, tracking, or crop selection in the `CVI` flow.
- Calibrated unknown-dog rejection or an operational open-set decision.
- A bundled, canine-trained identity model or validated biometric performance.
- A production service, access control, encryption, or supported CPU/CUDA
  deployment facade.

Search results are candidates from an enrolled closed set. They are not a
claim that the top candidate is the same dog. Passing tests demonstrates
software behavior, not identification accuracy. See
[Known Limitations](docs/KNOWN_LIMITATIONS.md).

## Platform And Installation

The supported development environment is Linux with POSIX filesystem
semantics, Python 3.12, and
[`uv`](https://docs.astral.sh/uv/). Gallery writer locking uses `fcntl`, so
native Windows is not supported. Other POSIX operating systems are unvalidated.

From a source checkout, choose one runtime lane:

```bash
# CPU runtime and development tests
uv sync --extra cpu --group dev

# CUDA runtime and development tests
uv sync --extra cuda --group dev
```

Do not install the `cpu` and `cuda` extras together: they select different
ONNX Runtime and PyTorch distributions. Additional opt-in dependencies are
available through the `data`, `models`, and `training` extras.

Detection and pose adapters under `cvi.localization` are research-only and are
not connected to `CVI`. Ultralytics is intentionally not a package extra;
install it in a separate environment only after reviewing its AGPL terms and
the OpenCV runtime conflict described in [Third-Party Licensing](THIRD_PARTY_LICENSES.md).

```bash
uv run python -c "import cvi; print(cvi.__all__)"
uv run pytest
```

## Crop-Level Example

This is a receipt-bound `dinov2_local` config-v2 template, not a runnable
out-of-the-box example. It requires an admitted local DINOv2-small model and
matching weight and preprocessor intake bundles. Replace every `/path/to`
placeholder with those supplied local artifacts before constructing `CVI`.
The runtime does not fetch them. `optional_channels` is empty, so the appearance
channel is required. The image filenames are also placeholders for user-supplied
dog crops.

```python
from PIL import Image

from cvi import CVI
from cvi.identity_registry import compute_registered_dog_id

config = {
    "schema_version": "cvi.retrieval_config.v2",
    "mode": "closed_set_retrieval",
    "index_dir": "/path/to/cvi-gallery",
    "channels": {
        "appearance": {
            "type": "dinov2_local",
            "model_dir": "/path/to/dinov2-small",
            "weight_intake_bundle": "/path/to/weight-intake.json",
            "preprocessor_intake_bundle": "/path/to/preprocessor-intake.json",
            "device": "cpu",
        }
    },
    "optional_channels": [],
    "open_set": {"enabled": False},
}

runtime = CVI(config)
registered_dog_id = compute_registered_dog_id("local:v1:dog:001")
runtime.enroll(
    Image.open("dog_001_crop.jpg").convert("RGB"),
    registered_dog_id,
)
matches = runtime.search(
    Image.open("query_crop.jpg").convert("RGB"),
    top_k=5,
)
runtime.close()
```

Use a stable, namespace-qualified source identity with
`compute_registered_dog_id`. `CVI.enroll` rejects display names, arbitrary
strings, non-v5 UUIDs, and non-canonical UUID text. Configuration details and
artifact-backed channel requirements are documented in
[Configuration](docs/CONFIGURATION.md).

## Data And Models

Datasets, pretrained weights, generated ONNX files, galleries, and experiment
outputs are not bundled. The Apache-2.0 repository license does not grant
rights to third-party data or weights.

```bash
export CVI_DATA_DIR=/path/to/cvi-data
export CVI_MODELS_DIR=/path/to/cvi-model-cache

uv sync --extra cpu --extra data --extra models --extra training
uv run python tools/download_datasets.py --list
uv run python tools/download_models.py --list
```

`/path/to/...` values are user-selected external directories. Some dataset
handlers only print manual acquisition instructions, and some model candidates
are disabled or have unresolved license status. Read
[Data and Models](docs/DATA_AND_MODELS.md) before downloading or using any
artifact.

The external root uses content-oriented paths such as
`datasets/<dataset-name>` and `checkpoints/<model-artifact-id>`. License and
research/deployment admission are registry metadata, not directory names; a
path alone never establishes that an artifact is deployment eligible.

## Repository Layout

| Path | Purpose |
|---|---|
| `src/cvi/` | Public API, runtime components, research contracts, and evaluation code |
| `src/cvi/evidence/` | Evidence extractors and artifact validation |
| `src/cvi/index/` | Versioned local gallery and exact scoring |
| `src/cvi/pipeline/` | Crop-level enrollment and search orchestration |
| `tools/` | Source-checkout data, model, training, and evaluation commands |
| `configs/` | Versioned protocol and backend examples, not a production config |
| `tests/` | Unit, contract, synthetic, and CLI regression tests |
| `docs/` | Architecture, configuration, limitations, and research protocols |

## Project Documents

- [Architecture](docs/ARCHITECTURE.md)
- [Configuration](docs/CONFIGURATION.md)
- [Data and Models](docs/DATA_AND_MODELS.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Roadmap](docs/ROADMAP.md)
- [Contributing](CONTRIBUTING.md)
- [Security Policy](SECURITY.md)

## License

Repository code and documentation are licensed under the
[Apache License 2.0](LICENSE). Third-party datasets, model weights, source code,
and generated artifacts retain their own terms. You are responsible for
license, privacy, and deployment review for every external artifact.
