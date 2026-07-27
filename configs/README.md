# Configuration Files

The files under `configs/` are versioned examples for several independent
schemas. They are not a single application configuration set, and none is a
ready-made production deployment.

| Path | Contents |
|---|---|
| `configs/pipeline/evidence/` | Image preprocessing and evidence-coverage examples |
| `configs/deployment/` | ONNX measurement, runtime discovery, worker, and production-policy examples |
| `configs/research/contracts/` | Dataset, split, duplicate, scoring, and artifact contract examples |
| `configs/research/benchmarks/` | Capacity, cache, and batch-invariance examples |
| `configs/research/processing/` | Control-transform and scoring examples |
| `configs/data/` | Crop-export policy example |
| `configs/pdq/` | Pinned PDQ implementation metadata |
| `configs/pretrained-weights/` | Pretrained weight and preprocessor intake contracts |

Each consumer validates its own exact schema. A filename ending in
`.example.json` is documentation input, not evidence that local data, models,
or runtime support exist. Do not rename an example to `production.json` and
assume it becomes deployable.

The public `CVI` API requires `cvi.retrieval_config.v2`. No retrieval JSON is
shipped because artifact-backed channels need user-specific verified paths.
Use the audited in-memory example in the [README](../README.md) and the field
reference in [Configuration](../docs/CONFIGURATION.md).

Before using any other example, identify the tool or module that consumes it,
read that consumer's validation code, and keep secrets and machine-specific
paths outside tracked files.
