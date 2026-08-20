# Setup

`setup/` is the first-user entry. `pyproject.toml` and `uv.lock` stay at the
repository root because `uv` requires them there.

## Environment

Linux, Python 3.12, and [`uv`](https://docs.astral.sh/uv/). Choose one ONNX
Runtime lane; never install `cpu` and `cuda` together.

```bash
./setup/check_env.sh cpu
# CUDA host: ./setup/check_env.sh cuda
```

Equivalent manual sync:

```bash
uv sync --locked --extra cpu --extra data --extra models --extra training --group dev
```

The checker imports the declared extras, verifies the selected lane, and records
that SuperAnimal remains disabled. It does not download datasets or weights.

## Data And Models

Raw datasets, weights, galleries, caches, and run outputs stay outside Git.
This repository ships adapters and contracts only.

Resolve local roots through environment variables, then lock them in manifests.
Do not hard-code host paths into the checkout.

```bash
export CANINE_IDENTITY_DATA_DIR=/path/to/identity-data
export CANINE_IDENTITY_DATASETS_DIR=/path/to/datasets
export CANINE_IDENTITY_MODELS_DIR=/path/to/identity-model-cache
uv run python -m data.commands.download datasets --list
uv run python -m data.commands.download models --list
```

See [Data and Models](../docs/DATA_AND_MODELS.md).

## After Setup

- Public API shape: [README](../README.md)
- Package map: [Architecture](../docs/ARCHITECTURE.md)
- Stage commands: each stage `README.md` and the root README command list

```bash
uv run python -m parsing.commands.parse --help
uv run pytest tests/prototype/test_public_runtime_contracts.py
```

## Release Check

Release CI is `.github/workflows/release-ci.yml`. A local release check runs the
full tests, builds a wheel outside the source tree, inspects declared packages
and schema resources, and imports `prototype.runtime` in a clean environment.
