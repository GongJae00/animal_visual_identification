# Contributing

Contributions that improve correctness, portability, documentation, tests, or
reproducible evaluation are welcome. CVI is research software, so changes must
distinguish implemented behavior from proposed capability.

## Before Opening A Change

- Search existing issues and pull requests for related work.
- For substantial API, schema, or evaluation changes, open an issue describing
  the problem and compatibility impact before implementation.
- Do not post vulnerabilities, credentials, private images, owner information,
  licensed datasets, or model weights in a public issue. Follow
  [SECURITY.md](SECURITY.md) for vulnerabilities.

## Development Setup

CVI development is supported on Linux with Python 3.12 and `uv`.
Choose one runtime lane:

```bash
uv sync --extra cpu --extra data --extra models --extra training --group dev
# For CUDA work, replace --extra cpu with --extra cuda.
```

Do not combine `cpu` and `cuda`; they install different ONNX Runtime packages.
External data and models are optional and are not needed for documentation-only
changes.

## Testing

Run the smallest relevant test target while developing, then run the complete
suite when the required optional dependencies are available.

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

Tests requiring external data, model artifacts, CUDA, or network access should
be explicit about those prerequisites and should have a deterministic local
contract test where practical. Do not turn synthetic or unit-test success into
a performance claim.

## Pull Requests

A pull request should:

- Explain the user-visible problem and the chosen behavior.
- Identify compatibility effects on public APIs, JSON schemas, galleries, and
  model or data contracts.
- Include regression tests for fixes and failure-path tests for validation.
- Update documentation when commands, paths, configuration, or support status
  changes.
- State the commands run and any tests or platforms not exercised.
- Contain no generated model, dataset, gallery, receipt, cache, or experiment
  artifact.

Keep changes focused. Avoid unrelated formatting, broad file moves, silent
fallbacks, and compatibility shims without a concrete persisted or external
consumer requirement.

## Data, Models, And Results

Contributors must have the right to use every submitted artifact and must not
assume that Apache-2.0 covers third-party data or weights. Benchmark results
must identify the protocol, partitioning, model and preprocessing artifacts,
metric definition, and uncertainty or confidence interval where applicable.
Results without reproducible provenance should be described as exploratory.

## Licensing

By submitting a contribution, you agree that it may be distributed under the
repository's [Apache License 2.0](LICENSE). Do not submit code or content whose
terms are incompatible with that license.
