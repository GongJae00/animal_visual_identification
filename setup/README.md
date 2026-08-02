# Setup And Release Guidance

`pyproject.toml` and `uv.lock` remain at the repository root because `uv` requires them there. This directory owns setup and release guidance; it does not duplicate those tooling files.

Use one environment and one ONNX Runtime lane:

```bash
uv sync --locked --extra cpu --extra data --extra models --extra training --group dev
# Replace --extra cpu with --extra cuda for the CUDA lane.
```

Release verification is defined in `.github/workflows/release-ci.yml`. A local release check should run the full tests, build a wheel outside the source tree, inspect all declared top-level packages and schema resources, and import `canine_identity` in a clean environment. Do not place downloaded data, weights, caches, or build output in this directory.
