# Representation

In: encoder outputs and ROI evidence. Out: `EvidenceObservation` collections
(channel packing). Quality observations live in `quality/`.

This package has no trainers and must not import `identification.training` or
`parsing.training`.

Embedding views: `visualization.commands.render --stage representation`.
A `channels` object of named matrices plus `identity` writes `channel_gap`.

Commands: `uv run python -m representation.commands.embed --help`
(`produce`, `precommit`, `verify`, `compare`).
Optimization catalog: `evaluation/optimization_surfaces/representation.py`.
