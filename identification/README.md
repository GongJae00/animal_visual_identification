# Identification

In: crops (and optional Face/Nose ROIs). Out: per-channel embedding vectors.

`training/` holds Appearance/Face/Nose trainers. `export/` is inference only and
must not import `training/`. Appearance is the live end-to-end channel.

Embedding views: `visualization.commands.render --stage identification` with a
trace that contains per-channel `embeddings` plus optional `dataset`,
`identity` / `dog_id`, and `view` labels. Writes `pca2`, `pca3`,
`cosine_identity`, `pca_var`, `dim_contrib`, and `channel_gap` under
`Visualization/vis/01_identification/`.

Commands: `uv run python -m identification.commands.train --help`
and `uv run python -m identification.commands.export --help`.
Optimization levers for export channels are catalogued in `evaluation.optimization_surfaces.identification` (not measured values).
