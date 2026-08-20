# Identification

In: crops (and optional Face/Nose ROIs). Out: per-channel embedding vectors.

`training/` holds Appearance/Face/Nose trainers. `export/` is inference only and
must not import `training/`. Appearance is the live end-to-end channel.

Commands: `uv run python -m identification.commands.train --help`
and `uv run python -m identification.commands.export --help`.
