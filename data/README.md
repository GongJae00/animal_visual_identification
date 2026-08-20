# Data

In: local archives, source locks, public-corpus receipts.

Out: layout adapters (`UnifiedCanidSample`), acquisition status, crop export,
and public-intake audits. Raw datasets stay outside Git. Disk trees under
`$CANINE_IDENTITY_DATASETS_DIR` are not rearranged; each publisher layout
lives in `data/adapters/<dataset>.py`. Call `data.adapters.load(name, root)`.
Train/eval roles stay in `evaluation/splits/`. Archive intake stays in
`public_sources/`. This package does not import algorithm stages.

Commands: `uv run python -m data.commands.download --help`
and `uv run python -m data.commands.audit --help`
