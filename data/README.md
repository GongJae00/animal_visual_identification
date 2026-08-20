# Data

In: local archives, source locks, public-corpus receipts.

Out: adapters, acquisition status, crop export, and public-intake audits.
Public-corpus intake lives under `data/public_sources/`. Raw datasets stay
outside Git. This package does not import algorithm stages.

Commands: `uv run python -m data.commands.download --help`
and `uv run python -m data.commands.audit --help`
