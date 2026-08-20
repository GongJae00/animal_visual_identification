# Enrollment

In: canonical UUIDv5 identities, channel vectors, optional breed/metadata.

Out: GalleryKey / GalleryValue rows admitted by the registry policy.

`registry/` maps dataset labels to registered UUIDv5 values. `binding/` is the
fail-closed registered-only gallery policy. `write/` enrolls extracted vectors
into a gallery store.

Commands: `uv run python -m enrollment.commands.enroll --help` (label augment).
Registry build/bind and split check: `evaluation.commands.evaluate registry-build|registry-bind|split-check`.
