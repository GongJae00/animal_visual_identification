# Gallery

In: enrollment vectors, identity policy, schema-versioned directory.

Out: immutable gallery generations with GalleryKey / GalleryValue rows.

`store/` is the on-disk gallery. `migration/` copies a v3 gallery into a new
v4 directory. Preserve gallery bytes; do not rewrite a live generation in place.

Optimization catalog (no measured values): `evaluation.optimization_surfaces.gallery`.

Commands: `uv run python -m gallery.commands.migrate --help`
