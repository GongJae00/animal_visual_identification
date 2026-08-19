# Public Runtime Changes

- Export only `IdentityEngine` and `Match` from `runtime`.
- Accept caller-provided image crops; do not connect video decoding, localization, tracking, temporal aggregation, open-set rejection, or serving behavior here.
- Preserve canonical UUIDv5 identity rules, required-evidence fail-closed behavior, explicit optional evidence, exact scoring, deterministic ordering, and gallery byte compatibility.
- Keep imports lightweight and CPU portable. Do not import learning, evaluation, operations, experiments, or workflows.
- Public API, gallery, config, and failure behavior changes require contract and wheel-installed tests.
