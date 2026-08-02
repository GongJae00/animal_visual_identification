# Evaluation Changes

- Evaluation may consume algorithm packages; algorithm packages must not import evaluation.
- Keep fitting, threshold selection, and policy selection separate from evaluation identities.
- Preserve identity/session/source disjointness, pairing rules, deterministic ordering, metric definitions, and uncertainty semantics.
- Label synthetic and same-track diagnostics accurately. Do not turn them into biometric, cross-session, open-set, or deployment claims.
- Version and content-bind reports, policies, caches, and receipts; reject incompatible inputs rather than silently adapting them.
