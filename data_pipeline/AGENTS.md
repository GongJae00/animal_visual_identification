# Data Pipeline Changes

- Keep raw archives, extracted datasets, private manifests, caches, and generated crops outside Git.
- Preserve source lineage, exact hashes, identity/session metadata, and license/intake state; do not fabricate missing labels or provenance.
- Split at identity and protocol component boundaries. Never introduce random frame splitting as an identity-evaluation shortcut.
- Download and extraction paths must fail safely, reject traversal and ambiguous archives, and never imply dataset admission from acquisition alone.
- Update deterministic manifest, duplicate, crop-export, and failure-path tests with behavior changes.
