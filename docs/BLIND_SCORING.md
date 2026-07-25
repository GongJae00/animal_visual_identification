# Blind Score Receipt

## Boundary

The scorer receives only token-keyed query/reference artifacts and
`cvi.pair_scoring_requests.v1`. It returns one finite similarity score per
opaque pair ID. It never receives registered dog ID, session label, hard
stratum, breed, cage, expected dog ID, threshold, or test outcome.

The receipt binds:

- exact pair-set and scoring-request hashes;
- token-keyed artifact manifest hash;
- model and gallery hashes;
- inference config and dependency-lock hashes;
- code revision, scorer version, precision, and device;
- unique `(pair_id, score)` records.

The schema rejects unknown fields so identity labels cannot be silently carried
through the scorer receipt.

## Sealed join

The evaluator verifies exact equality of request, truth, and score pair-ID
sets. Missing, extra, duplicate, stale, non-finite, wrong-model, and
wrong-gallery scores are rejected. Only then are scores joined to sealed dog
IDs and passed to the frozen-threshold evaluation.

The sealed evaluator must also receive the exact `PairingPolicy` used for pair
construction. Its content hash must match the hash carried by the pair bundle,
and its RGB/IR verification direction must exactly match the frozen threshold
direction. A threshold calibrated for another modality direction therefore
fails closed even if model and gallery hashes happen to match.

Artifact manifest verification must include file-content hashes, not only token
names. `cvi.pair_artifact_manifest.v1` binds the pair set, protected binding
artifact, exact token set, byte sizes, media types, and content hashes. The
score receipt's artifact hash must match it.

The file verifier does reread byte size and SHA-256 with mutation checks. The
artifact directory may contain only regular files named exactly
`artifact-token.png`, `.jpg`, or `.jpeg`; symlinks, subdirectories, missing
files, and extra identity-named files are rejected. Image decode, dimensions,
pixel semantics, and embedded metadata remain a later exporter validation. The
contract does not yet create real crops or prove those semantic properties.
