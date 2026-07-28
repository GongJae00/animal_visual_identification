# Packed Embedding Cache Candidate

## Status

This is a storage-only optimization candidate. It is not a promoted production
format and carries no biometric, throughput, memory, or energy claim. The
protected reference remains one header-free little-endian float32 file per
unique embedding.

The first candidate is an offline deterministic repack. It does not change
model execution, batch composition, precision, normalization, scoring,
ranking, thresholds, or decisions. Native packed production is a later
candidate so storage and inference effects remain separated.

## Candidate objective

For `U` unique vectors of dimension `D`, the reference logical payload is

\[
B_{reference}=4UD.
\]

The candidate concatenates the exact reference payloads into one file:

```text
vectors.f32le.pack
```

The externally bound canonical JSON manifest is the authoritative index. No
second binary index is introduced before JSON parsing or memory is a measured
bottleneck. A duplicate index would add another parser, endianness rules, a
second mutation target, and a possible disagreement between truth sources.

For `D=512`, one vector is 2,048 bytes. With 4,096-byte allocation units,
100,000 individual files can require at least 409.6 MB of allocated payload
space before directory and metadata cost, while a contiguous payload is 204.8
MB. This is an analytical lower-bound example only. ext4 and WSL DrvFS require
separate measurement.

## Canonical layout

Packed entries are strictly increasing by lowercase hexadecimal `cache_key`.
For ordinal `i`,

\[
stride=4D,\qquad offset_i=i\,stride,\qquad size_i=stride.
\]

Each entry binds `cache_key`, `content_sha256`, `byte_offset`, and `byte_size`.
The storage descriptor binds the fixed path, whole-pack SHA-256, total byte
size, vector count, stride, and `cache_key_lexicographic` ordering.

Artifact-to-cache-key bindings and all model, preprocessing, dependency,
precision, normalization, inventory, code, and production provenance remain
unchanged. A parser derives every expected offset from the ordinal and stride;
it does not trust an arbitrary offset. Gaps, overlap, reordering, overflow,
duplicate keys, padding, and trailing bytes are rejected structurally.

## Repack boundary

The future repacker accepts only a completed protected embedding-production
outer bundle plus independently archived outer-receipt and completed-attempt
head hashes. It must:

1. validate the outer execution and publication receipt;
2. fully reverify the closed reference cache;
3. copy raw source bytes without float decode or re-encoding;
4. hash each source payload, destination slice, and complete pack;
5. verify every slice and L2 norm;
6. reverify the source cache after conversion;
7. publish a complete directory without replacement; and
8. emit a separate repack receipt and completed attempt head.

A repack receipt is not an `EmbeddingProductionReceipt`. It binds source outer
anchors, source manifest and verification, repack policy and code, logical
digest equality, target manifest and verification, publication strategy, and
attempt chain. A partial or orphan directory without external anchors is not
admissible.

Linux uses `renameat2(RENAME_NOREPLACE)` where supported. WSL DrvFS uses an
atomically reserved empty directory followed by same-filesystem rename. The
same-UID hostile-process limitation documented for embedding production still
applies.

## Reader boundary

Protected consumers must converge on one bounded vector-store reader instead
of constructing paths directly. The packed reader must:

- require a closed directory containing exactly `vectors.f32le.pack`;
- use directory-relative `O_NOFOLLOW` opens and require a regular file;
- retain one file descriptor for verification and reads;
- compare device, inode, size, modification time, and change time;
- use exact-length `pread` loops for bounded slices;
- reject short reads, undeclared bytes, non-finite values, hash mismatch, or
  norm failure; and
- perform a final full verification before admitting downstream output.

The per-file layout may remain only as an explicit protected reference adapter.
Silent fallback from packed to legacy format is forbidden.

## Hard equivalence gate

For every cache key `k`, the first gate requires exact equality:

\[
bytes_{packed}(k)=bytes_{reference}(k).
\]

Vector SHA-256, float score bits, Top-K and full rank order, Top-1/Top-2
margin, and threshold decision payload must all be identical. Any difference is
`REJECT`; a storage saving cannot compensate for semantic drift.

## Resource comparison

Only after hard equivalence passes are these paired metrics compared:

```text
logical and allocated bytes
regular file count
create plus fsync time
full verification time
sequential lookup time
seeded random lookup time
peak RSS
process read bytes and major faults when available
```

ext4 and the target DrvFS filesystem receive separate decisions. The promotion
workload uses the frozen real cache's `U` and `D`. The default protected
comparison uses 12 paired repetitions per filesystem, randomized AB/BA order,
a two-hour wall-time cap per filesystem, and a precomputed write cap. Hitting a
cap yields `INCONCLUSIVE`.

Warm measurements are reported as warm. `posix_fadvise(DONTNEED)` is
`EVICTION_REQUESTED_UNVERIFIED`; a fresh process or pathname is not called cold
without corroborating physical-I/O evidence.

Promotion requires all exact-equivalence checks, non-inferiority on protected
runtime and memory metrics, and a confidence interval excluding zero for at
least one predeclared resource improvement under
`docs/OPTIMIZATION_CONTRACT.md`.

## Fine-grained gates

1. **Format:** schema, overflow, ordering, exact-slice, mutation, and malformed
   pack unit tests.
2. **Offline repack:** protected source anchors, double source verification,
   fail-closed staging, atomic publication, and receipt tests.
3. **Synthetic smoke:** small deterministic vectors on ext4 and DrvFS; no
   promotion claim.
4. **Protected benchmark:** frozen real cache, paired precommitment, exact
   score/rank equivalence, and resource intervals.
5. **Reader integration:** scorer, numerical admission, and score-drift paths
   migrated in narrow slices with downgrade rejection.
6. **Native writer:** only after repack promotion; batch and inference behavior
   remain separately protected.

Actual RGB/IR camera adequacy and canine identity accuracy remain blocked on G0
acquisition evidence and are not implied by this optimization.
