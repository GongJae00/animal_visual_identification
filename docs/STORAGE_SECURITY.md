# Storage Security Boundary

## Active roots

Keep source code separate from private datasets, embeddings, checkpoints, and
experiment outputs. Configure the data root explicitly for each machine:

```bash
export CVI_DATA_DIR=/path/to/cvi-data
```

The configured root may use this layout:

```text
$CVI_DATA_DIR/
  downloads/
  datasets/<dataset-name>/
  checkpoints/<model-artifact-id>/
  experiments/
  receipts/
  manifests/
  cache/
```

Dataset and checkpoint directory names identify content, not permission or
license status. Keep research/deployment admission, source terms, hashes, and
revisions in the corresponding registry or receipt metadata. This avoids moving
large immutable artifacts when an admission decision changes. Do not infer
deployment eligibility from a filesystem path.

Before use, verify that the root has restrictive ownership and access controls.
Do not place sensitive CVI material in a directory with broad inherited access.

## DrvFS semantics

A Windows drive mounted in WSL under `/mnt/<drive-letter>` uses DrvFS rather
than a native Linux filesystem. Depending on mount configuration, Linux mode
bits may not be the confidentiality authority and a file written as mode 0600
may be displayed with broader permissions. Audit the backing Windows ACLs, for
example with `icacls.exe`, before admitting private camera data, biometric
embeddings, protected manifests, or checkpoints. Code must not interpret the
DrvFS mode display alone as a private-file guarantee.

Protected, independently archived precommitment/final hashes should remain in
the private WSL research-state layer or another independently controlled
store. A hash stored beside the artifact it authenticates is useful for
content addressing but is not an external anchor.

## Atomic publication

Native Linux filesystems use `renameat2(RENAME_NOREPLACE)`. DrvFS rejects that
operation, so the protected cache publisher uses this fallback:

1. atomically reserve the absent target name with `mkdir`;
2. bind the reserved directory identity and require it to remain empty;
3. atomically rename the complete same-filesystem staging directory over that
   cooperative reservation;
4. reverify every published vector and fsync the target and parent;
5. issue the externally anchored outer receipt only after verification.

An existing or populated target is never replaced. A crash before rename may
leave an empty reservation; a crash after rename may leave a complete orphan
cache. Neither is admissible without the externally archived outer receipt and
attempt-ledger head. Recovery must inspect and quarantine such orphans rather
than silently reusing them.

This protects against ordinary concurrent CVI jobs and worktree/data-pipeline
changes. It is not an isolation boundary against a hostile process running as
the same OS user.

Validate directory fsync, the reserved-directory rename fallback,
existing-target preservation, and effective ACLs on every target filesystem.
Such a storage smoke is not camera, recognition, performance, or optimization
evidence.
