# WSL and 4TB SSD Storage Boundary

## Active roots

Source code and private research ledgers remain on the WSL Linux filesystem.
Large CVI data belongs on the Samsung 990 PRO 4TB tier under:

```text
/mnt/r/research-data/canine_video_identity_secure/
  raw/
  downloads/
  datasets/{research-only,deployment-eligible}/
  checkpoints/{research-only,deployment-eligible}/
  experiments/
  artifacts/
  manifests/
  cache/
```

The secure root has Windows inheritance disabled. Its allowed principals are
the current workstation user, `SYSTEM`, and `BUILTIN\Administrators`, each with
full inheritable access. New sensitive CVI material must not be written to the
older, broadly inherited `/mnt/r/research-data/*/canine_video_identity` roots.
Those paths contain only historical infrastructure smokes at present.

## DrvFS semantics

`/mnt/r` is WSL 9P/DrvFS, not a native Linux filesystem. Linux mode bits are
not the confidentiality authority: a file written as mode 0600 is currently
reported as mode 0777. Windows ACLs must therefore be audited with `icacls.exe`
before admitting private camera data, biometric embeddings, protected
manifests, or checkpoints. Code must not interpret the DrvFS mode display as a
private-file guarantee.

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

The recorded DrvFS infrastructure smoke is:

```text
/mnt/r/research-data/canine_video_identity_secure/experiments/smoke/
  publication-jncG0rHk/receipt.json
```

It proves only that directory fsync, the reserved-directory rename fallback,
existing-target preservation, and inherited Windows ACLs worked on this
workstation. It is not camera, recognition, performance, or optimization
evidence.
