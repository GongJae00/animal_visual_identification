# Runtime Library Provenance

Version strings do not prove which executable binaries were used. The project therefore
tracks file-backed executable mappings from `/proc/self/maps` at dependency,
session-ready, first-output, and final-output boundaries.

`RuntimeLibraryTracker` groups mappings by device/inode, rejects anonymous,
deleted, special, or multiply-aliased executable identities, opens new paths
with `O_NOFOLLOW`, and retains file descriptors. Hashing occurs only after timed
work so cold-start measurements are not warmed by provenance reads. Each file
is streamed once with bounded memory, checked against its first observation and
final path identity, and followed by a stable final maps-set reread.

Strict admission requires an exact frozen path/size/SHA-256 set. Unknown loaded
DSOs therefore fail rather than being ignored. A discovery-only policy may
produce a candidate manifest for review, but can never authorize that run.
The frozen policy binds the discovery binary-set hash, so a hand-edited expected
set cannot silently claim the original discovery lineage.

Fresh ONNX workers embed and revalidate the full manifest. Use separate CPU and
CUDA dependency lanes and remove inherited CUDA loader variables. Store
host-specific discovery receipts and strict policies outside the repository,
with locations configured explicitly:

```bash
export CANINE_IDENTITY_CPU_PYTHON=/path/to/cpu-environment/bin/python
export CANINE_IDENTITY_CUDA_PYTHON=/path/to/cuda-environment/bin/python
export CANINE_IDENTITY_RUNTIME_POLICY_DIR=/path/to/runtime-policies
```

A strict run admits only its workload, host binaries, and recorded code
revision. It does not admit a future detector or recognizer graph: operators
such as convolution can lazily load additional DSOs and require a new
discovery, review, and strict rerun. It is not camera adequacy, biometric
accuracy, throughput, or deployment evidence.

WSL reports a different device minor for some `/usr/lib/wsl/lib` mappings than
`stat()`. The only exception is an explicit hashed policy flag restricted to a
Microsoft WSL kernel, `/usr/lib/wsl/lib`, device major zero, and identical inode;
all other device/inode mismatches still fail.

Batch-invariance and production embedding contracts can bind an admitted
runtime-manifest digest. Each run must still validate the exact schema it uses;
the existence of that binding machinery is not result or deployment admission.
