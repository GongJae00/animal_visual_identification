# Runtime Library Provenance

Version strings do not prove which executable binaries were used. CVI therefore
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

Fresh ONNX workers now embed and revalidate the full manifest. The workstation
uses separate `.venv-cpu` and `.venv-cuda` lanes and removes inherited CUDA
loader variables. A synthetic dynamic-batch MatMul smoke was run with two fresh
workers per lane. The strict reruns passed with 29 executable identities and
141,316,821 hashed bytes on CPU, and 43 identities and 1,747,565,589 bytes on
CUDA. The CUDA set used CUDA 13.3 libraries plus the WSL driver projection and
contained no CUDA 12.8 path. Receipts and host-specific strict policies are at:

```text
/mnt/r/research-data/experiments/canine_video_identity/
  runtime-policy-v1-20260722/
```

This admits only that synthetic workload, current host binaries, and current
uncommitted code label. It does not admit a future detector/recognizer graph:
operators such as convolution can lazily load additional cuDNN DSOs and require
a new discovery, review, and strict rerun. It is not camera adequacy, biometric
accuracy, throughput, or deployment evidence.

WSL reports a different device minor for some `/usr/lib/wsl/lib` mappings than
`stat()`. The only exception is an explicit hashed policy flag restricted to a
Microsoft WSL kernel, `/usr/lib/wsl/lib`, device major zero, and identical inode;
all other device/inode mismatches still fail.

Batch-invariance and production embedding receipts do not yet bind this runtime
manifest. Result anchoring and those downstream bindings remain open gates.
