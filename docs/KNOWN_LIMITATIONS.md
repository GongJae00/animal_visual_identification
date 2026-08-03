# Known Limitations

This document records current public behavior. It does not contain private
artifact hashes, internal receipts, or claims of final biometric performance.

## Recognition Boundary

- `IdentityEngine` accepts already-cropped still images. It does not decode video, detect
  dogs, associate tracks, select frames, or aggregate a track.
- Retrieval is closed-set candidate ranking. A returned top match is not an
  authenticated identity decision, and no-match or unknown-dog rejection is
  disabled in the canonical API.
- No canine-trained checkpoint or benchmark result is bundled. The README
  provides a receipt-bound local DINOv2 configuration template, but it is not
  runnable until the user supplies admitted model and preprocessing artifacts.
- Tests cover software contracts and synthetic behavior. They do not establish
  accuracy, fairness, robustness, longitudinal stability, or production safety.

## Evidence And Models

- `IdentityEngine` rejects the legacy `dinov2` and `appearance` channel types. The public
  runtime does not provide an opt-in to execute the unpinned Torch Hub loader.
- Optional channel implementations require exact local model, preprocessing,
  manifest, and in some cases parity artifacts. Those artifacts are not bundled.
- MiewID-msv3 is a general wildlife re-identification comparator, not a canine
  nose-print model. Its downloader records unresolved code and weight license
  status, and the public runtime requires additional manifest/parity material.
- SuperAnimal is disabled by the downloader and canonical runtime because the
  available path does not satisfy the required license and verified export
  contract.
- DogFLW download support does not create an admitted identity-evidence channel.
- Nose-print, landmark-graph, evidential-uncertainty, and multi-channel identity
  value have not been established by a released leakage-controlled ablation.
- The N4 residual metric adapter is an offline research candidate over frozen N3
  embeddings. Its positive publisher-panel result is same-track, and a frozen
  SiBeTan substitution did not improve Rank-1. It is not an admitted runtime
  channel or evidence of physical nose-ridge topology.

## Calibration And Evaluation

- `IdentityEngine` rejects open-set enablement. Metric and calibration modules elsewhere in
  the repository are research utilities, not an operational threshold.
- No frozen threshold is shipped for known, unknown, or review decisions.
- No released result supports a biometric performance, subgroup, domain-shift,
  low-light, motion-blur, seasonal, or long-term stability claim.
- Dataset identity leakage, near duplicates, sequence correlation, and crop
  context remain protocol risks that each evaluation must control explicitly.

## Scale And Operations

- Candidate scoring is exact over stored templates and has not been validated
  as a large-gallery, low-latency service.
- A gallery permits one POSIX-locked writer. There is no supported distributed
  writer, transaction service, replication protocol, backup workflow, or online
  migration service.
- Operations facades intentionally remain disconnected from the public runtime.
  ONNX CPU/CUDA components are measurement and artifact-validation tools until
  they are connected to the canonical gallery and decision contract.
- Linux with POSIX filesystem semantics is the supported environment. Native
  Windows is incompatible with the current `fcntl` gallery lock. Other POSIX
  platforms and network filesystems are unvalidated.

## Data, Privacy, And Security

- Data and model downloader coverage is partial. Several dataset handlers print
  manual acquisition instructions rather than downloading files.
- Third-party terms may restrict access, redistribution, research, or commercial
  use. The repository license does not override those terms.
- The project provides no service authentication, authorization, encryption, consent,
  retention, deletion, or audit-log product. Identity mappings and animal/owner
  imagery require an external privacy and security design.
- Model and data files are untrusted inputs. Download and conversion tools do
  not replace source review, sandboxing, or dependency governance.

See [Security Policy](../SECURITY.md),
[Data and Models](DATA_AND_MODELS.md), and [Roadmap](ROADMAP.md).
