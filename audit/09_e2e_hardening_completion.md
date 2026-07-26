# E2E Hardening Completion Receipt

## Scope

This receipt closes the pre-experiment runtime and repository hardening slice.
It does not claim canine ReID accuracy, video E2E completion, production
readiness, or commercial license clearance.

## Reproduced fixes

- Official MiewID 440x440 preprocessing, GeM pooling, pinned revision, strict
  weight loading, and PyTorch/ONNX parity are bound to the canonical wildlife
  ReID channel.
- Deprecated nose aliases remain for compatibility but do not claim a nose
  biometric model.
- Random TinyViT, DNP mask, landmark, and fabricated uncertainty channels fail
  closed instead of emitting identity evidence.
- Trainer inference embedding and metric-learning logits use separate
  `encode()` and `forward_train()` paths.
- `GpuIdentityIndex` uses FAISS GPU when available and a contract-compatible
  `IndexFlatIP` CPU fallback otherwise.
- Protected ONNX workers validate a fixed environment before package loading,
  bind the bootstrap command in receipts, and snapshot all recursive Python
  sources required by embedding production.
- The runtime-library identity cap is 256; the validated CPU discovery
  manifest contained 151 executable identities, each path/size/SHA256 bound.
- Development dependencies and the complete resolved graph are tracked in
  `pyproject.toml` and `uv.lock`.
- Dataset, model, checkpoint, result, artifact, and private research paths are
  excluded from Git.

## Validation

- Targeted model/trainer/GPU/geometric: 63 tests, 58 passed, 5 skipped, 0 failed.
- Targeted ONNX fresh workers: CPU CLI, benchmark, discovery/strict rerun,
  batch worker, and embedding worker passed.
- Full suite: 658 tests, 642 passed, 16 skipped, 0 failures, 0 errors.
- `research-implementation-check --full .`: 0 failures, 0 warnings.
- `research-stage-feedback implementation .`: completed.
- `research-git-checkpoint .`: implementation and ignore gates clean; the
  remaining blocker is the public origin policy, overridden only by the PI's
  explicit request to commit and push this public-repository slice.

## Explicitly unresolved

- `CVI-P0-007`: SuperAnimal export can produce a newly initialized replacement.
- `CVI-P0-011`: the public API has no canonical video decode/detect/track/ReID
  execution path.
- `CVI-P0-009`: calibrator persistence uses pickle.
- `CVI-P0-012`: MiewID weight license and commercial terms are unverified.
- `CVI-P0-013`: three FAISS implementations exist without a final canonical
  identity lifecycle/index decision.
- `CVI-EVAL-013`: small negative-trial counts can yield a threshold sentinel;
  unsupported FAR claims must remain blocked.
- `CVI-EVAL-014`: OSCR is deferred, not implemented.
- No frozen canine appearance baseline, fine-tuning result, channel ablation,
  cross-session/camera/dataset result, video metric, Raspberry Pi benchmark,
  or production service evidence exists yet.

## Next single action

Remove or fail-close `CVI-P0-007` by deleting the newly initialized
SuperAnimal export path and accepting only a pinned, architecture-matched,
hash-verified artifact with an explicit license state and regression test.
