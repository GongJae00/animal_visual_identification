# Model Contract Hardening Receipt

## Scope

This slice addresses only reproducible pre-experiment correctness defects.
It does not claim canine ReID accuracy or complete video/service deployment.

## Source Contract

- Upstream: `conservationxlabs/miewid-msv3`
- Revision: `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`
- Weight SHA256:
  `adff92b39678f37eb74861c6399a741639a8907ec2382738e903d6120727b348`
- Official role: 64-species wildlife ReID feature extractor
- Official input: RGB 440x440, ImageNet mean/std
- Official backbone/pooling: `efficientnetv2_rw_m` with learnable GeM(p=3)
- Output: 2152-dimensional BN feature; CVI normalizes it for cosine search
- Code license: UNVERIFIED
- Weight license: UNVERIFIED
- Commercial use/redistribution: UNVERIFIED

## Findings

### CVI-P0-001

- Severity: P0
- Status: FIXED
- Evidence Status: REPRODUCED
- File/Symbol: `src/cvi/evidence/miewid.py::MiewIDReIDExtractor`
- Observed Evidence: legacy runtime defaulted to 160 while the ONNX contract is
  440; legacy preprocessing also omitted ImageNet normalization.
- Reproduction Command: inspect ONNX input and instantiate the deprecated alias
  with `input_size=160`.
- Expected Behavior: reject any input/export contract other than
  `[batch,3,440,440]`.
- Actual Behavior Before Fix: 160 resize and unnormalized RGB.
- Impact: invalid embeddings and invalid channel conclusions.
- Root Cause: hand-written preprocessing was not bound to the upstream card.
- Minimal Fix: enforce tensor shape and official preprocessing.
- Long-Term Fix: PyTorch/ONNX parity receipt on fixed real images.
- Regression Test: `test_miewid_enforces_official_preprocessing_and_dimension`.
- Remaining Uncertainty: regenerated canonical ONNX parity is not yet measured.

### CVI-P0-002 / CVI-P1-002

- Severity: P0
- Status: FIXED
- Evidence Status: REPRODUCED
- File/Symbol: `tools/download_models.py::_download_miewid_msv3`
- Observed Evidence: exporter retained timm AvgPool and loaded backbone with
  `strict=False`; upstream replaces global pooling with GeM.
- Expected Behavior: pinned revision/SHA, GeM, strict state load, manifest.
- Actual Behavior Before Fix: unpinned download, AvgPool, mismatch suppression.
- Impact: architecture and output parity were unsupported.
- Root Cause: approximate reconstruction of upstream custom code.
- Minimal Fix: pinned source, SHA check, GeM, strict load, separate canonical
  artifact, manifest.
- Long-Term Fix: official PyTorch versus ONNX cosine/max-error parity gate.
- Regression Test: export parity remains required before baseline use.
- Remaining Uncertainty: export was not regenerated in this receipt.

### CVI-P0-003

- Severity: P0
- Status: FIXED
- Evidence Status: REPRODUCED
- File/Symbol: `src/cvi/evidence/miewid.py::MiewIDReIDExtractor`
- Observed Evidence: upstream describes wildlife ReID across fins, flukes,
  flanks, and faces; it does not describe nose biometrics.
- Expected Behavior: `wildlife_reid` role with a deprecated compatibility alias.
- Actual Behavior Before Fix: public class, path, docs, and API implied nose.
- Impact: scientifically invalid channel ablation interpretation.
- Root Cause: role conflation.
- Minimal Fix: canonical rename plus deprecated alias.
- Long-Term Fix: compare as an appearance baseline on frozen canine protocols.
- Regression Test: deprecated alias rejects legacy 160 input.
- Remaining Uncertainty: canine domain-shift performance is unmeasured.

### CVI-P0-004 / CVI-P0-005 / CVI-P0-006 / CVI-P0-010

- Severity: P0
- Status: FIXED
- Evidence Status: REPRODUCED
- File/Symbol: `cvi.backbones.get_backbone`, `DNPMask`,
  `LandmarkEvidencer`, `MultiEvidencePipeline.extract_with_uncertainty`
- Observed Evidence: untrained CNN/UNet/GNN paths and constant uncertainty
  entered runtime as evidence.
- Expected Behavior: untrained channels never produce identity evidence.
- Actual Behavior Before Fix: random or fabricated values could be fused.
- Impact: gallery corruption and unsupported open-set claims.
- Root Cause: research placeholders were registered as production channels.
- Minimal Fix: remove registry entry, fail closed, no-op mask, omit unavailable
  uncertainty.
- Long-Term Fix: artifact manifests, strict loaders, channel calibration and
  removal ablations.
- Regression Test: `tests/test_evidence_model_contracts.py`.
- Remaining Uncertainty: SuperAnimal export remains OPEN and unusable.

### CVI-P0-015

- Severity: P0
- Status: FIXED
- Evidence Status: REPRODUCED
- File/Symbol: `src/cvi/trainer.py::ArcFaceModel.forward_train`
- Observed Evidence: validation set `eval()` then applied cross entropy to the
  embedding returned by `forward`.
- Expected Behavior: training/validation loss uses class logits; retrieval uses
  normalized embeddings.
- Actual Behavior Before Fix: checkpoint selection used embedding-as-logits.
- Impact: training model selection was invalid.
- Root Cause: mode-dependent overloaded `forward`.
- Minimal Fix: explicit `encode` and `forward_train`, used in both loops.
- Long-Term Fix: retrieval validation and full optimizer/scheduler/scaler
  lineage checkpoint contract.
- Regression Test: `test_forward_train_returns_logits_in_eval_mode`.
- Remaining Uncertainty: no real fine-tuning run is authorized.

## Deferred

- `CVI-EVAL-014` OSCR: DEFERRED.
- SuperAnimal/landmark trained export: OPEN.
- Video tracking and causal identity events: OPEN.
- Multi-template lifecycle and migrations: OPEN.
- Raspberry Pi profiling/quantization: OPEN.
- Backend/frontend production services: OPEN.
