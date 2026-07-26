# License and Compliance Audit

> License evidence is component-specific. `UNVERIFIED` is not permission for
> commercial use or redistribution.

## Status
Gate G0 prerequisite. This document records license information for all model and software components. Legal determination is NOT made here — upstream license files and metadata are recorded for legal review.

## Component Rows

### CVI Repository Code
- **Component**: CVI source (src/cvi/, tools/, tests/)
- **Type**: source code
- **Version**: 0.2.0 (`audit/e2e-hardening`; resolve exact revision with Git)
- **License**: None declared (no LICENSE file in repository root)
- **Source**: https://github.com/GongJae00/canine_video_identity
- **Commercial use**: UNVERIFIED — no license file
- **Redistribution**: UNVERIFIED
- **Attribution**: UNVERIFIED
- **Status**: UNVERIFIED

### MiewID Model Weights
- **Component**: conservationxlabs/miewid-msv3 model.safetensors (196 MB)
- **Type**: pretrained weights
- **Version**: `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`
- **License**: UNVERIFIED — no LICENSE file in HF repo, cardData license field absent
- **Source**: https://huggingface.co/conservationxlabs/miewid-msv3
- **Upstream code**: https://github.com/WildlifeDatasets/miewid
- **Commercial use**: UNVERIFIED
- **Redistribution**: UNVERIFIED
- **Status**: UNVERIFIED — requires upstream repo inspection + weight-specific terms

### MiewID Custom Model Code
- **Component**: modeling_miewid.py, configuration_miewid.py, heads.py
- **Type**: custom modeling code (transformers integration)
- **Version**: `4f1d7f2b521149e5fe34bb85f377248ce9971a7d`
- **License**: Apache-2.0 for the pinned upstream code; weight terms remain separate
- **Source**: https://huggingface.co/conservationxlabs/miewid-msv3
- **Commercial use**: VERIFIED for pinned code only; weights remain UNVERIFIED
- **Status**: VERIFIED for pinned upstream code; runtime does not enable `trust_remote_code`

### MiewID Model Card
- **Component**: README.md on HF
- **Type**: model documentation
- **License field**: absent from card metadata
- **Source**: https://huggingface.co/conservationxlabs/miewid-msv3
- **Status**: UNVERIFIED — no license declared in card

### DINOv2
- **Component**: facebookresearch/dinov2 (via torch.hub)
- **Type**: pretrained model
- **Version**: dinov2_vits14
- **License**: Apache 2.0
- **Source**: https://github.com/facebookresearch/dinov2
- **Commercial use**: VERIFIED (Apache 2.0)
- **Status**: VERIFIED

### ConvNeXt
- **Component**: facebook/convnext-base-224 (via transformers)
- **Type**: pretrained model
- **Version**: from HF hub
- **License**: CC-BY-NC 4.0 (non-commercial) — via original ConvNeXt repo
- **Source**: https://github.com/facebookresearch/ConvNeXt
- **Commercial use**: RESTRICTED (CC-BY-NC)
- **Status**: VERIFIED as non-commercial — replacement needed for production

### ONNX Runtime
- **Component**: onnxruntime / onnxruntime-gpu
- **Type**: ML inference runtime
- **Version**: >=1.27 (pyproject.toml)
- **License**: MIT
- **Source**: https://github.com/microsoft/onnxruntime
- **Commercial use**: VERIFIED (MIT)
- **Status**: VERIFIED

### FAISS
- **Component**: faiss-cpu / faiss-gpu
- **Type**: similarity search library
- **Version**: >=1.7.4 / >=1.9.0
- **License**: MIT
- **Source**: https://github.com/facebookresearch/faiss
- **Commercial use**: VERIFIED (MIT)
- **Status**: VERIFIED

### PyTorch
- **Component**: torch (via pytorch-cu128 index)
- **Type**: deep learning framework
- **Version**: >=2.7
- **License**: BSD-style
- **Source**: https://github.com/pytorch/pytorch
- **Commercial use**: VERIFIED
- **Status**: VERIFIED

### timm
- **Component**: huggingface/timm
- **Type**: model zoo / image models
- **License**: Apache 2.0
- **Commercial use**: VERIFIED (Apache 2.0)
- **Status**: VERIFIED

### transformers
- **Component**: huggingface/transformers
- **Type**: model library
- **License**: Apache 2.0
- **Commercial use**: VERIFIED (Apache 2.0)
- **Status**: VERIFIED

### Ultralytics
- **Component**: ultralytics (training dependency)
- **Type**: YOLO training library
- **Version**: >=8.3
- **License**: AGPL-3.0
- **Commercial use**: RESTRICTED (AGPL-3.0 requires source distribution)
- **Status**: VERIFIED as AGPL — training-only dependency, not in inference path

### SuperAnimal
- **Component**: superanimal_quadruped_hrnet_w32.pt
- **Type**: pretrained weights
- **License**: UNVERIFIED — downloaded from HF, need to check
- **Source**: HF hub (repo not tracked)
- **Status**: UNVERIFIED

### scikit-learn / numpy / pillow / opencv
- **Component**: standard Python ML/data libraries
- **Type**: libraries
- **License**: BSD-3-Clause (scikit-learn, numpy, pillow), MIT (opencv)
- **Status**: VERIFIED

## Summary Table

| Component | License | Commercial Use | Status |
|-----------|---------|---------------|--------|
| CVI source | None declared | UNVERIFIED | UNVERIFIED |
| MiewID weights | None declared | UNVERIFIED | UNVERIFIED |
| MiewID code | Apache 2.0 | YES (code only) | VERIFIED |
| DINOv2 | Apache 2.0 | YES | VERIFIED |
| ConvNeXt | CC-BY-NC 4.0 | NO | RESTRICTED |
| ONNX Runtime | MIT | YES | VERIFIED |
| FAISS | MIT | YES | VERIFIED |
| PyTorch | BSD | YES | VERIFIED |
| timm | Apache 2.0 | YES | VERIFIED |
| transformers | Apache 2.0 | YES | VERIFIED |
| Ultralytics | AGPL-3.0 | conditional | VERIFIED |
| SuperAnimal | UNVERIFIED | UNVERIFIED | UNVERIFIED |

## Production Gate Status
**FAIL** — CVI source has no license file, MiewID license is UNVERIFIED, ConvNeXt is non-commercial.
