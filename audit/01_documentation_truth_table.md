# Documentation vs. Implementation Truth Table

> Historical Phase 0 snapshot. Entries superseded by fixes are retained as
> evidence; use `audit/STATUS.md` for current status.

## Legend

- `TRUE` — accurate
- `PARTIALLY_TRUE` — partially correct
- `PLANNED_NOT_IMPLEMENTED` — described as design but not built
- `MISLEADING` — gives wrong impression
- `FALSE` — demonstrably wrong
- `UNVERIFIED` — not tested

---

## README.md

| Claim | Status | Evidence |
|-------|--------|----------|
| "코 프린트(비문) + 얼굴 랜드마크 + 외형 특징을 조합" | `MISLEADING` | MiewID is a general wildlife re-ID model, not a nose-print specialist. Landmark channel uses random CNN+GNN. Only appearance (DINOv2) produces meaningful embeddings. |
| "입력 영상 → [개 검출] → [3채널 특징 추출] → [Fusion] → [FAISS 검색]" | `PARTIALLY_TRUE` | Pipeline exists as code but detector is not integrated into CVI API. Only image input supported. |
| "비문 2152d" | `MISLEADING` | MiewID output is 2152-d but it's a general wildlife re-ID model, not a nose-print model. Also runtime uses wrong 160x160 input (should be 440x440). |
| "랜드마크 256d" | `PARTIALLY_TRUE` | LandmarkGraphEmbedder outputs 256-d but both the CNN heatmap and GNN have random weights. Output is meaningless. |
| "외형 384d" | `PARTIALLY_TRUE` | DINOv2 outputs 384-d correctly. But Dinov2WithUncertainty returns hardcoded uncertainty (epistemic=0.05, aleatoric=0.1). |
| "YOLO" for detection | `PLANNED_NOT_IMPLEMENTED` | DogDetector class exists in detection.py but YOLO weights are never downloaded. DogDetector is not imported or wired into any API path. |
| `uv sync --extra cuda --extra training` | `TRUE` | Works correctly. Verified. |
| `uv sync --extra cpu` | `TRUE` | Works correctly. |
| `DogDetector.detect("dog_video.mp4")` | `FALSE` | DogDetector.detect() accepts a PIL Image, not a video path. Detection is not integrated into the CVI API. |
| `CVIDeploymentCUDA` usage example | `PARTIALLY_TRUE` | CVIDeploymentCUDA exists and enroll/search work. But evidence channels are misconfigured (landmark = random, no real nose). |
| "단위 테스트 (77개)" | `FALSE` | Actual count: 605 test methods across 66 files. |
| "성능 지표: AUC, EER, TAR@FAR, d'" | `FALSE` | compute_metrics() returns these keys correctly, but rank1 is always 0 due to missing key (P0-008). Evaluate_multichannel reports broken results. |
| "라이선스: MIT / Apache 2.0" | `UNVERIFIED` | Repo has no LICENSE file. MiewID-msv3 has no license. Other deps have various licenses. |

## AGENTS.md

| Claim | Status | Evidence |
|-------|--------|----------|
| "75개 평면 모듈" | `FALSE` | 115 Python files in src/cvi/. |
| "564개 테스트" | `FALSE` | 605 test methods found. |
| "64개 CLI 도구" | `TRUE` | 65 tools found (close enough). |
| "4종 데이터셋" | `TRUE` | DogFaceNet, MPDD, SiBeTan, YT-BB-dog all present. |
| `from cvi import CVI, Match` | `TRUE` | Works. |
| `cvi.enroll(image, "뽀삐", breed="비글")` | `PARTIALLY_TRUE` | Works but breed parameter is only soft metadata — no breed classifier is actually used for search filtering. |
| `from cvi.training import TrainConfig` | `FALSE` | Module path should be `from cvi.train.config import TrainConfig`. |

## ARCHITECTURE.md

| Claim | Status | Evidence |
|-------|--------|----------|
| "Biometric hierarchy: nose print > skeletal landmarks > appearance" | `FALSE` | Nose channel is non-functional (wrong input, random mask, random backbone). Landmark is random. Only appearance works. |
| "Nose Print Pipeline: YOLO-Nose → SuperRes → UNet Mask → TinyViT+MagFace → 512-d" | `FALSE` | YOLO-Nose detector is a dummy (returns center crop when no model). No super-resolution. UNet mask is random. TinyViT is a 3-layer random CNN. MagFaceNoseHead has random weights. The claimed 512-d vs actual 2152-d from MiewID. |
| "Skeletal Landmark Graph: HRNet 17-pt → DGCNN → 256-d" | `FALSE` | HRNetHeatmap is a 5-layer random CNN (not HRNet-W32). LandmarkGraphEmbedder (EdgeConv-based) has random weights. SuperAnimal ONNX is a 9 KB dummy. |
| "Appearance: DINOV2/ConvNeXt ArcFace → 384-d → evidential head → uncertainty" | `PLANNED_NOT_IMPLEMENTED` | DINOv2 backbone works (384-d). But ArcFace head is never trained. Evidential head is untrained. Uncertainty is hardcoded to 0.05/0.1. |
| "Learned Fusion: uncertainty-gated weighted sum" | `PLANNED_NOT_IMPLEMENTED` | LearnedWeightFuser exists but uncertainty is fake (hardcoded). No training loop for fusion weights. |
| "Open-Set: evidential threshold" | `PLANNED_NOT_IMPLEMENTED` | EvidentialOpenSet exists but uses fake uncertainty and uncalibrated thresholds. |
| "TemporalAggregator: learned weighted median" | `PLANNED_NOT_IMPLEMENTED` | TemporalAggregator exists but is never connected to the CVI API or any pipeline. |
| "PetFace (257K individuals) fine-tune ConvNeXt ArcFace" | `PLANNED_NOT_IMPLEMENTED` | PetFace data is not downloaded. No fine-tuning has been performed. |
| "YOLOv8-nose detector (from Nose-to-ID, 2025)" | `PLANNED_NOT_IMPLEMENTED` | No YOLOv8 nose detector exists. YoloNoseDetector returns hardcoded center crop when no model. |
| "DNND (196K samples)" | `UNVERIFIED` | No reference to DNND dataset download or existence. |
| "DogFLW 17-point → HRNet-W32 heatmap" | `PLANNED_NOT_IMPLEMENTED` | DogFLW defined but never downloaded (HF repo 404/not created). HRNet-W32 not implemented. |

## Evidence Extractor (Flat Module)

| Claim | Status | Evidence |
|-------|--------|----------|
| `DogFaceNetExtractor` — visual channel | `PARTIALLY_TRUE` | Class exists with input_size=224, output_dim=384. ONNX wrapper. But no DogFaceNet ONNX exists in checkpoint directory. |
| `ConvNeXtExtractor` — texture channel | `PARTIALLY_TRUE` | Class exists. ConvNeXt ONNX exists (173 KB + 4.8 MB data). But no trained ArcFace head — uses raw backbone features. |
| `SuperAnimalExtractor` — structural | `FALSE` | SuperAnimal ONNX is a 9 KB dummy (not real HRNet). Class calls _keypoints_to_embedding on raw ONNX output which is random. |
| `PetReIDExtractor` — nose print | `FALSE` | No PetReID ONNX exists in model paths. Class calls OnnxExtractor which would fail on non-existent path. |

## Config Files

| Claim | Status | Evidence |
|-------|--------|----------|
| "deployment/production.json" referenced in README | `PLANNED_NOT_IMPLEMENTED` | No production.json exists. Only .example.json files. |
| "evidence/multi.json" in README eval command | `PLANNED_NOT_IMPLEMENTED` | No multi.json exists. Only .example.json files. |
| "data/registry/val_pairs.json" in README eval | `PLANNED_NOT_IMPLEMENTED` | No val_pairs.json exists. |

## Summary of False/Misleading Claims

| Document | False | Misleading | Planned | True | Unverified |
|----------|-------|------------|---------|------|------------|
| README.md | 4 | 3 | 0 | 4 | 1 |
| AGENTS.md | 2 | 1 | 0 | 2 | 0 |
| ARCHITECTURE.md | 4 | 0 | 6 | 0 | 1 |
| EvidenceExtractor | 2 | 3 | 0 | 0 | 0 |
| Configs | 0 | 0 | 3 | 0 | 0 |
