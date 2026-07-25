# Evidence Pipeline Architecture

## Overview

Replace the single-ONNX shared-backbone `FeatureExtractor` with a pluggable
`EvidenceExtractor` registry so each evidence channel uses its own
independently optimized open-source backbone.

## Channels

| Channel     | Model                  | Dims | Format  | Source |
|-------------|------------------------|------|---------|--------|
| Visual      | DogFaceNet (ResNet)    | 384  | ONNX    | [GitHub: GuillaumeMougeot/DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet) / [SachaDee ONNX](https://huggingface.co/SachaDee/DogFaceRecognition) |
| Texture     | ConvNeXt-Base (timm)   | 768  | ONNX    | [timm](https://github.com/huggingface/pytorch-image-models) / finetune on Stanford Dogs / YT-BB-dog |
| Structural  | SuperAnimal-Quadruped  | 256  | PyTorch | [DeepLabCut Model Zoo](https://deeplabcut.github.io/DeepLabCut/docs/ModelZoo.html) (HRNet-w32, 17 keypoints) |
| Nose Print  | Pet-ReID-IMAG / ResNeSt-101 | 2048 | ONNX  | [GitHub: muzishen/Pet-ReID-IMAG](https://github.com/muzishen/Pet-ReID-IMAG) (CVPR2022 Workshop top3) |

## Detection (pre-pipeline)

| Task          | Model   | Output        | Format |
|---------------|---------|---------------|--------|
| Dog body+face | YOLO11  | bounding box  | ONNX   |
| Face crop     | YOLO11  | 224×224 crop  | —      |

## Registry interface

```python
class EvidenceExtractor(ABC):
    @abstractmethod
    def output_dim(self) -> int: ...
    @abstractmethod
    def extract(self, image: PIL.Image) -> np.ndarray: ...
    @abstractmethod
    def extract_batch(self, images: list[PIL.Image]) -> np.ndarray: ...
```

Registered by name:

```python
extractors = EvidenceExtractorRegistry()
extractors.register("visual", DogFaceNetExtractor(model_path))
extractors.register("texture", ConvNeXtExtractor(model_path))
extractors.register("structural", SuperAnimalExtractor(model_path))
extractors.register("nose", PetReIDExtractor(model_path))
```

## Fused embedding

Channels that are ready (ONNX deployed) contribute to the fused 640-d
embedding. Channels still training are skipped at inference with a
graceful fallback.

```
fused = L2_NORM(concat([e_visual, e_texture, e_structural]))
```

Nose print (2048-d) is searched in a separate dedicated index with its
own cosine similarity, then fused at the score level (not embedding
level) to avoid dimensionality imbalance.

## Search flow

```
Frame → YOLO11 detection → face crop
         │
         ├→ DogFaceNet ──→ 384-d visual embedding ──┐
         ├→ ConvNeXt   ──→ 768-d texture embedding ──┤→ concat → 640-d → FAISS
         └→ SuperAnimal ─→ 17 kpts → 256-d struct ──┘
         └→ (optional) Pet-ReID ──→ 2048-d nose → separate FAISS index
                                                          ↓
                                              evidence breakdown + fusion
```

## Finetuning strategy

1. **Visual** — Start from SachaDee ONNX (already trained on DogFaceNet
   dataset, 96.7% accuracy). If YT-BB-dog performance is insufficient,
   finetune DogFaceNet (ResNet backbone, triplet loss) on our 12K training
   crops.
2. **Texture** — Finetune ConvNeXt-Base from `timm` (ImageNet-1K
   pretrained) on YT-BB-dog training crops using ArcFace loss (reuse
   existing `trainer.py` loop).
3. **Structural** — Use SuperAnimal-Quadruped zero-shot. If landmark
   accuracy is insufficient, finetune HRNet-w32 on DogFLW dataset (3,732
   images, 46 landmarks) via DeepLabCut.
4. **Nose Print** — Pet-ReID-IMAG weights via Google Drive (ResNeSt-101,
   2048-d). Requires nose-crop training data; blocked until we collect or
   license a nose-print dataset.

## Blocked

- GPU training for any channel until `causal_routing_denoiser` releases
  VRAM (~6.9 GiB free of 16 GiB needed for ConvNeXt finetuning).
- Nose-print channel requires a nose-print dataset (CVPR2022 Pet
  Biometric Challenge data is not publicly redistributable).
