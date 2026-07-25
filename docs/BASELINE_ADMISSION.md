# Baseline Admission

## Purpose

A current paper, an available repository, and a deployable project are different
things. CVI admits an implementation dependency only after checking:

1. primary technical evidence;
2. source and model-weight license;
3. maintenance and reproducible installation state;
4. export or deployment path;
5. numerical and task-level validation route;
6. compatibility with the CVI data and evidence boundary.

An absent or unclear license is treated as no deployment permission. This is an
engineering admission policy, not legal advice.

## Initial portfolio audit

Audit date: 2026-07-20.

| Candidate | Evidence/code status | Initial disposition |
|---|---|---|
| [D-FINE](https://github.com/Peterande/D-FINE) | Apache-2.0 code; maintained; official ONNX and TensorRT paths | Admit detector N/S variants for local accuracy/latency evaluation. Check checkpoint and pretraining lineage separately. |
| [D-FINE-Seg](https://github.com/ArgoHA/D-FINE-seg) | Apache-2.0 third-party extension with multi-backend claims | Exploratory segmentation candidate only; require independent correctness and export audit. |
| [ByteTrack](https://github.com/FoundationVision/ByteTrack) | MIT code; established online tracking control | Admit as the efficient tracker control, not as the only modern comparator. |
| [TrackTrack](https://openaccess.thecvf.com/content/CVPR2025/html/Shim_Focusing_on_Tracks_for_Online_Multi-Object_Tracking_CVPR_2025_paper.html) | Current primary paper; no official implementation admitted in this audit | Paper-level algorithm comparator; no copied dependency. |
| [GeneralTrack](https://openaccess.thecvf.com/content/CVPR2024/html/Qin_Towards_Generalizable_Multi-Object_Tracking_CVPR_2024_paper.html) | Primary paper; no official implementation admitted in this audit | Generalization comparator and experiment design reference. |
| [OpenAnimals](https://openaccess.thecvf.com/content/ICCV2025/html/Hou_OpenAnimals_Revisiting_Person_Re-Identification_for_Animals_Towards_Better_Generalization_ICCV_2025_paper.html) | Primary paper and public repository; repository has no explicit license in the audited state | Research evidence only. Do not vendor, copy, or use its weights in deployment. |
| [CARE](https://openaccess.thecvf.com/content/WACV2026/html/Wu_Overcoming_Fine-Grained_Visual_Challenges_in_Animal_Re-Identification_via_Semantic_Feature_WACV_2026_paper.html) | Current primary animal Re-ID paper | Method and ablation reference; implementation dependency not admitted. |
| [WildlifeDatasets](https://github.com/WildlifeDatasets/wildlife-datasets) | AGPL-3.0 code | Research tooling boundary only unless a later legal/architecture review explicitly admits its use. |
| [FAISS](https://github.com/facebookresearch/faiss) | MIT code; maintained | Candidate only after exact search violates a measured scale budget. |
| [ONNX Runtime](https://github.com/microsoft/onnxruntime) | MIT code; maintained | Portable CPU/GPU backend candidate after export equivalence tests exist. |
| [TensorRT](https://docs.nvidia.com/deeplearning/tensorrt/latest/) | NVIDIA deployment SDK | Local RTX fast path candidate; never the only runnable path. |

## Frozen initial comparison structure

- Detection: oracle boxes, then D-FINE N/S at frozen resolutions.
- Segmentation: no segmentation, a separately licensed baseline, then
  D-FINE-Seg only if the exploratory audit passes.
- Tracking: IoU/motion control, ByteTrack control, then bounded additions whose
  implementation and license are independently admissible.
- Identity: a clean implementation built from deployment-compatible primitives;
  paper methods may inform ablations, but unlicensed repositories and restricted
  weights do not enter the deployment lineage.
- Retrieval: exact cosine search first; FAISS is activated only by a measured
  gallery-scale violation.
- Runtime: portable eager/ONNX reference first, guarded TensorRT fast path after
  output and end-to-end non-inferiority tests.

No candidate in this document is a claimed winner. Admission only means it is
eligible for a controlled local comparison.
