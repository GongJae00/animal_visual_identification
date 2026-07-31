# CVI Report Visualization Index

이 디렉터리는 보고서 편집을 위한 재현 가능한 시각화 산출물입니다. 사진이 포함된 두 도판은 승인된 CC-BY-4.0 공개 데이터의 저해상도 합성 파생물이며, 나머지 수치는 실제 최종 평가 보고서의 집계값에서 생성됩니다. 개체 라벨, 샘플 식별자, 소유자 정보, 절대 경로, 원시 임베딩, 질의별 결과는 포함하지 않습니다.

## Figure Index

1. **Dataset overview** (`00_dataset/00_dataset_overview.png`)  
   다섯 공개 데이터셋의 실제 대표 이미지와 연구 역할을 비교합니다.  
   *Five actual public samples with train, validation, and localization roles.*
2. **Experimental preprocessing** (`01_preprocessing/01_preprocessing_pipeline.png`)  
   AP-10K 원본에서 검출 상자, 17개 포즈 점, dog/face/weak-nose crop, 유효 마스크까지의 실험적 전처리를 보입니다.  
   *A hash-verified AP-10K source-to-ROI trace.*
3. **Runtime boundary** (`01_preprocessing/01_pipeline_boundary.svg`, `.png`)  
   검출·포즈·ROI는 상류 실험 분기이고, 공개 CVI는 호출자가 제공한 crop에서 시작함을 구분합니다.  
   *Experimental localization is upstream; canonical CVI starts at a caller-provided crop.*
4. **Embedding architecture** (`02_embedding/02_embedding_architecture.svg`, `.png`)  
   Appearance 384D와 Face 384D + 256D regional 경로, 640D L2 출력, 동결된 융합 가중치를 표시합니다.  
   *Exact embedding dimensions and frozen Appearance/Face fusion weights.*
5. **Embedding geometry** (`02_embedding/02_embedding_geometry.svg`, `.png`)  
   명목 차원, entropy effective rank, 방향 중심 크기를 비교하여 regional collapse 신호를 제한적으로 해석합니다.  
   *Nominal versus effective rank and directional centroid metrics.*
6. **Gallery retrieval** (`03_retrieval/03_gallery_retrieval.svg`, `.png`)  
   cosine template score, identity-level max, 안정 정렬, closed-set 후보 출력의 계약을 설명합니다.  
   *Canonical max aggregation and stable closed-set identity ranking.*
7. **OOF calibration and fusion** (`04_calibration_fusion/04_oof_calibration_fusion.svg`, `.png`)  
   identity-disjoint 5-fold isotonic OOF 보정과 최종 평가 전 동결 경계를 나타냅니다.  
   *Five-fold OOF calibration, quality gate, simplex, and pre-final freeze.*
8. **Calibration selection** (`04_calibration_fusion/04_calibration_selection.svg`, `.png`)  
   실제 one-shot A/F/fused 결과와 three-shot 후보 Rank-1, 선택된 max를 보여줍니다.  
   *Actual calibration metrics and the selected max aggregation.*
9. **Final results** (`05_evaluation/05_final_results.svg`, `.png`)  
   one-/three-shot A/F/frozen-fusion Rank-1·MRR와 융합 Rank-1 95% CI를 보고합니다.  
   *Actual final closed-set metrics, cohort/query counts, and fused Rank-1 CIs.*
10. **Failure diagnostics** (`06_diagnostics/06_failure_diagnostics.svg`, `.png`)  
    effective-rank 활용률, 중심 크기, norm 거동, regional collapse 신호와 후속 조치를 정리합니다.  
    *Aggregate diagnostics and bounded next actions; no invented spectra or examples.*

## Capability Boundary

**Actual evidence:** 다섯 공개 데이터셋의 대표 이미지, 검증된 AP-10K ROI 기록, 최종 DogFace holdout 보고서의 집계 지표.  
**Evaluation-only / upstream research:** detector, pose, ROI localization은 upstream 실험 경로이며 canonical `cvi.CVI`에 연결되어 있지 않습니다. OOF calibration과 A/F fusion은 DogFace holdout 평가에서 실제 실행됐지만 canonical runtime에는 연결되지 않았습니다. 공개 런타임은 caller-provided crop의 closed-set enrollment/retrieval만 수행하며, 최종 DogFace 평가는 publisher face crop에서 시작해 검출·포즈 단계를 포함하지 않습니다.

## Attribution

| Dataset | License | Official source |
|---|---|---|
| AP-10K domestic dog subset | CC-BY-4.0 | https://github.com/AlexTheBad/AP-10K |
| DogFaceNet 224 (resized) | CC-BY-4.0 | https://zenodo.org/records/12578449 |
| Multi-pose dog dataset | CC-BY-4.0 | https://data.mendeley.com/datasets/v5j6m8dzhv/1 |
| Sibetan | CC-BY-4.0 | https://www.lirmm.fr/YT-BB-Dog_Sibetan/ |
| YT-BB-Dog | CC-BY-4.0 | https://www.lirmm.fr/YT-BB-Dog_Sibetan/ |

이미지 파생물은 각 원 데이터셋의 CC-BY-4.0 조건을 유지하며 위 출처에 귀속됩니다. 저장소 코드의 Apache-2.0 라이선스가 데이터셋 권리를 대체하지 않습니다.

## Reproduction

```bash
export CVI_SECURE_DATA_ROOT=/path/to/canine_video_identity_secure
export CVI_ROI_MANIFEST=/path/to/roi_manifest.json
export CVI_EVALUATION_REPORT=/path/to/evaluation.json

uv run python tools/generate_report_visualizations.py \
  --data-root "$CVI_SECURE_DATA_ROOT" \
  --roi-manifest "$CVI_ROI_MANIFEST" \
  --evaluation-report "$CVI_EVALUATION_REPORT" \
  --output-dir Visualization
```

생성기는 기존 출력 디렉터리를 덮어쓰지 않습니다. 동일한 입력 바이트, 소스 코드, Pillow/폰트 환경에서 결정적으로 생성되며 `provenance.json`에 입력·출력 SHA-256을 기록합니다.
