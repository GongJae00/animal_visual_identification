# Canine Video Identity

Canine Video Identity(CVI)는 영상과 이미지에서 **같은 강아지를 다시 찾는 연구 프로젝트**입니다.

전체 목표는 코 하나만 보는 것이 아닙니다. 강아지의 **Appearance(전체 외형), Face(얼굴), Nose(코 주변)**를 각각 분석하고, 영상 여러 frame의 품질과 이용 가능성을 고려해 하나의 identity score로 결합하는 multi-evidence ReID architecture를 지향합니다.

현재 공개 API인 `cvi.CVI`는 이 전체 영상 workflow가 아닙니다. 호출자가 제공한 crop을 local gallery에 등록하고, 다른 crop에 대해 등록된 identity 후보를 순서대로 반환하는 엄격한 closed-set runtime입니다. 미등록견 거절이나 인증 판정은 아직 제공하지 않습니다.

## 한눈에 보는 현재 상태

| 영역 | 상태 | 현재 의미 |
|---|---|---|
| Crop-level gallery/runtime | 구현됨 | UUIDv5 identity 등록, strict evidence contract, exact weighted cosine scoring, versioned persistence |
| Appearance | 부분 구현·별도 평가 | 전체 crop visual evidence와 DogFace holdout 결과가 있으나 unified A/F/N cohort 재평가가 필요 |
| Face | 부분 구현·별도 평가 | global/regional face evidence가 있으나 기존 A/F calibration에서 추가 가치가 확립되지 않음 |
| Nose | 가장 많이 진행됨 | localization, SAM 2.1 teacher, mask student, consistency fine-tuning, K5 temporal fusion, score fusion까지 실행 |
| Quality/availability | software contract 구현 | branch 누락은 기록하지만 quality가 identity 성능을 얼마나 높이는지는 통합 검증이 부족 |
| Temporal aggregation | Nose에서 검증 | K1/K3/K5를 비교해 K5를 선택했지만 Appearance+Face+Nose 전체 track fusion은 미완성 |
| Unified A/F/N fusion | 미완성 | 동일한 identity/session/camera cohort의 `A`, `F`, `N`, `A+F`, `A+F+N` report가 필요 |
| Detection/tracking | offline 일부/미연결 | localization 연구 도구는 있으나 canonical runtime의 video detector/tracker는 아님 |
| Open-set unknown rejection | 미구현 | threshold와 unknown-dog evaluation이 없으며 `cvi.CVI`는 enabled open set을 거부 |
| Deployment service | 미구현 | service, authentication, privacy controls, supported serving facade가 없음 |

## 전체 Architecture

```text
OFFLINE MULTI-EVIDENCE RESEARCH WORKFLOW

video / source images
        |
        +-- detection, pose, tracking, ROI selection       [부분 구현/미연결]
        |
        +-- Appearance branch: 전체 crop, 털색, 체형, 패턴
        +-- Face branch: 얼굴 global + regional evidence
        +-- Nose branch: raw + segmentation-mask + restoration evidence
        |
        +-- Quality and availability
        |     blur, pose, ROI failure, missing branch를 기록
        |
        +-- Temporal aggregation
        |     여러 frame의 evidence를 track 수준으로 결합
        |
        +-- Calibration and fusion
        |     development identity에서 score scale과 weight를 고정
        |
        +-- closed-set ranking / future open-set decision


CANONICAL PUBLIC RUNTIME: cvi.CVI

caller-provided crop
        -> configured local evidence extractor(s)
        -> versioned local gallery
        -> available-intersection weighted cosine scoring
        -> maximum template score per UUIDv5 identity
        -> ordered closed-set candidates
```

Research branch가 repository에 존재한다는 사실만으로 public product capability가 되지는 않습니다. 현재 detector, tracker, temporal fusion, calibrated unknown rejection은 `cvi.CVI`에 연결되지 않았습니다.

## 현재 성능을 어떻게 읽어야 하는가

**아직 Appearance+Face+Nose 전체 시스템의 총성능 숫자는 없습니다.** 아래 결과는 서로 다른 offline protocol이므로 한 시스템의 연속된 성능표처럼 합치면 안 됩니다.

### Appearance/Face DogFace holdout

Publisher가 제공한 face crop에서 시작한 closed-set 결과입니다. Detection, tracking, native video ROI 성능은 포함하지 않습니다.

| Protocol | Appearance | Face | Frozen A/F output |
|---|---:|---:|---:|
| One-shot, 125 identities/649 queries | Rank-1 82.0%, MRR 87.5% | Rank-1 82.3%, MRR 87.5% | Rank-1 82.0%, MRR 87.5% |
| Three-shot, 107 identities/404 queries | Rank-1 94.8%, MRR 96.7% | Rank-1 94.8%, MRR 96.7% | Rank-1 94.8%, MRR 96.7% |

Calibration이 선택한 A/F weight는 `1.00/0.00`이었습니다. 따라서 이 report는 Face가 Appearance에 추가 이득을 줬다는 증거가 아닙니다. 상세 도표는 [final A/F holdout](Visualization/05_evaluation/05_final_results.svg)과 [calibration selection](Visualization/04_calibration_fusion/04_calibration_selection.svg)을 참고하십시오.

### Nose subarchitecture same-track diagnostic

이 결과는 raw Nose K5, student-mask K5, restoration score를 결합한 것이며 Appearance+Face+Nose fusion이 아닙니다.

| Metric | Raw Nose K5 | Raw+Mask+Restoration |
|---|---:|---:|
| Rank-1 | 74.53% | 75.78% |
| Rank-5 | 87.58% | 88.20% |
| mAP/MRR | 80.91% | 81.96% |

Fusion weight는 67 calibration identities에서 선택하고 161 identities에서 고정 평가했습니다. Rank-1 변화는 `+1.24%p`, identity-bootstrap 95% CI는 `[-1.86, +4.97]%p`로 0을 포함합니다. Gallery와 query는 같은 YT-BB track의 앞 5장과 뒤 5장이므로 cross-session 또는 biometric validation이 아닙니다.

Nose native materialization은 1,082 identities의 11,009 ROI records를 만들었지만 `AVAILABLE=397`, `LOW_QUALITY=10612`입니다. Mask student의 Dice `0.8264`, IoU `0.7184`는 수동 정답이 아니라 accepted SAM 2.1 pseudo-mask와의 일치도입니다.

자세한 protocol, 선행연구 비교와 논문화 범위는 [LiteratureReview.md](LiteratureReview.md)에 정리되어 있습니다.

## 완성도를 높이기 위한 강화 순서

| 단계 | 다음 강화 | 완료 기준 |
|---|---|---|
| Data handling | identity/session/camera/source-video 단위 split, near-duplicate audit, untouched final cohort, manual mask subset | 같은 track과 촬영 burst가 partition을 넘지 않고 final exposure가 0 |
| Preprocessing | admitted detector, dog association tracker, pose/ROI uncertainty, quality-diversity frame selection | video에서 재현 가능한 track과 A/F/N crop 생성 |
| Appearance backbone | background shortcut stress test, canine metric fine-tuning, ConvNeXt/DINO 계열 동일 protocol 비교 | cross-session baseline을 frozen DINO보다 유의하게 개선 |
| Face backbone | regional collapse 수정, pose/alignment augmentation, identity-disjoint 재학습 | 동일 cohort에서 Appearance에 양의 보완 가치 입증 |
| Nose backbone | multi-scale/high-resolution path, manual-mask validation, consistency loss ablation | raw 보존과 masked evidence 개선을 독립 cohort에서 재현 |
| Fusion | `A`, `F`, `N`, `A+F`, `A+F+N` OOF calibration, missing-modality stress test | EVAL labels 없이 고정한 fusion이 baseline을 개선 |
| Temporal/postprocessing | track purity, K selection, quality-diversity sampling, multi-prototype identity template | single-frame 대비 track-level 개선과 latency 보고 |
| Open set | genuine/impostor probe, frozen threshold, DIR/TPIR@FPIR | 미등록견을 포함한 independent evaluation 통과 |
| Analysis | identity-bootstrap CI, breed/color/quality/camera subgroup, failure gallery, compute profiling | 성능·불확실성·실패 원인을 함께 설명 |
| Runtime integration | artifact admission, public config, CPU/CUDA parity, gallery migration | research workflow 중 승인된 기능만 `cvi.CVI`에 연결 |

가장 먼저 필요한 것은 새로운 backbone을 무작정 추가하는 일이 아니라, **한 cohort에서 모든 branch를 공정하게 비교하는 unified baseline**입니다. 이후 결과가 약한 branch만 교체하거나 fine-tune합니다.

## Public Runtime 사용 형태

Linux, Python 3.12, [`uv`](https://docs.astral.sh/uv/)를 사용합니다. CPU와 CUDA lane 중 하나만 선택하십시오.

```bash
uv sync --extra cpu --group dev
# CUDA를 사용할 때는 cpu 대신 cuda를 선택합니다.
```

API 형태는 다음과 같습니다. `config`에는 사용자가 별도로 준비하고 검증한 local model, preprocessing, gallery contract가 필요합니다. Repository가 canine identity model을 자동으로 내려받아 주지는 않습니다.

```python
from PIL import Image

from cvi import CVI
from cvi.identity_registry import compute_registered_dog_id

runtime = CVI(config)
dog_id = compute_registered_dog_id("local:v1:dog:001")

runtime.enroll(Image.open("enrollment_crop.jpg").convert("RGB"), dog_id)
matches = runtime.search(Image.open("query_crop.jpg").convert("RGB"), top_k=5)
runtime.close()
```

설정과 artifact 요구사항은 [Configuration](docs/CONFIGURATION.md), 데이터와 model 정책은 [Data and Models](docs/DATA_AND_MODELS.md)을 참고하십시오.

## Engineering 원칙

- Raw datasets, weights, checkpoints, caches, galleries, experiment outputs는 Git 밖에 둡니다.
- Source, split, config, checkpoint, ONNX, report를 SHA-256과 versioned schema로 연결합니다.
- 기존 artifact를 덮어쓰지 않고 새 version으로 publish합니다.
- Unit/synthetic test 통과를 biometric 성능으로 표현하지 않습니다.
- 한 기능의 code, focused test, artifact validator를 함께 변경합니다.
- `uv run pytest`, `uv build`, `uv lock --check`, `git diff --check`로 변경 surface를 검증합니다.
- Dirty worktree의 다른 작업을 되돌리거나 한 commit에 섞지 않습니다.

## 문서

- [Architecture](docs/ARCHITECTURE.md)
- [Known Limitations](docs/KNOWN_LIMITATIONS.md)
- [Roadmap](docs/ROADMAP.md)
- [Literature Review](LiteratureReview.md)
- [Visualization Index](Visualization/INDEX.md)
- [Configuration](docs/CONFIGURATION.md)
- [Data and Models](docs/DATA_AND_MODELS.md)
- [Third-Party Licensing](THIRD_PARTY_LICENSES.md)
- [Contributing](CONTRIBUTING.md)

## License

Repository code and documentation are licensed under Apache-2.0. Third-party datasets, weights, source code, images, and generated artifacts retain their own terms. The repository license does not override DogFLW non-commercial conditions, dataset privacy obligations, or other upstream restrictions.
