# 강아지 개체 식별(ReID) 선행연구 조사와 CVI 비교

> 조사 기준일: 2026-07-30  
> 대상: 얼굴, 전신 외형, 코무늬(nose print), 영상/track을 이용한 강아지 개체 식별  
> 목적: 선행연구의 데이터·분할·방법·성능을 같은 기준으로 정리하고, 현재 CVI 연구의 재사용 범위, 독립 기여, 성능 위치와 논문화 가능성을 판단한다.

## 1. 결론부터 읽기

1. 강아지 개체 식별 연구는 이미 존재하지만, 논문마다 문제 설정이 크게 다르다. 정지 얼굴 분류, 1:1 검증, 코무늬 인증, 1:N 검색, 여러 영상 프레임을 합친 track ReID의 숫자를 그대로 비교하면 잘못된 결론이 나온다.
2. 현재 CVI와 가장 가까운 직접 비교 대상은 BIFOR의 `YT-BB-Dog -> Sibetan` 영상 ReID다. BIFOR는 다른 날짜와 카메라가 포함된 Sibetan에서 Rank-1 82.7%, mAP 69.8%를 보고했다. 현재 CVI의 post-hoc CAL/EVAL-separated 결과는 Rank-1 75.78%, mAP 81.96%지만 같은 YT-BB video track 안의 앞 5장과 뒤 5장을 비교한 결과다. CVI Rank-1은 수치상 6.92%p 낮고 mAP는 12.16%p 높지만, BIFOR는 여러 positive sequence를 갖는 cross-day/camera 문제이고 CVI는 identity당 gallery가 하나인 same-track 문제여서 우열을 주장할 수 없다.
3. 대규모 얼굴 기준인 PetFace는 dog seen-ID Top-1 77.86%, unseen-ID verification AUC 99.45%를 보고했다. 최근 CVI Nose subarchitecture의 fused Rank-1 75.78%와 수치상 2.08%p 차이지만, PetFace의 얼굴 이미지 분류와 CVI의 native nose-region track 검색은 서로 다른 과제다. 이 75.78%는 Appearance+Face+Nose 전체 architecture의 최종 성능이 아니다.
4. 고해상도 코무늬 연구는 매우 높은 수치를 보고한다. Bae et al.은 수동 crop한 302마리 스마트폰 코무늬에서 Rank-1 98.972%를 보고했다. CVI보다 수치상 23.20%p 높지만, fold의 identity 분리 여부가 명확하지 않고 입력 해상도·촬영 통제·평가 방식이 다르다.
5. 전체 연구 목표는 Appearance(전체 crop/전신 외형), Face, Nose를 함께 사용하고 quality, missing evidence, temporal aggregation, calibrated fusion으로 결합하는 multi-evidence canine video ReID다. 최근 실험은 이 중 Nose evidence를 깊게 구현·검증한 단계다. 단일 새 backbone보다 여러 evidence를 감사 가능하게 결합하는 architecture가 핵심 후보이며, 이 offline research workflow 전체가 public `canine_identity.IdentityEngine` runtime에 연결된 제품 capability는 아니다.
6. 현재 결과만으로도 workshop, applied computer vision, reproducibility/system 논문 초안은 가능하다. 그러나 강한 biometric 또는 일반화 ReID 논문을 위해서는 새로운 cross-session/camera cohort, 외부 benchmark, open-set unknown rejection, 동일 protocol의 강한 baseline 비교가 필요하다.
7. 현재 project artifact metadata는 DogFLW를 사용한 localizer와 downstream artifact를 `CC-BY-NC-4.0-derived`, `RESEARCH_ONLY`로 분류한다. 이는 상업 사용을 자동 허용한다는 결론을 피하기 위한 보수적 내부 정책이다. trained model의 법적 파생물 지위에 대한 확정적 법률 의견은 아니다.

## 2. 숫자를 비교하기 전에 알아야 할 용어

### 2.1 Identification과 verification은 다른 문제다

| 문제 | 질문 | 대표 지표 | 쉬운 예 |
|---|---|---|---|
| Closed-set classification | 사진이 미리 아는 N마리 중 누구인가? | Accuracy, Top-1 | 42마리 classifier |
| Closed-set 1:N retrieval/ReID | query와 같은 개가 gallery 안에 있다고 가정할 때 몇 등인가? | Rank-1, Rank-5, mAP | 등록된 개 중 후보 검색 |
| 1:1 verification | 두 사진이 같은 개인가? | ROC AUC, EER, TAR/VR@FAR | 코무늬 두 장 비교 |
| Open-set 1:N identification | 같은 개가 gallery에 없을 수도 있을 때 누구인지 또는 unknown인지 판단하는가? | DIR/TPIR@FAR/FPIR, FNIR@FPIR | 미등록견 거절 포함 |
| Track-level ReID | 여러 frame을 합친 query track이 어느 identity인가? | track Rank-k, track mAP | CCTV 영상 조각 비교 |

ROC AUC 99%는 Rank-1 99%와 같은 뜻이 아니다. AUC는 모든 threshold에 걸친 1:1 분리 능력이고, Rank-1은 gallery 크기와 distractor 구성에 영향을 받는 1:N 순위다. Open-set은 top candidate를 반환하는 것만으로 성립하지 않는다. unknown을 threshold로 거절해야 한다 [R23, R24].

### 2.2 데이터 분할이 성능보다 먼저다

| 분할 | 의미 | 난이도와 위험 |
|---|---|---|
| Random image/frame split | 같은 개와 같은 촬영 burst의 이미지가 train/test에 섞일 수 있음 | 가장 쉽고 near-duplicate·배경 누출 위험이 큼 |
| Identity-disjoint split | train과 test 개체가 다름 | unseen identity로 일반화하는 embedding 평가에 적합 |
| Session-disjoint gallery/query | 같은 test identity지만 날짜·촬영 세션이 다름 | 시간 변화와 재촬영에 더 현실적 |
| Camera/site-disjoint gallery/query | 카메라·장소가 다름 | 배경 shortcut을 줄임 |
| Source-video/track-disjoint | 같은 원본 영상의 frame이 양쪽에 가지 않음 | video ReID의 최소 조건 |
| Final untouched cohort | 개발·threshold·fusion 선택에 전혀 쓰지 않은 identity | 최종 주장에 필요 |

Kuncheva et al.은 인접 video frame을 random split하면 "deceptively high accuracy"가 나올 수 있다고 지적하고, 영상을 시간적으로 연속된 앞·뒤 block으로 나눴다 [R22]. WildlifeDatasets도 closed, open, identity-disjoint, time-aware split을 명시적으로 구분한다 [R21].

### 2.3 mAP도 protocol에 따라 다르다

현재 CVI post-hoc architecture diagnostic은 identity마다 gallery vector가 하나이고 relevant gallery도 하나다. 이 조건에서는 AP가 reciprocal rank와 같아져 `mAP = MRR`이다. 반면 BIFOR Sibetan은 한 identity에 여러 short-term sequence가 있어 query마다 relevant sequence가 여러 개일 수 있다. 따라서 두 mAP는 계산 대상이 달라 숫자 크기만으로 비교할 수 없다.

## 3. 연구 지형: 강아지의 어디를 보고 식별하는가

| 계열 | 장점 | 약점 | 대표 연구 |
|---|---|---|---|
| 얼굴 | 눈·귀·주둥이·털 패턴을 함께 사용, 일반 사진에 적용 가능 | pose, 귀 모양 변화, 가림, 털 길이, 배경 영향 | Moreira, DogFaceNet, PetFace |
| 전신 외형 | 멀리서도 사용 가능, camera-trap 영상과 잘 맞음 | 옷·목줄·털 상태·배경·자세 shortcut | MPFNet, BIFOR |
| 얼굴+전신 | 한 modality 실패 시 보완 가능 | detection과 score calibration이 복잡 | Azizi & Zaman, iFBI |
| 코무늬 | 사람 지문과 비슷한 국소 패턴을 기대, 얼굴보다 배경 영향이 작을 수 있음 | 실제 ridge를 보려면 근접·고해상도 촬영 필요 | Jang, Bae, Pet Biometric Challenge |
| 영상/track | 여러 frame 평균으로 pose·blur 변동을 줄임 | 같은 track frame끼리 비교하면 과대평가 위험 | BIFOR, 현재 CVI |

## 4. 얼굴·전신·영상 기반 직접 선행연구

### 4.1 핵심 비교표

| 연구 | 데이터와 분할 | 방법 | 보고 성능 | 비교 시 주의점 |
|---|---|---|---|---|
| Kumar & Singh (2014) [R1] | 40마리×10장, identity당 gallery 6장/probe 4장, 5회 partition | 얼굴 LBP·Gaussian scale | Rank-1 94.86% | 40마리의 소규모 seen-ID closed set, 데이터 비공개 |
| Moreira et al. (2017) [R2] | Flickr-dog 42마리/374장, Snoopybook 18마리/251장, stratified 10-fold image CV | BARK, WOOF 등 handcrafted descriptor | WOOF balanced accuracy 66.9%/89.4% | 같은 identity가 train/test에 존재하는 seen-ID 분류, session/source 분리 미보고 |
| Mougeot et al., DogFaceNet (2019) [R3] | 논문 당시 약 1,400장 규모, 48 unseen dogs open set | ResNet 계열, triplet loss, 3-landmark alignment | verification accuracy 92%, one-shot Rank-5 88% | 현재 8,363장 Zenodo 확장판과 논문 데이터가 다름. Rank-1 미보고 |
| Lai et al. (2019) [R4] | Flickr-dog 42마리×5장, identity별 image 5-fold | Xception 등 얼굴 classifier, breed·gender score fusion | 얼굴 78.09%, oracle breed+gender fusion 84.94% | train/test에 같은 identity. 84.94%는 정답 속성 조건이 포함됨 |
| Yoon et al. (2021) [R5] | DogFaceNet 901 train dogs/6,460장, 100 identity-disjoint evaluation dogs/580장 | triplet/ArcFace embedding과 vector-layer 변형 | proposed+ArcFace verification 88.8%; proposed+triplet one-shot Rank-1 39.74%, Rank-5 68.80% | evaluation set의 loss/accuracy를 training 중 확인해 checkpoint 선택에도 사용했으므로 untouched final test가 아님 |
| Azizi & Zaman (2023) [R6] | 여러 출처를 합친 245 pet profiles, identity당 1장 등록, 나머지 query | 얼굴·전신 cosine score 곱 late fusion | face 80%, body 81%, fusion 86.5%, metadata 포함 92% | dog/cat별 구성, session/camera, query 수가 충분히 보고되지 않음 |
| He et al., MPFNet (2023) [R7] | MPDD-192: 192마리/1,657장, 다양한 pose | pose-specific global/local branch와 weighted fusion | 원 논문의 정확한 MPDD 표 수치는 이번 조사에서 원문 확인 제한 | 공개 데이터는 유용하지만 미검증 수치를 인용하지 않음 |
| Shinoda & Shiohara, PetFace (ECCV 2024) [R8] | 전체 257,484 IDs/1,012,934장, dog 71,613 IDs. seen-ID re-ID와 identity/source-disjoint unseen verification 분리 | ResNet-50, Softmax/Center/Triplet/ArcFace | dog ArcFace seen Top-1 77.86%, unseen AUC 99.45% | 대규모 얼굴 정지영상. AUC는 open-set 1:N unknown rejection이 아님 |
| Neto et al., BIFOR (online 2025, issue 2026) [R9] | YT-BB-Dog 2,723 IDs/27,036 crops를 2,000 train/723 test IDs로 분리. Sibetan 59 dogs/5일/12 cameras, 평가 39 dogs/203 sequences/1,603 crops | ConvNeXt-Small, triplet loss, background-similar batch mining, sequence frame 평균 | YT-BB-Dog->Sibetan Rank-1 82.7%, Rank-5 93.6%, mAP 69.8% | 현재 CVI와 가장 가까우나 BIFOR는 cross-domain·cross-day/camera full-body sequence 문제 |

### 4.2 BIFOR가 특히 중요한 이유

BIFOR는 현재 조사에서 CVI와 가장 가까운 직접 dog video ReID 연구다 [R9]. 주요 특징은 다음과 같다.

- YT-BB-Dog identity를 train/test에서 분리했다.
- 최종 일반화 평가는 YT-BB-Dog가 아닌 장기 camera-trap Sibetan에서 수행했다.
- Sibetan은 5일과 12개 카메라로 수집되어 same-track보다 현실적인 변화를 포함한다.
- 각 short-term sequence의 frame embedding을 평균해 track/sequence 수준으로 비교한다.
- 배경을 무작위로 바꾸는 stress test에서 기존 방법의 큰 하락을 보여, dog ReID가 개보다 배경을 학습할 위험을 실험적으로 드러냈다.

단, 논문은 Sibetan mAP를 보며 일부 hyperparameter를 조정했다고 기술한다. 별도의 Sibetan validation split이 없다면 target benchmark가 development 역할도 했을 가능성을 논문 비교에서 밝혀야 한다.

### 4.3 PetFace가 보여주는 것

PetFace는 개만의 데이터셋은 아니지만 dog identity가 71,613개로 매우 크다 [R8]. 같은 identity를 학습한 seen-ID retrieval과, 학습에 없는 identity의 1:1 verification을 분리했다. 이 연구는 다음 두 점에서 중요하다.

- ArcFace가 animal face embedding에서도 강한 기준임을 보여준다.
- unseen identity와 source를 분리한 verification benchmark가 가능함을 보여준다.

그러나 PetFace AUC 99.45%를 "미등록견을 99.45% 정확도로 거절한다"고 해석하면 안 된다. gallery search와 unknown threshold를 함께 평가한 open-set 1:N 지표가 아니기 때문이다.

## 5. 코무늬(nose print) 기반 선행연구

### 5.1 핵심 비교표

| 연구 | 데이터와 분할 | 방법 | 보고 성능 | 비교 시 주의점 |
|---|---|---|---|---|
| Coldea (1994) [R10] | 코를 건조하고 먹물을 묻혀 판지에 접촉 인쇄 | 회전·중첩 수동/영상 비교 | 정량 benchmark 없음 | 역사적 contact nose-print 방법이며 현대 ReID 기준이 아님 |
| Jang et al. (2020) [R11] | 11마리, 원본 55장, 변형 영상 990장 | SIFT/SURF/ORB/BRISK 특징점과 matching | 전체 deformation 포함 ORB EER 0.35% | 11마리로 매우 작고 990장은 독립 재촬영이 아닌 합성 변형 |
| Bae et al. (2021) [R12] | 302마리/2,561장, 스마트폰 4,032×3,024에서 수동 nose crop, 5-fold CV | ResNet-152 Siamese, dual attention, contrastive+ArcFace | Rank-1 98.972%, VR 72.2%@FAR 0.1%, 63.5%@FAR 0.01% | fold별 identity overlap이 명시되지 않음. 데이터·코드 비공개 |
| Pet Biometric Challenge (2022) [R13, R14] | train 6,000마리/20,000장, validation 2,000 pairs, test 2,000 pairs | 다중 backbone, metric loss, augmentation, ensemble | 1위 test AUC 0.908699, 2위 0.888061, 3위 0.866735 | 1:1 pair AUC. 데이터는 대회 약관상 제한적이며 identity overlap 세부 미보고 |

### 5.2 Bae et al.의 높은 성능을 어떻게 봐야 하는가

Bae et al.은 현재 확인된 학술 논문 중 강한 고해상도 dog nose-print 기준이다 [R12]. 수동으로 nose ROI를 자르고, ResNet-152 Siamese embedding에 non-local dual attention, contrastive loss와 ArcFace를 결합했다.

Rank-1 98.972%는 인상적이지만 다음 이유로 CVI와 직접 비교할 수 없다.

- 원본이 4K 스마트폰 근접 촬영이고 CVI는 native 저해상도 video crop이다.
- 수동 crop과 자동 localizer의 오차 조건이 다르다.
- 5-fold가 image-disjoint인지 identity-disjoint인지 논문 설명만으로 확정하기 어렵다.
- 데이터가 공개되지 않아 같은 protocol 재현이 어렵다.
- VR@FAR와 Rank-1은 서로 다른 평가다.

### 5.3 Pet Biometric Challenge가 보여주는 것

CVPR 2022 Biometrics Workshop의 Pet Biometric Challenge는 1:1 nose-print verification을 AUC로 평가했다 [R13, R14]. 상위권은 다음 공통 요소를 사용했다.

- ResNet/ResNeSt/Swin/EfficientNet 계열의 강한 backbone
- cross-entropy, triplet, circle, supervised contrastive 등 metric losses
- blur, resize, JPEG, affine 등 저화질 augmentation
- GeM/SPoC/MAC 같은 descriptor aggregation
- 여러 모델과 scale의 ensemble

현재 CVI의 degraded-view consistency와 raw/masked score fusion은 이런 robustness·ensemble 방향과 문제의식은 유사하지만, 대회 코드나 weights, 제한 데이터는 사용하지 않았다.

## 6. 현재 CVI 연구의 실제 데이터와 protocol

이 절의 수치는 repository 코드와 외부 hash-bound artifact에서 확인한 실제 결과다. canonical public runtime, 전체 multi-evidence research architecture, 최근 Nose subarchitecture 결과를 서로 혼동하지 않는다.

### 6.1 Public runtime 경계

`canine_identity.IdentityEngine`은 사용자가 제공한 crop을 enrollment하고 closed-set 후보를 반환한다. video decoding, detection, tracking, frame selection, temporal aggregation, unknown rejection은 canonical public capability가 아니다. 자세한 경계는 `README.md`, `AGENTS.md`, `docs/KNOWN_LIMITATIONS.md`에 기록되어 있다.

### 6.2 전체 multi-evidence research architecture

CVI는 코만 보는 구조가 아니다. 논문화할 전체 architecture를 아주 단순하게 그리면 다음과 같다.

```text
강아지 video/crops
      |
      +-- Appearance: 전체 crop, 털색, 체형, 전신 패턴
      +-- Face: 얼굴 형태, 눈·귀·주둥이 주변 특징
      +-- Nose: raw nose, segmentation mask, restoration evidence
      |
      +-- Quality/availability: 흐림, 자세, ROI 실패, branch 누락 기록
      +-- Temporal fusion: 여러 frame을 identity evidence로 집계
      +-- Calibrated fusion: branch score를 development identity에서 보정
      |
      +-- Closed-set gallery ranking
```

| Evidence | 하는 일 | 현재 위치 |
|---|---|---|
| Appearance | 전체 dog crop의 broad visual identity signal을 384D 계열 embedding으로 표현 | crop-level research/evidence path가 있으며 전체 multi-channel baseline 역할 |
| Face | 얼굴 global 및 regional 특징으로 appearance와 다른 단서를 제공 | frozen F0를 same-track unified cohort에서 평가했지만 Appearance 대비 추가 이득은 미확립 |
| Nose | raw, student-masked, restoration branch와 K5 temporal score를 제공 | 현재 가장 깊게 구현·검증된 신규 evidence branch |
| Quality/availability | 저화질·ROI 실패·optional evidence 누락을 숨기지 않고 기록 | strict contract와 research fusion에서 사용 |
| Temporal aggregation | frame 하나보다 여러 frame의 안정된 identity evidence를 사용 | Nose K1/K3/K5와 frozen A0/F0/N3 공통 K5 same-track diagnostic에서 검증 |
| Calibrated fusion | Appearance, Face, Nose의 서로 다른 score scale을 calibration 후 결합 | DEV 29에서 weight를 고정해 EVAL 105에 적용했으나 Appearance baseline 개선에는 실패 |

따라서 아래 표와 75.78% 결과는 **전체 CVI 중 최근 Nose subarchitecture의 상세 evidence**다. 별도의 same-track unified diagnostic은 첫 공통 baseline을 제공하지만 최종 main table은 동일한 cross-session cohort에서 `Appearance`, `Face`, `Nose`, `A+F`, `A+F+N`, temporal fusion을 다시 비교해야 한다.

### 6.3 최근 Nose subarchitecture pipeline

| 단계 | 실제 구성 | 학습 또는 평가 역할 |
|---|---|---|
| Nose localization | AP-10K domestic dog + DogFLW facial landmarks | nose ROI/localizer 학습 |
| Base embedding | DINOv2-small ViT-S/14 + ArcFace/parent consistency | DogFaceNet224, old YT crop으로 train; MPDD로 DEV |
| Native video materialization | YT-BB-Dog native publisher frame에서 nose crop | 1,082 identities, 11,009 ROI records. AVAILABLE 397, LOW_QUALITY 10,612, NO_ROI 0 |
| Mask teacher | 공식 SAM 2.1 small | prompt 기반 pseudo-mask 6,835 accepted |
| Mask student | MobileNetV4 Conv Small 기반 spatial decoder | identity-disjoint teacher-mask DEV에서 SAM 2.1 accepted pseudo-mask 대비 Dice 0.8264, IoU 0.7184. 수동 ground-truth segmentation 정확도가 아님 |
| Embedding consistency v3 | raw parent anchor + masked/degraded/native temporal consistency | SSL TRAIN 777 identities; parent-unseen DEV 77/EVAL 228 |
| Temporal fusion | frame별 L2 -> K개 균등 평균 -> 최종 L2 | K=1/3/5 비교, K5 선택 |
| Architecture fusion | raw K5, student-mask K5, restoration score를 row z-score 후 nonnegative simplex search | embedding optimization에서 제외된 228 identities를 post-hoc CAL 67/EVAL 161로 재분할 |

### 6.4 Frozen A0/F0/N3 공통 cohort diagnostic

기존 trained Appearance-v3와 Face-v4는 대상 YT identity에 노출됐으므로 공통 평가에서 제외했다. Appearance와 Face에는 frozen DINOv2-small, Nose에는 consistency-v3 raw embedding을 사용했다. Face crop availability만 보면 DEV 40/EVAL 142였지만 다중견 image에서 대상 identity와 Face를 결합할 수 없는 경우를 fail-closed로 제외해 DEV 29/EVAL 105를 사용했다.

| Method | EVAL Rank-1 | EVAL Rank-5 | EVAL MRR/mAP |
|---|---:|---:|---:|
| A0 frozen Appearance K5 | 94.29% | 99.05% | 96.56% |
| F0 frozen Face K5 | 89.52% | 95.24% | 92.24% |
| N3 consistency raw Nose K5 | 77.14% | 86.67% | 82.16% |
| A0+F0 | 94.29% | 99.05% | 96.03% |
| A0+F0+N3 | 94.29% | 99.05% | 96.03% |

DEV-only simplex calibration은 A/F/N weight `0.75/0.25/0.00`을 선택했다. A0+F0+N3의 A0 대비 Rank-1 delta는 `0.00%p`, 95% identity-bootstrap CI `[-2.86, +2.86]%p`이고 MRR delta는 `-0.53%p`, CI `[-1.97, +0.95]%p`다. 즉 첫 공통 cohort에서도 Face와 Nose의 추가 가치가 입증되지 않았다. Gallery/query가 같은 track의 앞·뒤 frame이므로 이 수치는 cross-session 최종 성능이 아니다.

근거 artifact는 `yt-unified-multievidence-a0-f0-n3-v1-20260802.json`, report payload `report_sha256=1d66e4a87e34db37f786d38e42cb8fbe080b50ba2bf7a87293c66e5a626b4352`다.

학습·평가 population을 수치로 풀면 다음과 같다.

| Artifact 단계 | TRAIN/SSL | DEV/CAL | EVAL | 비고 |
|---|---|---|---|---|
| Parent nose embedding | DogFaceNet 3,624 records/952 identities + old YT 806 records/405 identities, 합계 4,430 records/1,357 identities | MPDD 52 records/20 identities, 이 중 leave-one-out eligible 43 records/11 identities | 별도 untouched final 없음 | DogFaceNet과 old YT는 identity-supervised ArcFace train |
| Segmentation student | accepted SAM 2.1 masks 5,533 records/862 identities | accepted masks 1,302 records/216 identities | 수동 정답 final 없음 | identity-disjoint train/dev지만 target은 pseudo-mask |
| Consistency embedding v3 | native YT 6,944 records/777 identities, identity label classification이 아닌 SSL consistency | 1,024 records/77 parent-unseen identities | 3,041 records/228 parent-unseen identities | EVAL 228은 embedding optimization에서 제외 |
| Architecture calibration | 해당 없음 | 위 228 중 67 identities로 fusion weight 선택 | 나머지 161 identities로 weight-fixed diagnostic | EVAL 161은 fusion fitting에서만 분리됐고 완전 untouched는 아님 |

### 6.4 Nose 실험 분할을 쉽게 설명하면

1. 기존 embedding이 old YT crop으로 본 405 identities를 기록했다.
2. native YT에서 10장 이상 localized frame이 있고 parent가 보지 않은 305 identities를 선택했다.
3. SHA-256 identity split으로 DEV 77과 EVAL 228을 만들었다.
4. 나머지 777 identities는 identity label을 직접 맞히는 ArcFace 대상이 아니라 consistency SSL에 사용했다.
5. EVAL 228 identities는 embedding optimization에서 제외했다.
6. 228 전체의 raw/masked diagnostic을 먼저 계산한 뒤, architecture weight 선택을 위해 독립 seed로 CAL 67과 EVAL 161로 다시 나눴다.
7. EVAL 161은 fusion weight fitting에는 사용되지 않았지만 이전 K1/K3/K5 protocol 분석에 포함됐고, segmentation student는 identity supervision 없이도 최종 228 identities 전부의 teacher-mask 이미지에 노출됐다. 따라서 final untouched 또는 완전한 image-unseen cohort가 아니라 post-hoc CAL/EVAL-separated same-track diagnostic이다.

### 6.5 현재 Nose subarchitecture 성능

#### Embedding consistency 전후, parent-unseen 228 identities

| Branch | Parent embedding | Consistency v3 | 차이 |
|---|---:|---:|---:|
| Raw K5 Rank-1 | 73.68% | 72.81% | -0.88%p |
| Raw K5 mAP/MRR | 79.53% | 79.34% | -0.19%p |
| Masked K5 Rank-1 | 57.89% | 64.04% | +6.14%p |
| Masked K5 mAP/MRR | 67.60% | 71.88% | +4.28%p |

raw branch는 거의 보존되고 masked branch가 개선됐다. 이것은 mask consistency가 목적대로 representation gap을 줄였다는 evidence다. 다만 같은 YT track 내 평가이므로 일반적 biometric 개선으로 확대 해석할 수 없다.

#### Post-hoc calibrated architecture diagnostic, 161 EVAL identities

CAL 67에서 선택된 score weight는 raw `0.65`, student-mask `0.30`, restoration `0.05`다.

| Metric | Raw K5 baseline | Calibrated fusion | 차이 | Identity-bootstrap 95% CI |
|---|---:|---:|---:|---:|
| Rank-1 | 74.53% | 75.78% | +1.24%p | [-1.86, +4.97]%p |
| Rank-5 | 87.58% | 88.20% | +0.62%p | [-1.24, +3.11]%p |
| mAP/MRR | 80.91% | 81.96% | +1.06%p | [-0.70, +2.94]%p |

point estimate는 모두 양수지만 CI가 0을 포함한다. 따라서 "fusion이 유의하게 향상됐다"가 아니라 "fusion-weight-fit에서 분리된 EVAL subset의 point estimate에서 일관된 양의 개선이 관찰됐고, 완전히 untouched인 더 큰 독립 cohort가 필요하다"고 써야 한다.

현재 근거 artifact:

- `nose-region-embedding-consistency-v3-20260730`, lineage payload `lineage_sha256=46b70a9222d154a5661b221f64ea8f2a9451eeec050e10621806c38dcab1c15e`
- `yt-nose-architecture-eval-v4-consistency-heldout-20260730.json`, report payload `report_sha256=7d9fa6ea1f2010fcaa764893d59e9870c35a0375a6b266e108b09adc58e59e0e`
- 구현: `localization/nose_region/embedding_consistency_training.py`, `experiments/nose_architecture.py`

## 7. 선행연구와 최근 CVI Nose subarchitecture의 성능 비교

Appearance+Face+Nose의 same-track 동일-cohort diagnostic은 생겼지만 cross-session final report는 아직 없으므로, 아래 비교는 전체 CVI 최종 성능표가 아니다.

### 7.1 숫자 차이표

아래 차이는 독자가 규모를 이해하기 위한 산술값이다. `직접 비교 가능성`이 낮으면 성능 우열의 근거가 아니다.

| 기준 | 선행 결과 | CVI 대응 결과 | 산술 차이 | 직접 비교 가능성 |
|---|---:|---:|---:|---|
| BIFOR Sibetan Rank-1 [R9] | 82.70% | Nose fused Rank-1 75.78% | CVI Nose -6.92%p | 중간 이하: 둘 다 video aggregation이지만 BIFOR는 cross-domain/day/camera, CVI는 same-track |
| BIFOR Sibetan mAP [R9] | 69.80% | Nose fused mAP 81.96% | CVI Nose +12.16%p | 낮음: relevant gallery 수와 ranking protocol이 다름 |
| PetFace dog seen Top-1 [R8] | 77.86% | Nose fused Rank-1 75.78% | CVI Nose -2.08%p | 낮음: 얼굴 seen-ID image benchmark 대 nose-region unseen-ID track 검색 |
| Moreira Flickr WOOF [R2] | 66.90% | Nose fused Rank-1 75.78% | CVI Nose +8.88%p | 매우 낮음: 42-ID seen-ID image CV 대 161-ID post-hoc same-track subset |
| Azizi face+body fusion [R6] | 86.50% | Nose fused Rank-1 75.78% | CVI Nose -10.72%p | 매우 낮음: pet profile still image와 split 보고 차이 |
| Bae nose Rank-1 [R12] | 98.972% | Nose fused Rank-1 75.78% | CVI Nose -23.20%p | 매우 낮음: 4K 수동 nose crop과 저해상도 자동 video crop |
| DogFaceNet verification [R3] | Accuracy 92% | 해당 없음 | 계산 금지 | 서로 다른 metric |
| PetFace unseen verification [R8] | AUC 99.45% | 해당 없음 | 계산 금지 | AUC와 Rank-1은 다름 |
| Pet Challenge 1위 [R13] | AUC 90.87% | 해당 없음 | 계산 금지 | 1:1 pair AUC와 1:N retrieval은 다름 |

### 7.2 공정하게 말하면 현재 Nose branch는 어디쯤인가

- 현재 Nose fused Rank-1 75.78%는 "전체 CVI 성능"도, "압도적 SOTA"도 아니다.
- 대규모 얼굴 PetFace의 seen-ID Top-1과 수치상 비슷하지만 task가 다르다.
- BIFOR보다 Rank-1이 낮고, BIFOR가 더 현실적인 cross-day/camera 조건을 갖는다.
- 고해상도 수동 코무늬 benchmark보다 수치는 크게 낮다. CVI는 저해상도 자동 ROI라는 어려움을 다루지만 same-track 평가라는 상대적으로 쉬운 축도 있어 전체 난이도의 우열은 단정할 수 없다.
- Nose branch의 현재 가장 강한 주장은 최고 절대 성능이 아니라, native video nose crop에서 mask evidence를 개선하고 CAL/EVAL-separated score fusion까지 offline research workflow로 재현했다는 점이다. 전체 CVI의 강한 주장은 Appearance+Face+Nose가 동일 cohort에서 실제로 보완적임을 보인 뒤에 가능하다.

## 8. 우리가 선행연구를 얼마나 직접 활용했는가

### 8.1 직접 사용한 데이터·weights·아이디어

| 외부 자원 | 사용 형태 | 직접성 | 논문에서 해야 할 일 |
|---|---|---|---|
| DINOv2-small [R15] | exact pretrained ViT-S/14 weight를 embedding 초기화에 사용 | 매우 직접적 | 논문, model ID/revision, preprocessing, hash를 인용 |
| SAM 2.1 small [R16] | 공식 checkpoint를 pseudo-mask teacher로 실행 | 매우 직접적 | SAM 2 논문, SAM 2.1 checkpoint/config/commit을 기록 |
| MobileNetV4 Conv Small/timm [R17] | segmentation student backbone 초기화 | 직접적 | MobileNetV4와 timm weight/model card를 함께 인용 |
| ArcFace [R18] | 자체 구현한 angular-margin classification loss | 아이디어 직접 활용 | ArcFace 논문과 scale/margin을 명시. InsightFace weight를 썼다고 쓰지 않음 |
| AP-10K [R19] | domestic dog pose/localization 학습 데이터 | 데이터 직접 사용 | dog subset 추출과 split을 설명하고 CC-BY attribution |
| DogFLW [R20] | 46-landmark 얼굴 데이터에서 nose localizer supervision | 데이터 직접 사용 | `CC-BY-NC-4.0`과 research-only 파생 계보를 명시 |
| DogFaceNet Zenodo [R3] | identity-labeled nose crop embedding 학습 | 데이터 직접 사용 | PRICAI 논문과 Zenodo dataset DOI를 모두 인용 |
| YT-BB-Dog [R9, R25] | native video track, SSL, temporal/architecture 평가 | 데이터 직접 사용 | 원 YT-BB와 BIFOR/YT-BB-Dog 계보를 모두 인용 |
| MPDD-192 [R7] | base embedding DEV diagnostic | 데이터 직접 사용 | Mendeley DOI와 ICASSP 논문 인용 |

### 8.2 사용하지 않은 것

- DogFaceNet의 TensorFlow model checkpoint나 network code를 현재 embedding으로 사용하지 않았다. 데이터와 연구 맥락은 직접 활용했지만 모델은 DINOv2 기반 독립 pipeline이다.
- BIFOR code, ConvNeXt checkpoint, background sampler를 현재 model에 복사하거나 fine-tune하지 않았다. YT-BB-Dog 데이터 계보와 sequence-ReID 문제 설정만 직접 관련된다.
- Bae et al.의 DNNet code, private 302-dog 데이터, ResNet-152 weights를 사용하지 않았다.
- Pet Biometric Challenge의 제한 데이터나 참가자 ensemble weights를 사용하지 않았다.
- PetFace 데이터나 checkpoint를 사용하지 않았다.
- MiewID/MegaDescriptor는 repository에 comparator scaffold가 있으나 현재 reported nose architecture 결과를 만든 실행 model이 아니다.

### 8.3 직접 활용 비중에 대한 정직한 판단

정량적인 "몇 %가 독립" 계산은 의미가 없다. 대신 연구 구성 요소를 나누면 다음과 같다.

| 구성 | 기존 연구 의존도 | CVI 독립성 |
|---|---|---|
| Backbone architecture | 높음 | DINOv2, SAM 2.1, MobileNetV4를 새로 발명한 것이 아님 |
| Metric-learning 원리 | 높음 | ArcFace, cosine retrieval, consistency learning은 표준 원리 |
| Dog data | 높음 | 공개 DogFaceNet, YT-BB-Dog, AP-10K, DogFLW, MPDD를 사용 |
| Native video nose materialization | 중간 이하 | publisher frame에서 exact source-bound nose crop과 manifest를 만든 pipeline은 독립 구현 |
| SAM teacher-to-student nose mask | 중간 | foundation teacher와 mobile student 자체는 기존이지만 dog nose prompt, acceptance, uncertainty, lineage는 독립 설계 |
| Raw/masked/degraded consistency | 중간 | consistency 원리는 기존이지만 parent anchor, quality usage, temporal pair, selection gate 조합은 project-specific |
| Strict K5 temporal fusion | 낮은 알고리즘 novelty | 단순 평균은 표준적이지만, quality heuristic보다 검증된 균등 평균을 선택하고 artifact-bound한 것은 재현성 기여 |
| Three-branch calibrated score fusion | 중간 | z-score/simplex fusion은 표준적이지만 CAL/EVAL identity 분리와 raw/mask/restoration 적용은 project-specific |
| Provenance와 fail-closed artifact contract | 높음 | source hash, code hash, checkpoint, split, ONNX parity, no-overwrite publication을 하나의 ReID 흐름에 통합한 project-specific engineering contribution candidate. 학술적 novelty는 별도 provenance/reproducible-ML 문헌조사가 필요 |

## 9. 무엇이 논문 기여가 될 수 있는가

### 9.1 전체 CVI에 가장 설득력 있는 논문 주제

가장 적합한 framing은 코 하나가 아니라 다음과 같은 multi-evidence architecture다.

> **비통제 dog video에서 Appearance, Face, Nose evidence를 quality·availability-aware temporal fusion으로 결합하는 provenance-controlled canine ReID architecture**

Nose 연구는 이 전체 논문의 중요한 신규 branch이자 상세 ablation이다. 별도 소논문으로 분리한다면 segmentation-aware nose-region ReID framing도 가능하다.

가능한 contribution 문장은 다음 수준이어야 한다.

1. Appearance, Face, Nose가 서로 다른 identity 단서를 제공하도록 분리하고, branch 누락과 quality를 명시적으로 처리하는 multi-evidence architecture.
2. frame-level evidence를 track-level로 집계하고 development identity에서만 score를 calibration하는 temporal fusion protocol.
3. AP-10K와 DogFLW localization, SAM 2.1 teacher, MobileNetV4 student를 연결한 weakly supervised native-video Nose branch.
4. raw parent signal을 보존하면서 masked/degraded/native temporal views를 정렬하는 Nose consistency fine-tuning.
5. `Appearance`, `Face`, `Nose`, `A+F`, `A+F+N`의 same-track exposure-audited baseline과 후속 cross-session ablation.
6. source, split, checkpoint, ONNX, code까지 content-bound하는 reproducibility artifact contract.

### 9.2 현재만으로 주장하기 어려운 것

- "코무늬 biometric을 검증했다": native YT crop에는 실제 nasal ridge가 충분히 보이지 않을 수 있다.
- "cross-session dog identity를 해결했다": 현재 query/gallery는 같은 video track의 앞·뒤 frame이다.
- "open-set 시스템이다": unknown rejection threshold가 없다.
- "BIFOR/PetFace/Bae보다 우수하다": protocol이 다르다.
- "segmentation fusion이 통계적으로 유의하다": 현재 bootstrap CI가 0을 포함한다.
- "Appearance+Face+Nose 전체 fusion이 검증됐다": 세 evidence를 같은 cross-session cohort에서 평가한 unified final report가 아직 없다.
- "상용화 가능하다": DogFLW NC 계보와 데이터 권리, privacy, deployment validation이 해결되지 않았다.

### 9.3 현재 증거에 적합한 논문 positioning

| 목표 | 현재 가능성 | 이유 |
|---|---|---|
| 내부 연구보고서/technical report | 높음 | 실제 artifact와 재현 가능한 결과가 충분함 |
| Workshop 또는 applied CV paper | 중간~높음 | pipeline과 ablation이 있으며, 한계를 정직하게 쓰면 가능 |
| Dataset/protocol paper | 중간 | 새로운 cross-session dataset과 공개 가능한 annotation이 추가되면 강해짐 |
| 일반 animal ReID journal | 중간 이하 | BIFOR, PetFace, MPFNet 등 동일 split baseline과 외부 평가가 더 필요 |
| Biometric journal의 강한 인증 claim | 낮음 | macro nose, cross-session/device, open-set FAR/FRR, longitudinal cohort가 없음 |
| Product/clinical/forensic claim | 현재 불가 | 실제 운영 validation과 법적·privacy·unknown-risk 근거가 없음 |

## 10. 논문 전에 반드시 추가할 실험

### 10.1 최우선: cross-session/camera test

1. 같은 개를 날짜, 장소, 카메라가 다른 최소 2~3개 session에서 촬영한다.
2. train, development, calibration, final identities를 분리한다.
3. final identity의 모든 이미지는 model, segmentation, threshold, fusion weight 선택에서 제외한다.
4. gallery와 query를 다른 session에 둔다.
5. 같은 원본 video/track/burst frame은 절대 partition을 넘지 않게 한다.

### 10.2 동일 protocol baseline

최소한 아래 모델을 같은 crop, 같은 identity, 같은 gallery/query로 다시 평가해야 한다.

| Baseline | 목적 |
|---|---|
| DINOv2 frozen raw | canine fine-tuning의 가치 확인 |
| Current parent embedding | consistency v3의 가치 확인 |
| DogFaceNet reproduction | 고전 dog-face metric baseline |
| ArcFace/Triplet single-stream | 복잡한 pipeline이 단순 metric learning보다 나은지 확인 |
| BIFOR 또는 저자 checkpoint | full-body/video 기준과 비교. weight license 확인 필요 |
| PetFace/MegaDescriptor 계열 | 대규모 animal face/general wildlife representation 비교 |
| Raw K1/K3/K5 | temporal aggregation의 독립 기여 |
| Raw+mask, raw+restoration, all | branch별 ablation |
| Appearance only / Face only / Nose only | 각 큰 evidence channel의 독립 성능 |
| Appearance+Face / Appearance+Face+Nose | multi-evidence architecture의 실제 추가 가치 |
| Modality missing/low-quality stress test | 얼굴 또는 코가 안 보일 때 availability-aware fusion 검증 |

### 10.3 Open-set protocol

Liao et al.의 구조처럼 train, gallery, genuine probe, impostor probe를 분리한다 [R24].

- `DIR/TPIR @ FAR/FPIR`를 Rank-1과 Rank-5에서 보고한다.
- threshold와 fusion weight는 development/calibration에서 고정한다.
- final에서 threshold를 다시 고르지 않는다.
- 데이터로 지지할 수 없는 매우 낮은 FPIR operating point는 보고하지 않는다.
- closed-set Rank-k와 open-set decision 성능을 별도 표로 둔다.

### 10.4 통계와 failure analysis

- identity-clustered bootstrap CI를 유지한다.
- breed, coat color, age, sex, camera, illumination, blur, nose resolution subgroup을 가능한 범위에서 보고한다.
- background replacement 또는 masking stress test를 넣어 배경 shortcut을 확인한다.
- mask 실패, localizer 실패, low-resolution, pose, 동일 품종 hard negative를 정성·정량 분석한다.
- segmentation student가 final identity image를 본 영향을 분리하기 위해 image-unseen teacher/student split을 추가한다.

## 11. 논문 구성 권고

### 11.1 전체 CVI 논문의 권장 제목 예시

> **Multi-Evidence Canine Video Re-Identification with Appearance, Face, and Nose Cues**

또는 quality와 provenance를 강조하면 다음과 같다.

> **Availability-Aware Temporal Fusion of Appearance, Face, and Nose Evidence for Canine Re-Identification**

Nose branch만 별도 논문화할 때의 제목 예시는 다음과 같다.

> **Segmentation-Aware Temporal Nose-Region Re-Identification of Dogs in Low-Resolution Videos**

`Nose-print biometric`이라는 표현은 실제 ridge를 검증한 macro 데이터가 생기기 전에는 피하는 것이 안전하다. 현재 입력은 "nose-region" 또는 "muzzle/nasal-region" evidence가 더 정확하다.

### 11.2 권장 논문 목차

1. Introduction: 비접촉 dog ID와 Appearance·Face·Nose evidence의 상호 보완성
2. Related Work: dog face/body ReID, nose-print verification, video animal ReID
3. Data and Governance: 데이터 계보, license, identity/session split, leakage audit
4. Method: Appearance, Face, Nose branches, quality/availability, SAM teacher/student, temporal aggregation, calibrated fusion
5. Experimental Protocol: closed-set와 open-set 분리, baselines, metrics, bootstrap
6. Results: A/F/N single branch, A+F/A+F+N fusion, temporal scaling, Nose ablation, external test
7. Failure and Ethics: 배경 shortcut, 저해상도 ridge 한계, privacy, NC license
8. Conclusion: 연구 범위만 요약하고 제품·biometric claim을 하지 않음

### 11.3 표와 figure 우선순위

- Dataset/split diagram: identity, session, camera, track의 경계
- Pipeline figure: Appearance/Face/Nose parallel branches -> quality/availability -> temporal aggregation -> calibrated fusion
- Main comparison table: 동일 protocol로 재실행한 baselines만 포함
- Literature comparison table: 본 문서처럼 protocol 차이를 명시
- K1/K3/K5 curve
- Raw/masked embedding consistency ablation
- CAL/EVAL weight와 fusion-fit-separated EVAL CI
- Failure gallery: blur, profile pose, mask leakage, same-breed confusion

## 12. 라이선스와 공개 시 주의사항

| 자원 | 확인 조건 | 논문화·배포 주의 |
|---|---|---|
| Repository code | Apache-2.0 | 제3자 데이터·weights 권리를 대신하지 않음 |
| DINOv2, SAM 2/2.1, timm | Apache-2.0 계열 | 정확한 model/revision/checkpoint attribution 유지 |
| DogFaceNet Zenodo | CC-BY-4.0 | Zenodo DOI, crop/resize 변경 표시 |
| AP-10K | CC-BY-4.0 | domestic dog subset 사용을 명시 |
| DogFLW | CC-BY-NC-4.0 | 연구 전용. 상업 사용·파생 model 권리는 별도 검토 |
| MPDD-192 | CC-BY-4.0 | Mendeley dataset DOI 인용 |
| YT-BB/YT-BB-Dog | 배포 페이지 CC-BY-4.0, 원 YouTube 권리는 별도 문제 | 논문 사용과 frame 재배포를 구분. source snapshot과 attribution 유지 |
| Pet Challenge data | 표준 public license 없음, competition agreement 제한 | 데이터·이미지·weights를 본 연구가 사용했다고 오해하지 않게 명시 |
| BIFOR checkpoint | 명시적 weight license 미확인 | baseline 실행 전 저자 조건 확인 |

논문 참고문헌 인용만으로 CC attribution 의무가 항상 끝나는 것은 아니다. 공개 figure나 파생 crop에는 출처, license URL, 변경 여부를 함께 표시해야 한다.

## 13. 최종 판단

현재 CVI는 코만 보는 연구가 아니다. 전체 목표는 Appearance, Face, Nose를 병렬 evidence로 만들고, quality와 availability를 기록하며, video temporal aggregation과 calibrated fusion으로 개체 후보를 검색하는 것이다. 최근 Nose branch에서는 localization, segmentation distillation, consistency adaptation, K5, score fusion까지 실제 evidence를 만들었다. 이 multi-evidence 조합은 project-specific research contribution candidate지만, 학술적 novelty를 주장하려면 unified A/F/N 평가와 reproducible ML, provenance-bound artifacts, weakly supervised segmentation 문헌 비교가 더 필요하다.

현재 증거는 multi-evidence architecture technical report와 Nose branch workshop/applied-paper 초안에 적합하다. 전체 CVI 논문을 강하게 만들려면 동일 cross-session/camera cohort에서 Appearance, Face, Nose, A+F, A+F+N을 모두 실행해야 한다. 그 결과와 open-set protocol이 추가되면 일반 canine/animal video ReID 논문 주장이 강해진다. high-resolution macro nose와 longitudinal cohort는 전체 architecture의 Nose evidence를 biometric 수준으로 확장하는 별도 단계다.

## 14. 참고문헌

### 강아지 얼굴·전신·영상 ReID

**[R1]** S. Kumar and S. K. Singh, “Biometric Recognition for Pet Animal,” *Journal of Software Engineering and Applications*, 2014. [DOI: 10.4236/jsea.2014.75044](https://doi.org/10.4236/jsea.2014.75044).

**[R2]** T. P. Moreira, M. L. Perez, R. de O. Werneck, and A. Rocha, “Where is my puppy? Retrieving lost dogs by facial features,” *Multimedia Tools and Applications*, 76, 15325–15340, 2017. [DOI: 10.1007/s11042-016-3824-1](https://doi.org/10.1007/s11042-016-3824-1).

**[R3]** G. Mougeot, D. Li, and S. Jia, “A Deep Learning Approach for Dog Face Verification and Recognition,” *PRICAI 2019*, pp. 418–430. [DOI: 10.1007/978-3-030-29894-4_34](https://doi.org/10.1007/978-3-030-29894-4_34). Code: [DogFaceNet](https://github.com/GuillaumeMougeot/DogFaceNet). Dataset: [Zenodo 10.5281/zenodo.12578449](https://doi.org/10.5281/zenodo.12578449).

**[R4]** C.-L. Lai, C.-Y. Tsai, and H.-Y. Man, “Dog Identification using Soft Biometrics and Neural Networks,” *IJCNN 2019*. [DOI: 10.1109/IJCNN.2019.8851971](https://doi.org/10.1109/IJCNN.2019.8851971).

**[R5]** Y. Yoon et al., “A Methodology for Utilizing Vector Space to Improve the Performance of a Dog Face Identification Model,” *Applied Sciences*, 11(5), 2074, 2021. [DOI: 10.3390/app11052074](https://doi.org/10.3390/app11052074).

**[R6]** E. Azizi and B. Zaman, “Deep Learning Pet Identification Using Face and Body,” *Information*, 14(5), 278, 2023. [DOI: 10.3390/info14050278](https://doi.org/10.3390/info14050278).

**[R7]** Z. He, J. Qian, D. Yan, C. Wang, and Y. Xin, “Animal Re-Identification Algorithm for Posture Diversity,” *ICASSP 2023*. [DOI: 10.1109/ICASSP49357.2023.10094783](https://doi.org/10.1109/ICASSP49357.2023.10094783). Dataset: [Multi-pose dog dataset](https://doi.org/10.17632/v5j6m8dzhv.1).

**[R8]** R. Shinoda and K. Shiohara, “PetFace: A Large-Scale Dataset and Benchmark for Animal Identification,” *ECCV 2024*, pp. 19–36. [arXiv:2407.13555](https://arxiv.org/abs/2407.13555), [project page](https://dahlian00.github.io/PetFacePage/), [Springer DOI: 10.1007/978-3-031-72649-1_2](https://doi.org/10.1007/978-3-031-72649-1_2).

**[R9]** E. D. R. Neto et al., “Background-invariant re-identification of dogs from camera-trap videos in non-controlled environments,” *Ecological Informatics*, 93, 103547, online 2025 / issue 2026. [DOI: 10.1016/j.ecoinf.2025.103547](https://doi.org/10.1016/j.ecoinf.2025.103547), [code](https://github.com/eugeniodias5/BIFOR), [data page](https://www.lirmm.fr/YT-BB-Dog_Sibetan/).

### 코무늬와 pet biometric

**[R10]** T. E. Coldea, “Use of a dog’s nose print for identification,” *Journal of the American Veterinary Medical Association*, 204(1), 60S, 1994. 정량 benchmark가 아닌 contact-impression 방법 기록으로 인용한다.

**[R11]** G. Jang et al., “Dog Identification Method Based on Muzzle Pattern Image,” *Applied Sciences*, 10(24), 8994, 2020. [DOI: 10.3390/app10248994](https://doi.org/10.3390/app10248994).

**[R12]** H. B. Bae, D. Pak, and S. Lee, “Dog Nose-Print Identification Using Deep Neural Networks,” *IEEE Access*, 9, 49141–49153, 2021. [DOI: 10.1109/ACCESS.2021.3068517](https://doi.org/10.1109/ACCESS.2021.3068517).

**[R13]** Z. Li et al., “Pet Biometric Challenge 2022, first-place technical report,” 2022. [report and code](https://github.com/dashengge/pet-biometrics). 공식 leaderboard test AUC 0.908699.

**[R14]** F. Shen, Z. Wang, Z. Wang, X. Fu, J. Chen, X. Du, and J. Tang, “A Competitive Method for Dog Nose-print Re-identification,” arXiv, 2022. [arXiv:2205.15934](https://arxiv.org/abs/2205.15934). B. Li, Z. Wang, N. Wu, S. Shi, and Q. Ma, “Dog nose print matching with dual global descriptor based on Contrastive Learning,” [arXiv:2206.00580](https://arxiv.org/abs/2206.00580).

### 직접 사용한 기반 모델과 데이터

**[R15]** M. Oquab et al., “DINOv2: Learning Robust Visual Features without Supervision,” *TMLR*, 2024. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193), [OpenReview](https://openreview.net/forum?id=a68SUt6zFt).

**[R16]** N. Ravi et al., “SAM 2: Segment Anything in Images and Videos,” 2024. [arXiv:2408.00714](https://arxiv.org/abs/2408.00714), [official repository](https://github.com/facebookresearch/sam2).

**[R17]** D. Qin et al., “MobileNetV4: Universal Models for the Mobile Ecosystem,” *ECCV 2024*, pp. 78–96. [DOI: 10.1007/978-3-031-73661-2_5](https://doi.org/10.1007/978-3-031-73661-2_5), [arXiv:2404.10518](https://arxiv.org/abs/2404.10518).

**[R18]** J. Deng, J. Guo, N. Xue, and S. Zafeiriou, “ArcFace: Additive Angular Margin Loss for Deep Face Recognition,” *CVPR 2019*. [DOI: 10.1109/CVPR.2019.00482](https://doi.org/10.1109/CVPR.2019.00482).

**[R19]** H. Yu, Y. Xu, J. Zhang, W. Zhao, Z. Guan, and D. Tao, “AP-10K: A Benchmark for Animal Pose Estimation in the Wild,” *NeurIPS Datasets and Benchmarks*, 2021. [arXiv:2108.12617](https://arxiv.org/abs/2108.12617).

**[R20]** G. Martvel, A. Zamansky, G. Pedretti, C. Canori, I. Shimshoni, and A. Bremhorst, “Dog facial landmarks detection and its applications for facial analysis,” *Scientific Reports*, 15, 21886, 2025. [DOI: 10.1038/s41598-025-07040-3](https://doi.org/10.1038/s41598-025-07040-3), [DogFLW data](https://github.com/martvelge/DogFLW).

### 평가 protocol과 누출 방지

**[R21]** V. Cermak, L. Picek, L. Adam, and K. Papafitsoros, “WildlifeDatasets: An Open-Source Toolkit for Animal Re-Identification,” *WACV 2024*. [DOI: 10.1109/WACV57701.2024.00585](https://doi.org/10.1109/WACV57701.2024.00585), [CVF paper](https://openaccess.thecvf.com/content/WACV2024/html/Cermak_WildlifeDatasets_An_Open-Source_Toolkit_for_Animal_Re-Identification_WACV_2024_paper.html).

**[R22]** L. I. Kuncheva, J. L. Garrido-Labrador, I. Ramos-Perez, S. L. Hennessey, and J. J. Rodriguez, “An experiment on animal re-identification from video,” *Ecological Informatics*, 74, 101994, 2023. [DOI: 10.1016/j.ecoinf.2023.101994](https://doi.org/10.1016/j.ecoinf.2023.101994).

**[R23]** W. J. Scheirer, A. de R. Rocha, A. Sapkota, and T. E. Boult, “Toward Open Set Recognition,” *IEEE TPAMI*, 35(7), 1757–1772, 2013. [DOI: 10.1109/TPAMI.2012.256](https://doi.org/10.1109/TPAMI.2012.256).

**[R24]** S. Liao, Z. Mo, J. Zhu, Y. Hu, and S. Z. Li, “Open-set Person Re-identification,” 2014. [arXiv:1408.0872](https://arxiv.org/abs/1408.0872).

**[R25]** E. Real, J. Shlens, S. Mazzocchi, X. Pan, and V. Vanhoucke, “YouTube-BoundingBoxes: A Large High-Precision Human-Annotated Data Set for Object Detection in Video,” *CVPR 2017*. [DOI: 10.1109/CVPR.2017.789](https://doi.org/10.1109/CVPR.2017.789).

## 15. 조사 한계와 인용 원칙

- 출판사 원문, 저자 manuscript, arXiv, 공식 project/data page를 우선했다.
- 원문 full table을 확인하지 못한 MPFNet 등의 수치는 억지로 채우지 않았다.
- 현재 확장된 DogFaceNet 데이터 규모를 2019 논문 당시 결과의 데이터 규모로 소급하지 않았다.
- 논문에 identity-disjoint라고 쓰여 있지 않으면 임의로 identity-disjoint라고 해석하지 않았다.
- AUC, EER, Rank-k, mAP를 서로 변환하지 않았다.
- 숫자 차이를 계산한 표는 protocol 차이를 함께 표시했으며 SOTA 순위표로 사용하면 안 된다.
- 이 문서는 법률 의견이 아니다. 데이터·weights를 공개하거나 상업 사용하기 전에는 최신 약관과 권리자 허가를 다시 확인해야 한다.
