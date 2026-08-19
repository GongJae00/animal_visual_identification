# Animal Visual Identification

강아지 Appearance / Face / Nose evidence를 연구하고, crop-level closed-set retrieval
runtime을 제공합니다. 공개 runtime은 영상을 풀지 않습니다. 호출자가 준 crop을
등록하고 local gallery에서 후보를 찾습니다.

## 설치

Linux, Python 3.12, [`uv`](https://docs.astral.sh/uv/). 안내는 [setup/](setup/README.md).

```bash
./setup/check_env.sh cpu
# CUDA: ./setup/check_env.sh cuda
```

`cpu`와 `cuda` extra를 함께 설치하지 마십시오. 데이터와 가중치는 Git에 없고
자동으로 내려받지 않습니다.

## 파이프라인

연구 단계는 아래 패키지에 대응합니다. import 경로를 단계 이름에 맞춰 바꾸지 않습니다.

| 단계 | 패키지 | 역할 |
|---|---|---|
| Parsing | `parsing/` | Frozen detection/segmentation. 배경 shortcut을 줄인 crop/mask. `IdentityEngine`은 호출하지 않음. |
| Identification | `embedding/` | Channel embedding. 현재 E2E 기준은 Appearance. Face/Nose는 연구 후보. |
| GenID | `retrieval/` enroll | Crop → representation → gallery K/V. identity는 UUIDv5. |
| ReID | `retrieval/` search | QKV는 역할 이름이다. 교집합 채널 weighted cosine. attention 아님. |
| Evaluation | `evaluation/` | 분할, pairing, metric. 알고리즘 패키지는 evaluation을 import하지 않음. |

데이터 adapter는 `data/`, canonical ID와 split은 `identity/`, 스키마는 `contracts/`.
명령은 `workflows/<command>.py`이며 색인은 [workflows/README.md](workflows/README.md).

연구 diagnostic E2E: parser materialize → Appearance embed →
`workflows/evaluate_parsed_body_reid.py`. 완료된 ablation은
[legacy/version/](legacy/version/README.md), 결과 표는
[연구 진행 요약](docs/RESEARCH_PROGRESS.md). 제품 경로는 아래 API입니다.

## 공개 API

```python
from PIL import Image

from runtime import IdentityEngine, Match

engine = IdentityEngine("/etc/canine-identity/retrieval.json")

registered_dog_id = "877d96de-ba43-542d-9523-5c20213bfc09"
engine.enroll(
    Image.open("enrollment_crop.jpg").convert("RGB"),
    registered_dog_id,
)
matches: list[Match] = engine.search(
    Image.open("query_crop.jpg").convert("RGB"),
    top_k=5,
)
engine.close()
```

저장소에는 실행 가능한 retrieval config나 identity model이 없습니다. Channel
artifact는 [Configuration](docs/CONFIGURATION.md)을 따릅니다. `cvi.*`는
versioned schema identifier이며 Python 패키지 이름이 아닙니다.

`IdentityEngine`이 하는 일: UUIDv5 enroll, required evidence fail-closed,
explicit optional evidence, local gallery, available-intersection weighted
cosine, identity별 max template과 결정적 정렬.

하지 않는 일: detection, tracking, temporal aggregation, open-set rejection,
service deployment.

## 검증

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
git diff --check
```

구조는 [Architecture](docs/ARCHITECTURE.md), 제한은
[Known Limitations](docs/KNOWN_LIMITATIONS.md), 연구 trend는
[연구 진행 요약](docs/RESEARCH_PROGRESS.md).

## License

Repository code와 documentation은 Apache-2.0입니다. 외부 dataset, weight,
source, image, generated artifact는 각각의 upstream 조건을 따릅니다.
[Third-Party Licensing](THIRD_PARTY_LICENSES.md).
