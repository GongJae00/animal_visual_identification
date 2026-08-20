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

단계는 `parsing → identification → representation → enrollment → gallery → search → evaluation`입니다.
등록은 `enrollment/`, 검색은 `search/`입니다. GenID와 ReID는 단계 이름이 아닙니다.
Pet-ReID, MiewID 같은 공급업체 이름은 유지합니다. visualization은 파이프라인 밖에서
관찰하고, prototype은 `export/`만 조합합니다.

현재 구현은 아래 패키지에 있습니다. 대상 트리는 [AGENTS.md](AGENTS.md)입니다.

| 단계 | 현재 패키지 | 역할 |
|---|---|---|
| Parsing | `parsing/` | Frozen detection/segmentation. 배경 shortcut을 줄인 crop/mask. `IdentityEngine`은 호출하지 않음. |
| Identification | `identification/` | Channel embedding. 현재 E2E 기준은 Appearance. Face/Nose는 연구 후보. |
| Representation | `representation/` | Evidence, quality, channel packing. |
| 등록 | `enrollment/` | Crop/vector → gallery K/V. identity는 canonical UUIDv5. |
| Gallery | `gallery/` | Store, schema, migration. |
| 검색 | `search/` | Query/gallery-key/gallery-value는 역할 이름이다. 교집합 채널 weighted cosine. attention 아님. |
| Evaluation | `evaluation/` | 분할, pairing, metric. 알고리즘 패키지는 evaluation을 import하지 않음. |
| Prototype | `prototype/` | 공개 runtime과 ONNX export. |
| Operations | `operations/` | workers, measurement, video. IdentityEngine은 import하지 않음. |

데이터 adapter는 `data/`, canonical ID는 `enrollment/registry/`, split은 `evaluation/splits/`, 스키마는 `shared/contracts/`.
명령의 법은 `<stage>/commands/<verb>.py`입니다.

```bash
uv run python -m parsing.commands.parse --help
uv run python -m identification.commands.train --help
uv run python -m identification.commands.export --help
uv run python -m representation.commands.embed --help
uv run python -m enrollment.commands.enroll --help
uv run python -m gallery.commands.migrate --help
uv run python -m evaluation.commands.evaluate --help
uv run python -m visualization.commands.render --help
uv run python -m prototype.commands.export --help
uv run python -m data.commands.download --help
uv run python -m data.commands.audit --help
uv run python -m operations.commands.measure --help
```

검색 CLI는 없고 `IdentityEngine.search`가 제품입니다. 각 단계 README가 하위 명령을 색인합니다.

연구 diagnostic E2E: parser materialize → Appearance embed →
`evaluation/parsed_body.py`. 완료된 ablation은
[archive/](archive/README.md), 결과 표는
[연구 진행 요약](docs/RESEARCH_PROGRESS.md). 제품 경로는 아래 API입니다.

## 공개 API

공개 import:

```python
from PIL import Image

from prototype.runtime import IdentityEngine, Match

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
uv run pytest tests/prototype/test_public_runtime_contracts.py
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
