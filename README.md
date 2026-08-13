# Canine Video Identity

강아지의 Appearance, Face, Nose evidence를 연구하고, contract-tested crop-level identity retrieval runtime을 제공하는 프로젝트입니다. 공개 runtime은 영상 전체 시스템이 아니라 호출자가 제공한 crop을 등록하고 local gallery에서 closed-set 후보를 찾는 범위로 제한됩니다.

## 설치와 API 형태

Linux, Python 3.12, [`uv`](https://docs.astral.sh/uv/)가 필요합니다. CPU와 CUDA lane을 함께 설치하지 마십시오.

```bash
uv sync --extra cpu --extra data --extra models --extra training --group dev
# CUDA 환경에서는 --extra cpu 대신 --extra cuda를 사용합니다.
```

```python
from PIL import Image

from canine_identity import IdentityEngine, Match

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

이 예제는 API 형태를 보여줍니다. 저장소에는 실행 가능한 retrieval config나 identity model이 포함되지 않습니다. Channel별 요구 artifact는 [Configuration](docs/CONFIGURATION.md)을 따르며, dataset과 weight는 자동으로 내려받지 않습니다. `cvi.*` 문자열은 기존 artifact bytes와 gallery 호환성을 위해 유지하는 versioned schema identifier이며 Python package 이름이 아닙니다.

## 공개 Runtime 범위

`IdentityEngine`은 다음 동작만 제공합니다.

- canonical UUIDv5 identity에 대한 crop enrollment
- required evidence fail-closed 및 explicit optional evidence
- versioned local gallery persistence
- available-intersection weighted cosine scoring
- identity별 maximum template score와 deterministic ordering

Detection, tracking, temporal aggregation, open-set rejection, service deployment는 공개 runtime에 연결되어 있지 않습니다. 관련 코드는 offline 연구 또는 운영 검증 경로이며 product capability를 의미하지 않습니다.

## 연구 경로

연구 코드는 `contracts/`, `data/`, `identity_governance/`, `localization/`, `identity_methods/`, `representation_learning/`, `evidence_fusion/`, `retrieval/`, `evaluation/`, `experiments/`로 분리되어 있습니다. 실행 정의는 `workflows/`, 주요 tracked config는 각 기능 package와 `experiments/configs/`에 있습니다. Raw data, checkpoint, gallery, cache, run result는 Git 밖에 둡니다.

현재 확인된 trend와 해석 한계는 [연구 진행 요약](docs/RESEARCH_PROGRESS.md), 향후 admission 조건은 [Roadmap](docs/ROADMAP.md)을 참고하십시오.

## 개발 검증

```bash
uv run pytest tests/test_public_runtime_contracts.py
uv run pytest
uv build --out-dir /tmp/canine-identity-dist
git diff --check
```

세부 문서는 [문서 안내](docs/README.md), 구조와 dependency contract는 [Architecture](docs/ARCHITECTURE.md), 설정은 [Configuration](docs/CONFIGURATION.md), 제한은 [Known Limitations](docs/KNOWN_LIMITATIONS.md)을 참고하십시오.

## License

Repository code와 documentation은 Apache-2.0입니다. 외부 dataset, weight, source, image, generated artifact에는 각각의 upstream 조건이 적용됩니다. 자세한 내용은 [Third-Party Licensing](THIRD_PARTY_LICENSES.md)을 참고하십시오.
