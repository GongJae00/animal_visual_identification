"""Canine Video Identity — 다중 증거 기반 개체 식별 시스템.

┌─ 사용자 API ───────────────────────────────────┐
│  from cvi import CVI, Match                     │
│  cvi = CVI(config)                              │
│  cvi.enroll(img, "뽀삐") → cvi.search(img)      │
└────────────────────────────────────────────────┘

┌─ 도메인 하위 패키지 ───────────────────────────┐
│  cvi.identity/    ID 등록·관리·검색             │
│  cvi.models/      백본 + 손실함수               │
│  cvi.evidence/    특징 추출 (비문·랜드마크·외형)│
│  cvi.search/      탐색·융합·보정·OpenSet        │
│  cvi.evaluation/  평가 지표·ablation            │
│  cvi.breed/       품종 분류                     │
│  cvi.training/    학습 인프라                    │
│  cvi.pipeline/    등록·검색 통합                 │
│  cvi.deployment/  CUDA 배포                     │
│  cvi.detection/   YOLO 개 검출                  │
└────────────────────────────────────────────────┘
"""

from cvi.api import CVI, Match

# ── 연구 인프라 (backward compat) ──
from cvi.acquisition import *  # noqa: F403
from cvi.contracts import *  # noqa: F403
from cvi.capacity import *  # noqa: F403
from cvi.dataset import *  # noqa: F403
from cvi.coverage import *  # noqa: F403
from cvi.identity_registry import *  # noqa: F403
from cvi.trainer import *  # noqa: F403
from cvi.split_registry_binding import *  # noqa: F403
from cvi.inference import *  # noqa: F403
from cvi.model_paths import MODELS_DIR  # noqa: F401
from cvi.leakage import *  # noqa: F403
from cvi.decode import *  # noqa: F403
from cvi.telemetry import *  # noqa: F403
from cvi.evaluation import *  # noqa: F403
from cvi.pairing import *  # noqa: F403
from cvi.scoring import *  # noqa: F403
from cvi.detection import *  # noqa: F403
from cvi.face_aligner import *  # noqa: F403
from cvi.gpu_index import GpuIdentityIndex  # noqa: F401
from cvi.post_search import *  # noqa: F403
from cvi.crop_export import *  # noqa: F403
from cvi.controls import *  # noqa: F403
from cvi.mask_semantics import *  # noqa: F403
from cvi.evidence_extractor import *  # noqa: F403
from cvi.multi_head import *  # noqa: F403
from cvi.identity_index import *  # noqa: F403
from cvi.search_engine import *  # noqa: F403
from cvi.backbones import *  # noqa: F403
from cvi.heads import *  # noqa: F403
from cvi.deployment import *  # noqa: F403
from cvi.utils import cosine_similarity, l2_normalize  # noqa: F401

__all__ = ["CVI", "Match"]
