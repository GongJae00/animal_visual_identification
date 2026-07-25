"""개체 식별자 등록/관리/검색 — ID 생애주기 전체를 담당합니다.

역할:
- ID 생성    : registered_dog_id, identity_token, sequence_token
- ID 등록    : FAISS 벡터 인덱스 + JSON 메타데이터
- ID 검색    : cosine similarity 기반 Top-K
- ID 삭제    : 인덱스에서 제거
- 증거 분해  : 검색 결과에서 채널별 기여도 추출

사용:
    from cvi.identity import IdentityIndex, GpuIdentityIndex, SpeciesFilteredIndex
"""

from cvi.identity_index import (
    EVIDENCE_SLICES,
    EMBEDDING_DIM,
    EvidenceBreakdown,
    IdentityIndex,
    IndexedIdentity,
    SearchResult,
    make_evidence_slices,
)
from cvi.gpu_index import GpuIdentityIndex
from cvi.index.hierarchical import SpeciesFilteredIndex

__all__ = [
    "EVIDENCE_SLICES",
    "EMBEDDING_DIM",
    "EvidenceBreakdown",
    "IdentityIndex",
    "GpuIdentityIndex",
    "SpeciesFilteredIndex",
    "IndexedIdentity",
    "SearchResult",
    "make_evidence_slices",
]
