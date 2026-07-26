# ADR-001: Canonical Runtime Stack Selection

## Status
ACCEPTED — canonical runtime selected for pre-experiment hardening.

## Context
The repository has two competing implementation stacks:
1. **New package stack**: `cvi/api.py` → `cvi/pipeline/` → `cvi/evidence/` + `cvi/fusion/` + `cvi/index/` + `cvi/deployment/`
2. **Legacy flat stack**: `cvi/search_engine.py` + `cvi/evidence_extractor.py` + `cvi/identity_index.py` + `cvi/gpu_index.py`

Both stacks can perform enrollment and search, but they use different APIs, different index implementations, and different evidence abstractions.

## Alternatives Considered

### Option A: New Package Stack as Canonical
- `CVI` class (api.py) as the single public entry point
- All internal modules use `cvi/evidence/`, `cvi/fusion/`, `cvi/index/`, `cvi/pipeline/`
- Legacy flat modules get deprecation wrappers

### Option B: Legacy Flat Stack as Canonical
- `SearchEngine` + `EvidenceExtractorRegistry` as the single runtime path
- New package modules get deprecated
- Higher risk of architectural debt

### Option C: Hybrid — New Package Canonical, Legacy Adapter
- `cvi/pipeline/` as canonical runtime
- Legacy flat modules backward-compatible via thin wrappers
- Most practical transition path

## Decision
**Option A + C (New Package Canonical with Legacy Adapter)**

### Rationale
1. `cvi/pipeline/` and `cvi/evidence/` have cleaner separation of concerns (channel contract via AbstractEvidencer)
2. `SpeciesFilteredIndex` in `cvi/index/hierarchical.py` is the most recent index implementation with breed filtering
3. `CVI` class is the intended public API (documented as such in __init__.py)
4. Legacy `search_engine.py` has entangled concerns (extraction + index + fusion)
5. The new package is already wired into `CVI` — just needs bugfixes, not a rewrite

### Concrete Selections

| Concept | Canonical Implementation | Legacy Alternative | Deprecation |
|---------|------------------------|-------------------|-------------|
| Public API | `cvi.api.CVI` | `search_engine.SearchEngine` | Add DeprecationWarning to SearchEngine |
| Evidence extraction | `cvi.evidence.AbstractEvidencer` | `evidence_extractor.EvidenceExtractor` | Adapter wrapping AbstractEvidencer |
| Fusion | `cvi.fusion.fuser.LearnedWeightFuser` | `SearchEngine` hardcoded weights | Remove from SearchEngine |
| Open-set | `cvi.fusion.open_set.EvidentialOpenSet` | None in legacy stack | N/A |
| Temporal | `cvi.fusion.temporal.TemporalAggregator` | None in legacy stack | Wire into CVI API |
| Index | `cvi.index.hierarchical.SpeciesFilteredIndex` | `identity_index.IdentityIndex`, `gpu_index.GpuIdentityIndex` | Deprecate IdentityIndex, remove duplicate gpu_index |
| Deployment CPU | `cvi.deployment.cpu.CVIDeploymentCPU` | None | Add thin wrapper ADR |
| Deployment CUDA | `cvi.deployment.cuda.CVIDeploymentCUDA` | None | CVIDeploymentCUDA already wraps new stack |
| Trainer | `cvi.trainer.ArcFaceModel` | None | Needs encode() fix (P0-015) |
| Evaluation | `cvi/evaluation/` + `tools/evaluate_multichannel.py` | Canonical | OSCR remains deferred |

### Migration Strategy
1. Fix all P0 findings on new stack first
2. Add deprecation warnings to legacy classes
3. During Sprint 0, remove duplicate `identity/gpu_index.py`
4. Remove stale `identity/__init__.py.bak` from git
5. After G1 (P0 correctness), migrate tools to use new stack
6. After G2 (single-channel baseline), consider removing legacy flat modules

### Compatibility Adapter Requirements
- `SearchEngine.__init__` should accept same args and delegate to CVI
- `EvidenceExtractorRegistry` should wrap `AbstractEvidencer` instances
- `IdentityIndex` should delegate to `SpeciesFilteredIndex`
- Maintain backward-compatible method signatures for 1 release cycle

### Removal Conditions
All legacy flat modules can be removed when:
1. No tool or test imports them directly
2. No user code references them in docs or examples
3. Adapter warnings have been present for 1 release cycle
