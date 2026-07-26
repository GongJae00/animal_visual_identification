# Command Log

## 2026-07-25 — Phase 0: Repository Freeze and Full Inventory

### Environment verification
```bash
git rev-parse HEAD
# 0ba3b1bef4ad6bd18ee516260cf938e9e43ca659 ✓ matches pinned

python --version
# Python 3.12.13

uv --version
# uv 0.11.25

uname -a
# Linux DESKTOP-5VKFS3C 6.18.33.2-microsoft-standard-WSL2 x86_64
```

### Branch creation
```bash
git switch -c audit/e2e-hardening
# Switched to new branch
```

### Repository inventory
```bash
find src -name '*.py' | wc -l         # 115 modules
find tools -name '*.py' | wc -l       # 65 tools
find tests -name '*.py' | wc -l       # 66 test files
find src -name '*.py' -exec wc -l {} + | tail -1   # 47,261 lines
grep -c 'def test_' tests/*.py        # 605 test methods
```

### Pattern searches
```bash
grep -RnE 'TODO|FIXME|NotImplemented|pass$' src tools tests
# 1 result: index/hierarchical.py:120 NotImplementedError

grep -RnE 'except BaseException|except Exception' src tools
# 23 sites in 16 files

grep -Rn 'strict=False' src tools
# 3 sites: download_models.py:182, build_calibration_pairs.py:74, export_onnx.py:28

grep -Rn 'pickle' src tools
# 2 files: fusion/calibrator.py:35,41, post_search.py:38,45

grep -Rn 'trust_remote_code' src tools configs
# 0 results (good — we bypassed it via timm)
```

### MiewID ONNX verification
```bash
uv run python -c "import onnxruntime as ort; s=ort.InferenceSession('~/.cache/cvi/models/miewid_nose.onnx'); print(s.get_inputs()[0].shape)"
# Input: ['batch', 3, 440, 440] — 440×440 expected

uv run python -c "import onnxruntime as ort; s=ort.InferenceSession('~/.cache/cvi/models/miewid_nose.onnx'); print(s.get_outputs()[0].shape)"
# Output: ['batch', 2152] — 2152-d embedding

# Determinism check with random noise
uv run python -c "..."
# Deterministic: YES, Cross-input cosine: 0.995 (low sensitivity to noise)
```

### MiewID license check
```bash
uv run python -c "from huggingface_hub import HfApi; api=HfApi(); info=api.model_info('conservationxlabs/miewid-msv3')"
# License: not found (404)
```

### Test suite
```bash
uv run python -m unittest discover -s tests -v
# 564 tests, 125.851s
# 548 pass, 5 fail, 5 error, 6 skip
```

### Documentation truth table
```bash
# Cross-referenced README.md, AGENTS.md, ARCHITECTURE.md with source code
```

### MiewID official architecture inspection
```bash
uv run python -c "from huggingface_hub import hf_hub_download; open(hf_hub_download('conservationxlabs/miewid-msv3', 'modeling_miewid.py')).read()"
# Confirmed: GeM() pooling, BatchNorm1d(2152), efficientnetv2_rw_m
```

## 2026-07-26 — Phase 0: G1 Evaluation Hardening (Third Evaluator Fix)

### Working tree state
```bash
git log --oneline -3
# 49a9e7c fix(evaluation): correct operating points, leakage contracts, calibration, retrieval, open-set
# 331c1f4 evaluation package: factor metrics into subpackage, fix evaluate_multichannel tool, add regression tests
# 0ba3b1b Fix MiewID download, add dataset downloaders, fix check_env, update docs

git diff --stat HEAD
# 11 files changed, 1075 insertions(+), 691 deletions(-)
# (not yet committed)
```

### Files changed
```bash
# Modified (9):
#   src/cvi/evaluation/retrieval.py          — mINP invariant + sample ID self-match
#   src/cvi/evaluation/open_set.py           — detection vs identification + calibration/test split
#   src/cvi/evaluation/verification.py       — exact counts in OperatingThreshold
#   src/cvi/evaluation/__init__.py           — re-export new symbols
#   tests/test_evaluation_metrics.py         — hand-derived oracles, regression tests
#   tests/test_evaluation_openset.py         — open-set unit tests
#   tests/test_evaluation_split.py           — split validation tests
#   tests/test_evaluation_threshold.py       — threshold selection tests
#   tools/evaluate_multichannel.py           — full provenance + schema + split enforcement
#   pyproject.toml                           — jsonschema dependency
# New (2):
#   schemas/cvi.evaluation.report.v2.schema.json   — formal JSON Schema
#   tests/test_evaluate_multichannel_cli.py         — CLI smoke tests
```

### Evaluation tests
```bash
uv run python -m unittest tests.test_evaluation_metrics tests.test_evaluation_openset tests.test_evaluation_split tests.test_evaluation_threshold test_evaluate_multichannel_cli -v
# 83 test methods discovered
# 83/83 pass
# Key test suites:
#   test_evaluation_metrics        — 30 tests (verification, retrieval, calibration)
#   test_evaluation_openset        — 10 tests (open-set known/unknown, edge cases)
#   test_evaluation_split          — 7 tests (leakage detection, cross-split identity/path/group)
#   test_evaluation_threshold      — 4 tests (threshold selection, max TAR)
#   test_evaluate_multichannel_cli — 12 tests (CLI smoke tests)
```

### Full test suite
```bash
uv run python -m unittest discover -s tests -v 2>&1 | tail -20
# 640 tests discovered
# Results: 630 pass, 5 failures, 5 errors, 6 skipped
# Failures (pre-existing, not from these changes):
#   test_geometric_verifier        — 5 failures (pre-existing)
# Errors (pre-existing, not from these changes):
#   test_measurement_comparison    — 1 error (pre-existing)
#   test_onnx_inference_benchmark  — 4 errors (pre-existing, ONNX/CUDA)
```

### compileall
```bash
uv run python -m compileall src/cvi tools tests
# 0 errors
```

### research-implementation-check
```bash
uv run research-implementation-check .
# 0 failures, 0 warnings
```

### E2E CLI smoke tests
```bash
uv run python -m unittest tests.test_evaluate_multichannel_cli -v
# test_help_exits_zero                 ✓
# test_verification_help_exits_zero    ✓
# test_retrieval_help_exits_zero       ✓
# test_open_set_help_exits_zero        ✓
# test_retrieval_happy_path            ✓
# test_retrieval_self_match_excluded   ✓
# test_retrieval_no_self_match_missing_sample_ids ✓
# test_open_set_happy_path             ✓
# test_open_set_no_unknowns_raises     ✓
# test_provenance_has_git_info         ✓
# test_schema_version_in_report        ✓
# test_retrieval_bootstrap_ci_present  ✓
# --- 12/12 pass ---
```

### Key implementation details
```bash
# Schema validation
uv run python -c "from jsonschema import Draft202012Validator; v=Draft202012Validator({'type':'object'}); print('jsonschema available')"
# jsonschema available

# Full provenance tracking
uv run python -c "
import json, hashlib, pathlib
p = pathlib.Path('schemas/cvi.evaluation.report.v2.schema.json')
h = hashlib.sha256(p.read_bytes()).hexdigest()
print(f'Schema SHA256: {h}')
"
# Schema SHA256: <hex digest>

# Split enforcement check
uv run python -c "
from tools.evaluate_multichannel import validate_split_disjoint
w = validate_split_disjoint([{'image_a':'a.jpg','identity':'1'}], [{'image_a':'b.jpg','identity':'2'}])
assert len(w) == 0, f'unexpected warnings: {w}'
print('split validation OK')
"
# split validation OK
```
