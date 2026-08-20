# Version History

Current behavior is defined by `docs/ARCHITECTURE.md`,
`docs/CONFIGURATION.md`, and `docs/KNOWN_LIMITATIONS.md`. This file records why
versioned changes occurred, where their evidence lives, and which persisted
contracts remain readable.

## Unreleased

### Compact follow-up

- Renamed `IdentityRetrievalPipeline` to `SearchPipeline`.
- Parsing CLI no longer dispatches evaluation/archive/data jobs. Those live on
  `evaluation.commands.evaluate` and `data.commands.audit`.
- Registry build/bind and split-manifest check moved to
  `evaluation.commands.evaluate registry-build|registry-bind|split-check`.
  `enrollment.commands.enroll` only augments labels.
- Evaluation library modules are no longer `python -m` entrypoints; use
  `evaluation.commands.evaluate`.
- Moved pairing-score tests from `tests/search/` to
  `tests/evaluation/test_pair_scoring.py`.
- Root `AGENTS.md` is the only agent law. Removed unused example JSON and
  restating comments.
- Replaced persisted `cvi.*` and `canine_identity.*` schema identifiers with
  capability-owned names (`gallery.manifest.v5`, `search.config.v2`,
  `enrollment.registry_manifest.v1`, `pdq.fingerprint.v1`,
  `source.provenance.v3`). Old identifier strings are not accepted.


- Dataset files resolve from `CANINE_IDENTITY_DATASETS_DIR` when set, otherwise
  `$CANINE_IDENTITY_DATA_DIR/datasets`. Checkpoints, receipts, caches, and
  experiment state stay under `CANINE_IDENTITY_DATA_DIR`.
- Split tests by stage under `tests/<stage>/`. Completed-comparison tests live
  under `tests/archive/{full128,appearance_face_nose,nose_metric,nose,face,shared_helpers}/`.
  Cross-cutting scans stay at `tests/test_dependency_boundaries.py` and
  `tests/test_packaging_import_surfaces.py`. Tests locate the repository through
  `tests/repo_root.py`.
- Renamed `data/public/` to `data/public_sources/`.
- Packaged `evaluation/verification/` (`metrics.py` plus public re-exports).
- Moved independently runnable command siblings out of `commands/` into owner
  packages. Stage verbs remain `parse`, `train`/`export`, `embed`, `enroll`,
  `migrate`, `evaluate`, `render`, `download`/`audit`, and `measure`. Evaluation
  still keeps `compare_score_drift` and `create_batch_invariance_precommitment`
  next to `evaluate.py` because those CLIs import `operations`.

### Structure Overhaul

Version `0.6.0`. Public import is `from prototype.runtime import IdentityEngine, Match`.

- Recorded the capability-noun tree as contributor law in `AGENTS.md`. Pipeline
  stage names are Parsing, Identification, Representation, 등록
  (`enrollment/`), gallery, 검색 (`search/`), and Evaluation. GenID and ReID
  are not stage names. Vendor names (Pet-ReID, MiewID) stay.
- Public import is `from prototype.runtime import IdentityEngine, Match`.
  Persisted `cvi.*` identifiers, public field `dog_id`, env
  `CANINE_IDENTITY_DATA_DIR`, `IdentityEngine` behavior, and root
  `LiteratureReview.md` are unchanged.
- Commands live at `<stage>/commands/<verb>.py`. Do not add
  `workflows/`, `utils/`, `misc/`, `helpers/`, or `common/` dumps.
- Nested `foundation/` and `contracts/` under `shared/`. Imports are
  `shared.foundation` and `shared.contracts`. Persisted `cvi.*`
  identifiers are unchanged. Physical path moves create new source
  provenance.
- Split `parsing/` into `training/` and `export/` (detect → segment →
  region → quality → crop). Commands live at
  `python -m parsing.commands.parse`. Identity embedding trainers stay
  outside parsing.
- Split `embedding/` into `identification/` and `representation/`.
  Appearance/Face/Nose live under identification `training/` vs `export/`.
  Evidence and channel packing live under `representation/`. Full128 moved
  to `archive/full128/`. PDQ/pHash moved to `data/audit/`. The `embedding/`
  package is gone.
- Split `retrieval/` and `identity/` into `enrollment/`, `gallery/`, and
  `search/`. Identity-disjoint splits, exposure, admission, leakage, face
  governance, and research kfold live under `evaluation/splits/`. Scoring
  lives in `search/scoring/roles.py` (query / gallery-key / gallery-value
  roles, not attention). Commands:
  `python -m enrollment.commands.enroll` and
  `python -m gallery.commands.migrate`. Search has no extra CLI.
  `identity/` and `retrieval/` are gone.
- Moved `runtime/` to `prototype/runtime/` and `systems/` to
  `prototype/export/` (ONNX backends) plus `operations/{workers,measurement,video}`.
  Commands: `python -m prototype.commands.export` and
  `python -m operations.commands.measure` (`onnx`, `probe`, `decode`,
  `capacity`). `runtime/` and `systems/` are gone. `pyproject.toml` is `0.6.0`.
- Moved `evaluation/retrieval.py` to `evaluation/search_metrics/metrics.py`.
  Commands live at `python -m evaluation.commands.evaluate`. Completed
  comparison sets live under `archive/{full128,appearance_face_nose,nose_metric,nose,face,shared_helpers}`
  with `commands/` not `workflows/`. `legacy/` is gone.
- Visualization writes `Visualization/vis/00_parsing` … `05_search` via
  `python -m visualization.commands.render --stage`. Paper
  `FIGURE_REGISTRY` 00–17 stays a separate sequence (`--paper` writes
  `Visualization/paper/`). Leftover `workflows/` CLIs moved into stage
  `commands/` (or archive PDQ worker/regression). `workflows/` is gone.
- Sweep leftover imports and current-path docs. `embedding`, `identity`,
  `retrieval`, `runtime`, `systems`, `workflows`, and `legacy` are gone.
  Historical provenance families (`embedding.methods.face.*` logical
  component, PDQ `workflows.build_native_pdq_worker` entrypoints,
  `foundation/` path inventories) stay readable. Vendor Pet-ReID/MiewID
  and provisional GenID identity records stay. `dist/` old
  `canine_video_identity` wheels remain gitignored.

### Compactness

- Moved environment bootstrap from `scripts/check_env.sh` to
  `setup/check_env.sh`. `setup/` owns first-user environment and release
  guidance; `pyproject.toml` and `uv.lock` remain at the repository root.
- Documented the research pipeline (Parsing, Identification, GenID, ReID,
  Evaluation) as a reading order over the existing packages. Import paths are
  unchanged.
- Added `workflows/README.md` as the stage index. Command paths remain
  `workflows/<command>.py`.
- Removed untested checkout CLIs that were not on the Parsing → Appearance →
  GenID/ReID path and had no test, document, or receipt command:
  `run_foundation_a_panel`, `generate_dinov2_region_candidates`,
  `render_dinov2_region_qa`, `calibrate_yt_masked_multievidence`,
  `summarize_yt_masked_multievidence`, `evaluate_noseid_oracle`,
  `mine_nose_hard_negatives`, `train_masked_afn_kfold`,
  `evaluate_canid_roi_manifest`, `export_canid_roi_manifest`,
  `intake_threatexchange_pdq_regression`, `audit_yt_nose_signal_quality`,
  and the v1 `build_face_identity_protocol` wrapper. Package APIs and
  versioned schemas remain. Removed unlinked `docs/FEASIBILITY_GATES.md`;
  admission gates stay in `docs/ROADMAP.md`.
- Moved completed ablation experiments, their CLIs, and historical protocol
  notes to `legacy/version/`. Functional checkout commands remain
  `workflows/<command>.py`. Ablation outcomes are tabulated in
  `docs/RESEARCH_PROGRESS.md`. Ablation code is grouped by set under
  `legacy/version/{full128,afn,n4,nose,face}` with one README each.
  Transplanted protocol essays under `legacy/version/docs/` were removed.
- `workflows/evaluate_multichannel.py` now fails closed on git and input-file
  hashing instead of writing `__GIT_FAILED__` / `UNVERIFIED` placeholders.
- Reduced live checkout CLIs from 84 to 54. Deleted untested wrappers
  (`prepare_ap10k_dog_yolo_pose`, `summarize_evidence_coverage`,
  `audit_timestamps`, `evaluate_capacity`, `convert_to_evidence_observations`,
  `audit_public_dataset_archive`, `audit_nested_public_dataset_archive`,
  `audit_public_canine_pdq`). Absorbed sibling CLIs into tested filenames:
  segmentation manifests → `build_animal_parsing_runtime_manifest`;
  embedding precommit/verify → `produce_embedding_cache`;
  batch-invariance evaluate → `create_batch_invariance_precommitment`;
  acquisition/check/camera → `download_datasets`;
  bind/check-split → `build_identity_registry`;
  source-admissions → `build_research_cycle_manifest`;
  score-drift precommit/plan/verify → `compare_score_drift`;
  visual plan/execute/score → `evaluate_visual_controls`;
  parser summarize/render → `compare_parser_materializations`.
  Package APIs remain. Protected prepare/verify stay separate because
  `evaluate_multichannel protected` is not the same `--help` contract.
- Removed unpinned legacy CLIs:
  `analyze_full128_successors`, `build_full128_successor_evaluation_panel`,
  `build_full128_successor_inventory`, `materialize_sibetan_multievidence`,
  `prepare_nose_region_crops`, `train_nose_identity`. Receipt-bound and
  test-imported set CLIs remain.
- Deleted unused nose helpers `embedding/methods/nose/training/config.py`
  and `hard_negative.py`. Live Face/Nose/Full128 libraries were not moved.
- Classified remaining `except Exception` / `except BaseException` sites.
  Cleanup-and-re-raise, typed wrap, quality fail-closed, and destructor
  swallows stay. One remask in public ZIP extraction now chains the cause.
- `data/public/public_dataset_extraction.py` now chains the original
  `_validate_portable_component` error when a relative path is Windows-ambiguous
  instead of replacing it with an unchained `ValueError`.
- Destaphettied live packages in place (no file moves): unused private
  helpers and same-file duplicates in `parsing`, `embedding`, `evaluation`,
  `identity`, `data`, `systems`, `contracts`, `retrieval`, and
  `visualization`. `legacy/` and one-off checkout CLIs were left out.
  Public exports, schemas, and `cvi.*` identifiers are unchanged.
- Removed unused `runtime/configs/` example JSON that was not
  loaded by `IdentityEngine` or tests. Retrieval config v2 remains a
  caller-supplied object. Deduplicated `foundation` file-stat identity
  used by protected I/O.
- Removed leftover unused live helpers after CLI reduction: unused
  data-root aliases, unused timestamp/PDQ/quality/telemetry wrappers,
  unused parsing types, and unused face-binding verification. Foundation
  JSON writers now share `json_document_bytes`.

### Architecture

- Renamed the public runtime package from `canine_identity` to `runtime`.
  Import `runtime.IdentityEngine`. Persisted `canine_identity.*` schema
  identifiers, `CANINE_IDENTITY_DATA_DIR`, and gallery contracts are
  unchanged.
- Renamed the distribution and repository to
  `animal-visual-identification` / `animal_visual_identification`.
  Historical GitHub path `canine_video_identity` remains readable as a
  redirect if the remote is renamed.
- Renamed the internal research-figure package from `vis` to `visualization`.
  The supported public runtime remains `runtime.IdentityEngine`; persisted
  `cvi.vis.*` renderer and style identifiers remain unchanged.
- Renamed the internal artifact-schema package from `artifact_contracts` to
  `contracts`. Persisted `cvi.*` identifiers and artifact-contract field names
  remain unchanged.
- Renamed the internal retrieval package from `identity_retrieval` to `retrieval`.
  Gallery v3-v5 readers and persisted `cvi.gallery_*` identifiers remain unchanged.
- Renamed the internal data-processing package from `data_pipeline` to `data`.
  Historical content-bound source paths remain unchanged in archived receipts;
  their v2 readers reject execution against a different current source inventory.
- Renamed the internal detection and region-processing package from `localization`
  to `parsing`. Archived `localization/...` source inventories remain readable;
  executable runtimes and caches still require exact current-checkout bindings.
- Consolidated internal identity methods, representation learning, and evidence
  handling under `embedding/{methods,learning,evidence}`. Persisted `cvi.*`
  identifiers remain unchanged; new source provenance records the new paths.
- Reorganized identity governance under responsibility-specific `identity/`
  subpackages. UUID namespaces and persisted identity contracts remain unchanged.
- Reorganized guarded execution under `systems/{inference,workers,measurement}`.
  Exact historical worker commands and bootstrap hashes remain readable, while
  current execution uses only the `systems.*` modules.
- Nested pretrained intake under `contracts/intake`, moved tracklet split and
  duplicate-adjudication policy under `identity/splits`, and moved decode and
  capacity measurement under `systems/measurement`.
- Moved landmark and Nose representation-learning code out of `parsing` into
  `embedding/methods`, and moved supervised localization metrics and K-fold
  protocols into `evaluation`. Complete historical source-path inventories remain
  readable where persisted readers require them; mixed generations fail closed.
- Moved Full128 source route planning to `data/full_segment` and persisted PDQ
  contracts to `contracts`. Duplicate adjudication now consumes neutral evidence
  contracts instead of embedding implementations, and the dependency tests reject
  all top-level internal package cycles. Persisted `cvi.*` identifiers are unchanged.

### Runtime And Resource Use

- Reduced exact gallery-search allocation without changing scorer arithmetic,
  identity aggregation, or tie ordering. Identity ordinals now use compact shared
  storage, unrestricted searches avoid an eligibility mask, required-only searches
  avoid a denominator array, and bounded top-k selection replaces a full identity
  sort.
- Reduced duplicate-adjudication peak memory by hashing candidate sets in bounded
  blocks and by computing immutable ledger hashes once per graph assembly.
- Removed duplicate bulk-enrollment normalization, released validated ONNX graph
  objects before session construction, lazily imported unconfigured evidence
  implementations, and included `parsing` in protected embedding-worker snapshots.
- Tightened public configuration and metadata validation so boolean fusion weights
  and reserved metadata fail before model inference. Gallery bytes, persisted
  schema identifiers, exact scores, and deterministic result ordering are unchanged.

### Part 1 - Parser Policy And Full128 Materialization

#### Parser policy v6 and Full128 route policy v3 - 2026-08-13

- Reason: the v5 parser requested dog and cat candidates and the generic route
  policy rejected useful single-dog observations when any additional candidate
  was present. Auxiliary datasets also lacked a deterministic single-dog
  selection rule.
- Change: parser policy v6 requests dogs only. AP-10K keeps authoritative global
  bbox association; DogFLW and Oxford select the largest valid dog by foreground
  pixels; identity-bearing datasets require exactly one post-suppression dog.
  Full128 route-plan v3 binds those rules and records candidate counts, selection,
  authority, and terminal reasons in lineage.
- Evidence: Oxford dog evaluation report digest
  `9a7cf4db1e32dd247797f06c29bb53f2908b2ba1c520f58d1fb402fb5faa97df`;
  AP-10K panel digest
  `360a07e976418b72bf2c9092a975616cbf514a635cf5aac7a6dfcb8ebf70cab6`;
  runtime manifest digest
  `21ca43fe71cbf7bcdd0f13bb39dea4045b029e4b8beebbd2c58de800a299363a`;
  Full128 route-plan digest
  `4bed7c8d2689307bbd2018d9400d79c352c018ffd298e24ca725eceb0fc522f8`;
  assembly digest
  `714c429f97888d7072e8951a7179384f778adcd2b3664c5809408a66b0b8f162`.
  These artifacts are external and research-only.
- Outcome: 36,195 of 49,253 observations materialized, 518 more than v5, with
  no v5-success-to-v6-terminal regressions. The comparison mixes parser policy,
  runtime source closure, and route-policy versions; it is not a biometric result.
- Compatibility: parser policies v4/v5, parser runtime manifest/bundle v1, and
  Full128 route-plan v2 remain accepted
  because completed external artifacts bind those schemas. They are compatibility
  readers, not current defaults.

#### Parser batch protocol - 2026-08-13

- Reason: batch sizes 8 and 16 offered modest throughput gains but had not been
  checked for output equivalence.
- Change: a fixed 512-image benchmark records one warm-up and three measured runs,
  parser/end-to-end timing, peak CUDA memory, exact and semantic fingerprints, and
  terminal-decision mismatches.
- Evidence: benchmark digest
  `3b75f00eff209587294e4348c84f9b4723634a536d39e8ec156ebca87f31851c`.
- Outcome: batch 4 remains canonical because batches 8 and 16 changed semantic
  predictions and terminal decisions despite higher throughput.

### Part 2 - Persisted Contracts

- Gallery v4 and v5 remain readable; gallery v3 requires explicit migration.
- Full128 route-plan v2 remains readable beside current v3.
- Parser policies v4/v5 and runtime manifest/bundle v1 remain readable beside
  current dog-only policy v6 and runtime v2.
- Successor private evaluation v1 remains readable; new reports use compact v2
  traces that omit repeated Q/K vectors and bind them through the embedding cache.

### Part 3 - Research Decisions

- Full128 governance-v2 selected B5-SPATIAL by DEV point estimates in three seeds,
  but its precommitted paired interval gate returned `NO_GO`. B3 remains the stable
  research candidate. Evidence digests are listed in
  `docs/RESEARCH_PROGRESS.md`.

## 0.4.0 - 2026-08-03

### Breaking

- Replaced the `cvi.CVI` import surface with
  `runtime.IdentityEngine`; `Match` remains the public result type.
- Reorganized the former `src/cvi` tree into functional top-level packages for
  data, identity governance, localization, identity methods, learning, fusion,
  retrieval, evaluation, runtime, and operations.
- Moved source-checkout commands from `tools/` to `workflows/` and removed
  unsupported compatibility exports, deployment facades, and duplicate runtime
  scaffolding.

### Architecture And Performance

- Added recursive logical source provenance v2 while retaining narrow readers
  for persisted Face and PDQ v1 evidence.
- Added dependency-boundary tests that keep evaluation and operations out of the
  public runtime and algorithm packages.
- Removed duplicate A4 frozen-backbone inference, decode-only label collection,
  eager gallery metadata copies, and an unnecessary retrieval score-matrix copy.
- Added a single-distribution build containing all functional packages and the
  versioned evaluation schemas.

### Validation

- Reproduced the full 15-epoch A4 training trajectory, selected epoch, and DEV
  metrics after the migration.
- Reproduced all eight DogFaceNet, MPDD, and SiBeTan external protocol results;
  the new report differs only by the newly generated checkpoint hash.
- Verified the complete test suite, wheel contents, isolated package import,
  documentation links, lockfile, and source diff.

### Research Infrastructure

- Added strict AP-10K and DogFLW adapters, content-bound localization caches,
  instance-aware ROI manifests, and research-only face ReID tooling.
- Added a content-addressed model catalog with logical role aliases; filesystem
  paths no longer imply research or deployment admission.
- Added a separate provisional generated-identity namespace with explicit
  merge, supersede, and rejection states.

### Security And Compliance

- Bound generic ONNX model and preprocessing manifests to gallery contracts.
- Made ROI crop paths, bytes, dimensions, identity propagation, and landmark
  geometry fail closed.
- Kept Ultralytics outside package extras; localization adapters require a
  separately managed research environment.
- Removed host-specific experiment scripts, generated dataset visualizations,
  and stale internal architecture audit material from the public surface.

## 0.3.0 - 2026-07-27

### Breaking

- Removed legacy index implementations from the supported public runtime.
- Made runtime configuration and `index_dir` explicit instead of relying on
  implicit storage or compatibility defaults.
- Advanced persisted galleries to gallery schema v4; incompatible galleries
  must be migrated explicitly rather than accepted silently.
- Added the versioned evaluation schema as a packaged `cvi.schemas` resource.

### Security And Compliance

- Disabled model exporters whose artifacts or licenses were not admitted to a
  supported runtime contract.
- Removed direct Torch Hub execution; public and research evaluators now require
  caller-supplied or receipt-bound local model artifacts.
- Removed Ultralytics from the published training extra. Detection remains
  outside the canonical product, and user-supplied detectors are separately
  licensed.
- Changed manual or license-gated dataset selectors to fail with acquisition
  guidance instead of reporting an empty directory as a successful download.
- Disabled the unpinned DogFaceNet Hugging Face materializer. All dataset
  selectors now require manual acquisition and fail before network or framework
  imports; the default `all` selector is an explicit successful no-op.

### Claims

- This release makes no biometric accuracy, identification performance, or
  deployment-readiness claim. Unit and synthetic tests validate software
  contracts, not biometric performance.
- Removed unverified dataset cardinality and operational-safety wording from
  source documentation.
