# Roadmap

This roadmap defines future evidence gates, not completed work, a schedule, or a performance promise. The public runtime remains crop-level closed-set retrieval until a gate is implemented, artifact-bound, tested, and independently evaluated.

`search/` is the working directory name for 검색. A more precise term may replace the folder later. Do not revert the stage name to ReID or GenID.

## Data Gate

- Keep source archives and extracted data outside Git.
- Require identity/session/source-video disjointness where the protocol claims it.
- Close allocation over exact, near-duplicate, dependency, and unresolved-review components.
- Preserve a future sealed target cohort with no development exposure.

Promotion requires deterministic manifests, authenticated source receipts, duplicate-closure evidence, and zero prohibited cross-partition identity exposure.

## Localization Gate

- Admit a full-frame dog detector and target-association tracker.
- Materialize Face and Nose-region crops in publisher source coordinates.
- Record pose, resolution, uncertainty, and missing evidence without invalidating other branches.
- Select frames by quality and diversity rather than availability alone.

Promotion requires precommitted recall, precision, track-purity, and target-association bounds on an appropriate independent cohort.

## Representation Gates

Appearance candidates must improve the frozen reference across more than one source domain without a material regression or background shortcut failure.

Face candidates must improve identity-disjoint Face evaluation and show reproducible complementary value when fused with Appearance.

Nose-region candidates must pass manual ROI/mask checks, preserve a raw-pixel control, and show positive complementary value under explicit availability.

## Fusion And Temporal Gate

- Fit calibration and fusion only on development/calibration identities.
- Compare A, F, N, pairwise combinations, and A+F+N on fixed panels.
- Stress missing modalities and report branch-specific effective sample counts.
- Compare single-frame, quality-selected, diversity-selected, and multi-prototype aggregation on cross-session tracks.

Promotion requires a frozen policy with a positive uncertainty bound over the strongest available baseline and no hidden evaluation-label selection.

## Open-Set Gate

- Add genuine and impostor probes with independent unknown identities.
- Bind thresholds to model, preprocessing, gallery scale, precision, and score semantics.
- Report DIR/TPIR at FPIR, accepted wrong identity, known rejection, unknown acceptance, and review coverage.

Open-set behavior remains outside `IdentityEngine` until the independent target cohort meets the precommitted assurance requirement.

## Runtime Gate

- Admit only research artifacts that pass model, preprocessing, parity, license, and source-provenance contracts.
- Verify CPU behavior and guarded CUDA behavior without eager optional imports.
- Preserve or explicitly migrate gallery and receipt formats.
- Define service, authentication, privacy, deletion, and audit ownership before claiming deployment support.

See [Research Progress](RESEARCH_PROGRESS.md) for completed research-only observations and [Known Limitations](KNOWN_LIMITATIONS.md) for the current boundary.
