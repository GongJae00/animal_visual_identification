# Oracle Verification Evaluation

## Boundary

G3 evaluates manually admitted body/head crops after G0–G2 pass. It estimates
recognizer feasibility without detector, tracker, or quality-selector errors.
It does not establish automated end-to-end performance.

Mode 1 verification is primary:

```text
query tracklet versus expected registered template
```

Each pair must cross sessions. Calibration and final-test manifests are
different content-addressed objects. A frozen threshold records the model,
gallery, modality direction, target FMR, confidence level, and calibration
manifest hash. Test evaluation applies the fixed rule
`same if score >= threshold`; it never searches for a better test threshold.
The test evaluator rehashes the supplied pairing policy against the pair bundle
and rejects any threshold whose modality direction differs from that policy.

## Uncertainty

Every FMR and FNMR report includes event count, trial count, point estimate, and
a two-sided Wilson score interval. A zero empirical error is not zero risk.
For planning a zero-event one-sided exact binomial upper bound is

\[
u=1-(1-c)^{1/n},
\]

where \(c\) is confidence and \(n\) is the number of trials. At 95% confidence,
29,956 zero-error negative trials are needed before the one-sided upper bound is
at most \(10^{-4}\).

Pair-level intervals assume independent events. Repeated dogs, sessions, or
templates violate this assumption. CVI therefore also requires a content-
addressed whole-cluster percentile bootstrap with a predeclared query-dog or
query-session unit, at least 1,000 resamples, a fixed seed, and the same
confidence level as the frozen threshold. Pair multiplication from adjacent
frames does not create independent evidence.

Percentile bootstrap is still weak with few clusters and reproduces zero when
no rare error is observed. It does not replace the exact zero-event upper bound,
cluster-count disclosure, subgroup multiplicity handling, or a hierarchical
analysis when the final design requires one.

## Required G3 report

- oracle crop admission policy and rejected-crop counts;
- positive/negative construction and same-breed hard negatives;
- unique dogs, sessions, templates, and pair counts;
- RGB→RGB and IR→IR first, followed by cross-modal directions;
- ROC/AUC/EER for exploration only;
- FNMR at frozen FMR operating points with event counts and intervals;
- closed-set Rank-1/5/10 separately from verification;
- score distributions by quality, breed/color hard set, camera, cage, and time;
- exact model, checkpoint, gallery, split, and threshold hashes;
- comparison to automated crops only after G4.

No AUC, pair count, or zero-error result alone can pass G3.
