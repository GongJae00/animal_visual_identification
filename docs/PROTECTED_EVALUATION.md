# Protected Evaluation Receipt Chain

The protected evaluator is a separate closed-set retrieval path. It does not
upgrade reports from the legacy verification, retrieval, or open-set commands.
Those v2 reports remain `UNVERIFIED`; only the `protected` command can emit the
strict `cvi.evaluation.report.v3` format.

## Trust boundary

The chain requires out-of-band hashes at three boundaries:

1. Preparation requires the raw SHA-256 of `external_pins.json`. That pinned
   file contains raw-byte hashes for the policy, protected split assignment and
   receipt, actual role-exposure ledger and receipt, and both embedding
   manifests. It also pins each embedding-production receipt.
2. Evaluation requires the externally archived plan-receipt hash and advanced
   exposure-declaration hash. A hash copied only from the preparation directory
   is not an external anchor.
3. Verification requires an externally archived output-receipt hash in
   addition to the two preparation anchors.

Hashes stored beside an artifact provide content addressing, not independent
immutability. Archive the three external values in the private research-state
layer before proceeding to the next phase.

## Input contract

Protected scoring consumes `cvi.protected_embedding_manifest.v1` JSON only. It
never instantiates an evidencer or performs image/model inference. Every record
has exact keys for `sample_token`, `identity_token`, `public_subject_token`,
`template_token`, and a finite embedding vector. Manifests are sorted by sample
token and bind an external embedding-production receipt.

The preparation command verifies that:

- the split receipt binds the exact canonical assignment payload;
- each sample and identity has exactly the policy-selected split role;
- gallery and query samples/templates are disjoint and every query identity is
  enrolled;
- the actual exposure ledger and receipt agree;
- every final sample, identity, and public subject has only declared
  `BYTES_EXPORTED` history;
- the transition to `FINAL_TEST_SCORED` is valid;
- byte, JSON structure, sample, dimension, total-value, and score-matrix caps
  pass before dense arrays are allocated.

Preparation atomically publishes `policy_receipt.json`, `input_receipt.json`,
`advanced_exposure_declaration.json`, and `plan_receipt.json` in one new
directory. The final-score exposure is therefore durable before scoring starts.

## Commands

Create `external_pins.json` in an independently controlled step, record its raw
SHA-256, then run:

```bash
uv run python workflows/prepare_protected_evaluation.py \
  --policy experiments/configs/contracts/protected_evaluation_policy.example.json \
  --external-pins /secure/external_pins.json \
  --expected-external-pins-raw-sha256 "$PINS_RAW_SHA256" \
  --split-assignment /secure/split_assignment.json \
  --split-receipt /secure/split_receipt.json \
  --exposure-ledger /secure/exposure_ledger.json \
  --exposure-receipt /secure/exposure_receipt.json \
  --gallery /secure/gallery_embeddings.json \
  --queries /secure/query_embeddings.json \
  --output-directory /secure/evaluation_preparation
```

Archive the printed plan and exposure hashes before scoring:

```bash
uv run python workflows/evaluate_multichannel.py protected \
  --preparation-directory /secure/evaluation_preparation \
  --expected-plan-receipt-sha256 "$PLAN_SHA256" \
  --expected-advanced-exposure-declaration-sha256 "$EXPOSURE_SHA256" \
  --policy /secure/policy.json \
  --split-assignment /secure/split_assignment.json \
  --split-receipt /secure/split_receipt.json \
  --exposure-ledger /secure/exposure_ledger.json \
  --exposure-receipt /secure/exposure_receipt.json \
  --gallery /secure/gallery_embeddings.json \
  --queries /secure/query_embeddings.json \
  --output-directory /secure/evaluation_output
```

The output directory is atomically published with `report.json` and
`output_receipt.json`. Archive the printed output-receipt hash, then verify:

```bash
uv run python workflows/verify_protected_evaluation.py \
  --preparation-directory /secure/evaluation_preparation \
  --output-directory /secure/evaluation_output \
  --expected-plan-receipt-sha256 "$PLAN_SHA256" \
  --expected-advanced-exposure-declaration-sha256 "$EXPOSURE_SHA256" \
  --expected-output-receipt-sha256 "$OUTPUT_SHA256"
```

## Report semantics

A v3 report uses `protocol_status: "RECEIPT_CHAIN_VERIFIED"` and
`receipt_chain_verified: true` only to state that its protected inputs,
preparation artifacts, report, schema, and output receipt are content-bound.
Both `valid_for_model_selection` and `valid_for_final_reporting` are always
`false`. Receipt verification does not review the protocol design, model,
dataset construction, representativeness, statistical analysis, or scientific
claims.

Scientific validity therefore requires a separate external protocol, model,
and data review. No certification process or certification field is implemented
by this evaluator. A verified v3 report must not be promoted to model-selection
or final-reporting evidence without such an independently governed review.

The canonical v3 JSON Schema is the installed package resource
`artifact_contracts/schemas/cvi.evaluation.report.v3.schema.json`. Validation and schema
receipt hashing load that resource through `importlib.resources`, so they do
not depend on a source checkout or current working directory.

This correction is intentionally incompatible with reports that used the old
v3 final-reporting claim. Such reports fail the corrected v3 schema, and old
`cvi.protected_evaluation_output_bundle.v1` receipts are not accepted as v2
receipt-integrity outputs. Regenerate the report and output receipt from the
unchanged, externally anchored preparation artifacts.

## Limitations

This implementation covers protected closed-set, max-template cosine retrieval.
Protected verification and open-set identification remain unavailable rather
than falling back to unpinned inference. Atomic no-replace publication protects
against ordinary concurrent jobs and partial output, not a hostile process with
the same OS credentials. A verified receipt proves artifact integrity relative
to the supplied external anchors; it does not independently establish dataset,
model, protocol, or scientific validity.
