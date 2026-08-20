# Data And Model Artifacts

## Non-Bundling Policy

The repository does not bundle datasets, pretrained weights, generated ONNX
files, trained checkpoints, galleries, caches, or experiment outputs. Keep
those artifacts outside the checkout and review their source terms separately.
The repository's Apache-2.0 license does not relicense third-party material.

Data, admitted checkpoints, receipts, and experiment roots resolve from
`CANINE_IDENTITY_DATA_DIR`, defaulting to `~/canine_identity_data`.
Dataset files resolve from `CANINE_IDENTITY_DATASETS_DIR` when set, otherwise
`$CANINE_IDENTITY_DATA_DIR/datasets`.
`CANINE_IDENTITY_MODELS_DIR`, defaulting to
`~/.cache/canine_identity/models`, is a candidate-model cache used only by disabled
or unadmitted acquisition paths; cataloged artifacts live under
`$CANINE_IDENTITY_DATA_DIR/checkpoints`.

```bash
export CANINE_IDENTITY_DATA_DIR=/path/to/identity-data
export CANINE_IDENTITY_DATASETS_DIR=/path/to/datasets
export CANINE_IDENTITY_MODELS_DIR=/path/to/identity-model-cache
```

Those values are placeholders for user-controlled directories.

Use one content-oriented directory per dataset. License and workflow admission
belong in registry metadata rather than directory names:

```text
$CANINE_IDENTITY_DATA_DIR/
  checkpoints/<model-artifact-id>/
  experiments/
  receipts/
  manifests/
  cache/
  artifacts/
$CANINE_IDENTITY_DATASETS_DIR/{ap10k,dogflw,dogfacenet224,mpdd,sibetan,yt-bb-dog}/
```

## Dataset Downloader

Inspect the current status first:

```bash
uv run python -m data.commands.download datasets --list
uv run python -m data.commands.download datasets --help
```

Current behavior is deliberately narrower than the command's name suggests:

| Dataset selector | Status | Current behavior |
|---|---|---|
| `ap10k-dog` | Disabled/manual | Displays the official repository tip; selecting it fails without network access or directory creation |
| `dogflw` | Disabled/manual | Displays the publisher dataset tip; selecting it fails without network access or directory creation |
| `dogfacenet` | Disabled/manual | The Hugging Face location is an unpinned discovery tip, not an admitted download; selecting it fails without network access or directory creation |
| `yt-bb-dog` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |
| `sibetan` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |
| `mpdd` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |

Use `--data-root /path/to/identity-data` to override the working root for this command. The
no-argument and `--dataset all` forms are intentional successful no-ops because
there are no admitted automatic downloads. They print that no network request
or filesystem change was attempted. Every named selector fails with manual
guidance.

A source tip is not an artifact admission or license determination. Before use,
independently record and review the source revision, content hash, retrieval
date, applicable terms, identity labels, duplicate policy, and permitted use.
Do not publish source labels or owner information as registered identities.
Dataset folder, cluster, and video-track labels are protocol labels: their
deterministic UUID mappings do not establish lifelong animal identity.

Unlabeled clusters may receive a `cvi.generated_identity_registry.v1`
provisional GenID for SSL, mining, or research bookkeeping. GenIDs use a
separate UUIDv5 namespace and never become registered identities implicitly.
Only an explicit audited transition may merge a provisional record into a
canonical registered dog ID; rejected and superseded records remain auditable.

## Model Downloader

Inspect model acquisition status before model work:

```bash
uv run python -m data.commands.download models --list
uv run python -m data.commands.download models --help
```

Current model behavior:

| Model selector | Operation status | Current behavior and boundary |
|---|---|---|
| `dogflw-landmark` | Disabled | Fails before download because no publisher-authoritative artifact URL, checksum, and redistribution contract are verified |
| `miewid` | Disabled and unadmitted | Fails before network or model-framework imports; no admitted `cvi.miewid_artifact_bundle.v1` runtime manifest or genuine passing parity receipt |
| `superanimal` | Disabled | Fails before download because the weights and official-architecture export contract are not approved |

`supported` operations are automatic and included by the default `all`
selector. `manual` operations provide instructions only and are skipped by
`all`. `disabled` operations fail closed when selected and are also skipped by
`all`. There are currently no supported automatic or manual model operations,
so the default command performs no network or conversion work.

The model command is not a production bundle installer. A file already present
at a displayed cache path is inventory only, not validation or admission. A
runtime artifact still requires the exact manifest and preprocessing contract,
genuine parity evidence, license review, and identity-evaluation admission
expected by its channel implementation.

## Artifact Handling

Known local model artifacts are inventoried by `shared.contracts.model_catalog`. Call
`get_model_artifact("<role>")` for logical selection and
`verify_model_artifact("<role>")` when exact bytes are required. Role aliases
are logical lookups, not filesystem symlinks. Current roles include
`appearance-backbone`, `appearance-onnx`, `dog-detector`, `dog-pose`, and
research teacher/initializer variants. Every record carries its source model,
revision status, SHA-256, license identifier, and admission state. An
`unverified-*` revision explicitly blocks treating acquisition metadata as an
upstream release identifier.

- Store immutable source files read-only where practical.
- Record source URL or repository, revision, file hash, retrieval date, and
  license for every artifact.
- Treat model formats and image archives as untrusted input; isolate conversion
  and extraction from secrets and production systems.
- Record research-only and deployment-reviewed admission in versioned metadata;
  directory names identify artifact content and must not imply legal status.
- Never commit private data, weights, generated galleries, or receipt bundles.
- Use identity-disjoint partitions and audit duplicate or sequence leakage
  before interpreting metrics.

See [Known Limitations](KNOWN_LIMITATIONS.md) and
tracked configurations in their owning package directories. Keep datasets, model weights,
generated exports, and caches outside the repository.
