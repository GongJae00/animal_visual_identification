# Data And Model Artifacts

## Non-Bundling Policy

The repository does not bundle datasets, pretrained weights, generated ONNX
files, trained checkpoints, galleries, caches, or experiment outputs. Keep
those artifacts outside the checkout and review their source terms separately.
The repository's Apache-2.0 license does not relicense third-party material.

Data roots resolve from `CVI_DATA_DIR`, defaulting to `~/cvi_data`. Model roots
resolve from `CVI_MODELS_DIR`, defaulting to `~/.cache/cvi/models`.

```bash
export CVI_DATA_DIR=/path/to/cvi-data
export CVI_MODELS_DIR=/path/to/cvi-model-cache
```

Both values above are placeholders for user-controlled directories.

## Dataset Downloader

Inspect the current status first:

```bash
uv run python tools/download_datasets.py --list
uv run python tools/download_datasets.py --help
```

Current behavior is deliberately narrower than the command's name suggests:

| Dataset selector | Status | Current behavior |
|---|---|---|
| `dogfacenet` | Disabled/manual | The Hugging Face location is an unpinned discovery tip, not an admitted download; selecting it fails without network access or directory creation |
| `yt-bb-dog` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |
| `sibetan` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |
| `mpdd` | Disabled/manual | Displays a manual source tip in `--list`; selecting it fails without network access or directory creation |

Use `--data-root /path/to/cvi-data` to override the root for this command. The
no-argument and `--dataset all` forms are intentional successful no-ops because
there are no admitted automatic downloads. They print that no network request
or filesystem change was attempted. Every named selector fails with manual
guidance.

A source tip is not an artifact admission or license determination. Before use,
independently record and review the source revision, content hash, retrieval
date, applicable terms, identity labels, duplicate policy, and permitted use.
Do not publish source labels or owner information as registered identities.

## Model Downloader

Inspect model acquisition status before model work:

```bash
uv run python tools/download_models.py --list
uv run python tools/download_models.py --help
```

Current model behavior:

| Model selector | Operation status | Current behavior and boundary |
|---|---|---|
| `dogflw-landmark` | Disabled | Fails before download because no publisher-authoritative artifact URL, checksum, and redistribution contract are verified |
| `miewid` | Disabled and unadmitted | Fails before network or model-framework imports because the removed converter did not produce the exact `cvi.miewid_artifact_bundle.v1` runtime manifest or a genuine passing parity receipt |
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

- Store immutable source files read-only where practical.
- Record source URL or repository, revision, file hash, retrieval date, and
  license for every artifact.
- Treat model formats and image archives as untrusted input; isolate conversion
  and extraction from secrets and production systems.
- Keep research-only and deployment-reviewed artifacts in separate directories.
- Never commit private data, weights, generated galleries, or receipt bundles.
- Use identity-disjoint partitions and audit duplicate or sequence leakage
  before interpreting metrics.

See [Known Limitations](KNOWN_LIMITATIONS.md) and
[`configs/README.md`](../configs/README.md). Keep datasets, model weights,
generated exports, and caches outside the repository.
