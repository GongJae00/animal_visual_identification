# Camera and Source Intake

## Boundary

Raw video remains in a protected external data root. The repository receives
only code, small manifests, checksums, and summarized measurements. Probing is
read-only and does not re-encode or copy the source.

## One-source probe

An absolute recording start time is required because container PTS commonly
does not establish facility wall-clock time.

```bash
uv run python tools/probe_video.py /protected/path/source.mp4 \
  --source-id source-0001 \
  --camera-id camera-01 \
  --cage-id cage-01 \
  --camera-setting-version settings-v1 \
  --recording-start-ns 1784516400000000000 \
  --modality RGB > /protected/manifests/source-0001.json
```

The command records a bounded-memory SHA-256, byte size, codec/container,
resolution, measured average FPS, duration, bitrate when present, frame count
when present, time base, and the declared modality interval. A file containing
a day/night switch must later be divided into contiguous `RGB`,
`RGB_IR_TRANSITION`, `IR`, or explicit `UNKNOWN` intervals; gaps are rejected.

## Acquisition manifest

`cvi.acquisition.v1` combines:

- versioned camera specifications;
- protected source records;
- camera and cage linkage;
- source hashes;
- RGB/IR modality intervals.

Validate a JSON manifest with:

```bash
uv run python tools/check_acquisition_manifest.py \
  /protected/manifests/acquisition.json
```

Exit code 0 means that the structural G0 gate has no blocker. It does not mean
that image evidence is adequate. The checker requires complete camera facts,
unique source IDs and hashes, known camera-setting references, at least three
cages, a contiguous 24-hour interval per cage, RGB and IR coverage in every
cage, and a per-cage transition interval when that cage uses a day/night-
switching camera. Evidence is never pooled across cages to satisfy this gate.

## Timestamp audit

`TimestampAuditAccumulator` consumes decoded or packet timestamps online and
retains only constant-size state:

```text
observed count
unavailable timestamp count
first/last timestamp
inversion count
duplicate count
estimated missing frames
maximum forward gap
```

This avoids storing millions of frame timestamps while preserving the initial
integrity statistics. Packet-level extraction and frame-level decoding costs
must be measured separately on admitted camera files.

```bash
uv run python tools/audit_timestamps.py /protected/path/source.mp4 \
  --expected-fps 29.97 \
  --level frame
```

`frame` uses decoded-frame best-effort timestamps and is the decision-timeline
audit. `packet` uses container packet DTS and is a faster decode-order
diagnostic. PTS is intentionally not used in packet iteration because B-frame
reordering creates legitimate non-monotonic PTS in decode order. Both modes
stream ffprobe output instead of materializing a multi-million-row JSON
document.
