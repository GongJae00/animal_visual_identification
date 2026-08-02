# Decode Profiling

## Scope

Decode is measured before model inference because every downstream path depends
on it. The portable reference is FFmpeg software decode. `cuda` is an optional
guarded path and is never the only runnable configuration.

```bash
uv run python workflows/benchmark_decode.py /protected/path/source.mp4 \
  --source-id source-0001 \
  --source-sha256 SHA256_FROM_ACQUISITION_MANIFEST \
  --backend cpu \
  --duration-seconds 60 \
  --warmup 1 \
  --repeats 5
```

Run the same command with `--backend cuda` only when the local FFmpeg build and
codec support it.

For device-wide GPU telemetry, both telemetry arguments and the operator
attestation are required:

```bash
uv run python workflows/benchmark_decode.py /protected/path/source.mp4 \
  --source-id source-0001 \
  --source-sha256 SHA256_FROM_ACQUISITION_MANIFEST \
  --backend cuda \
  --duration-seconds 60 \
  --warmup 1 \
  --repeats 5 \
  --gpu-device-index 0 \
  --gpu-telemetry-interval-seconds 0.5 \
  --attest-no-unrelated-gpu-work
```

The attestation is an operator assertion, not an automatic proof. If unrelated
GPU work cannot be excluded, replace it with `--declare-unrelated-gpu-work`;
the run is retained as contaminated diagnostics and cannot support optimization
promotion. Omitting both declarations rejects a telemetry run. Short clips
primarily measure process and CUDA initialization; use admitted camera
intervals long enough to amortize startup before comparing backends.

## Receipt

The receipt records:

- source ID and protected-source hash;
- decode config hash and exact FFmpeg command;
- FFmpeg version;
- backend, duration, and CPU thread setting;
- warm-up and repeat counts;
- invariant decoded-frame count;
- wall-time p50/p95/max and derived decoded FPS/real-time factor;
- mean FFmpeg user/system time;
- maximum child-process RSS.
- optional device-wide GPU memory, utilization, decoder utilization, power,
  and raw sampled board-energy envelope.

FFmpeg `maxrss` is process resident memory, not GPU VRAM. GPU allocator memory,
NVDEC engine utilization, host↔device transfers, and power require separate
measurements. A synthetic codec smoke validates the tool but is not a camera
performance result.

GPU telemetry covers measured repeats only, excluding warm-up. It includes the
whole GPU board and does not subtract idle power, so its energy value is not
process energy or incremental workload energy. See `GPU_TELEMETRY.md`.

The tool rehashes the source after all timed repeats and rejects hash mismatch
or size/mtime changes. Hash I/O is outside the decode timing so provenance
verification cannot masquerade as decode cost or warm the first measured run.
