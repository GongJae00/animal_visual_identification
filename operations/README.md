# Operations

In: frozen models, receipts, source video, and worker policies.

Out: isolated worker runs, decode/ONNX/capacity measurements, video probes.

`workers/` are sanitized fresh processes. `measurement/` is capacity, telemetry,
and ONNX inference benches. `video/` is FFmpeg decode measurement. IdentityEngine
does not import this package.

Commands: `uv run python -m operations.commands.measure --help`
(`onnx`, `probe`, `decode`, `capacity`, `compare`).
