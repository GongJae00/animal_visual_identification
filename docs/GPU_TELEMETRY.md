# GPU Telemetry

## Measurement scope

Telemetry uses the locally installed `nvidia-smi` query interface and is
explicitly device-wide:

- `memory.used`: memory allocated by active contexts;
- `power.draw`: whole-board power (one-second average on Ampere and newer);
- `utilization.gpu`: sampled kernel-active time;
- `utilization.memory`: sampled global-memory activity;
- `utilization.decoder`: sampled decoder-engine activity.

The installed driver documents utilization sample windows between roughly
1/6 second and 1 second depending on the product. Faster polling does not create
finer sensor resolution. Board power accuracy is documented as approximately
±5 W.

Sampling uses one persistent `nvidia-smi --loop-ms` process. Spawning a new
process per sample is forbidden because its startup cost can dominate short
operations. Receipts preserve the requested and effective mean sample interval
plus the exact sampler command.

## Contamination rule

These values are not process-attributed. A receipt must retain
`scope=device-wide`, and a run with unrelated GPU work is contaminated. Device
memory includes driver and other contexts; it is not interchangeable with
FFmpeg process RSS or framework allocator peak memory.

The streaming accumulator stores counts, sums, maxima, endpoints, and
trapezoidal board-energy approximation in constant memory. Unsupported `N/A`
fields remain `null`.
