# Event-driven Capacity Model

## Purpose

CVI capacity is determined by stage invocation rates as well as per-call
latency. A faster kernel can still increase total cost if it is invoked more
often. The first-order model therefore separates empty-cage and occupied-cage
rates:

\[
\lambda_s(p)=(1-p)\lambda_{s,empty}+p\lambda_{s,occupied},
\]

where \(p\) is measured occupied time from admitted evidence rather than an
assumed constant. For \(C\) cameras and interval \(T\),

\[
n_s=C\,T\,\lambda_s(p).
\]

If stage \(s\) consumes mean service time \(d_s\) on resource \(r\), the
serial-demand utilization approximation is

\[
\rho_r=
\frac{C\sum_{s\in r}\lambda_s(p)d_s}{u_r},
\]

where \(u_r\) is the declared number of parallel service units. Both the
measured state mix and the maximum declared state rate are reported.

This is an admission and planning model, not a queueing-latency guarantee.
Burst arrivals, batch waiting, synchronization, transfers, contention, and
tail latency require replay on admitted camera traces.

## Event-driven optimization order

1. Decode and stream-health monitoring remain continuous where required.
2. Empty-cage detection uses the lowest rate that preserves dog-entry event
   recall and maximum detection delay.
3. Occupied-cage detection is combined with per-frame motion/tracking rather
   than rerunning the detector at stored FPS.
4. Region, quality, and embedding work is executed only on eligible,
   non-duplicate evidence.
5. Search is executed per admitted tracklet/template update, not per frame.
6. Sequential identity consensus may stop new embeddings after a safe decision,
   but monitoring must detect occupancy/identity changes.

Every rate reduction is a model change. It must pass event recall, track purity,
evidence coverage, decision latency, and false-assignment constraints; computed
call savings alone do not promote it.

## Peak memory

Each allocation is classified exactly once:

\[
M =
M_{shared}
+C M_{stream}
+A M_{track}
+B M_{batch}
+R M_{workspace},
\]

where \(A\) is simultaneous active tracks, \(B\) is actual batch items, and
\(R\) is concurrent engine/workspace replicas. Framework allocator reserve,
driver contexts, host pinned buffers, and index overhead remain explicit
components rather than hidden multipliers.

Sharing model weights is valid only when the runtime truly shares the same
engine/context allocation. Increasing batch size or silently serializing work
does not count as a memory optimization.

## Required measurements

- G1 duration-weighted occupancy fraction and entry/exit burst distribution;
- per-stage call rate from trace receipts;
- per-call p50/p95/max at the exact shape, precision, batch, and backend;
- CPU, GPU, and video-decoder demand separated by resource;
- steady, transition, dynamic-shape, and concurrent-stream memory peaks;
- fixed-rate comparator and worst-case cap for every adaptive scheduler.

The returned maximum camera count is only the floor implied by the declared
mean-service demand and target utilization. It cannot be presented as supported
camera capacity until trace replay satisfies latency and safety gates. The
implementation advances the floating-point ratio by one representable value
before flooring so an exact integer boundary such as \(0.7/0.05=14\) is not
reported as 13 solely because of binary representation error.

The checked-in example contains placeholders, not measurements:

```bash
uv run python tools/evaluate_capacity.py \
  configs/research/benchmarks/capacity.example.json \
  --duration-seconds 86400
```
