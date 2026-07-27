# ONNX Inference Measurement

This harness measures one frozen ONNX workload in fresh worker processes. It
does not decide whether an optimization is admissible and does not provide
canine biometric evidence.

Each worker authenticates the model, backend configuration, preprocessing,
dependency lock, and ordered artifact contents before and after inference. It
records the exact float32 preprocessed tensor hash and requires every identical
evaluation in that worker to produce the same output-byte hash.

The summary embeds the full policy, exact backend/preprocessing file hashes,
vector dimension, host/boot identity, and all authenticated worker results.
When reparsed, aggregate timing and RSS fields are recomputed from those worker
values. A changed aggregate with unchanged workers is rejected.

Fresh workers also receive a fixed value-free environment contract. The parent
removes `LD_LIBRARY_PATH`, `LD_PRELOAD`, `CUDA_HOME`, `CUDA_PATH`, `PYTHONHOME`,
and `PYTHONPATH`; the worker requires those keys to be absent and rehashes the
exact Python executable. Only the names that existed in the parent are recorded,
never their values. A non-empty parent `LD_PRELOAD` fails before launch.

CPU and CUDA are separate dependency lanes. A CPU worker requires exactly the
`onnxruntime` distribution and a CUDA worker requires exactly
`onnxruntime-gpu`. Configure each environment explicitly. Measurements from
temporary isolated environments are useful for tests but cannot support a
reusable absolute-path runtime policy.

The phases are deliberately separate:

- supervisor process wall time includes Python startup through worker exit;
- dependency import time covers the delayed CVI backend import and config parse;
- session construction includes ONNX/NumPy/Pillow imports, model checking,
  provider setup, and CUDA/cuDNN preload where applicable;
- preprocessing time constructs and hashes one exact input tensor;
- first preprocessed inference is the cold first API call;
- warm preprocessed inference reuses the exact tensor and includes the public
  tensor validation plus `session.run` with a CPU output;
- end-to-end inference includes file validation, Pillow decode, preprocessing,
  tensor validation, and `session.run`.

Linux `ru_maxrss` is a worker-process high-water mark. `/proc/<pid>/status`
sampling is a separate sampled worker-main-process scope and can miss short
peaks. NVIDIA telemetry is explicitly device-wide, includes initialization and
inter-worker gaps, and is not process-attributed VRAM or incremental energy.
The CUDA policy therefore requires an operator declaration; set it to `false`
unless unrelated GPU work was actively excluded for the complete interval.
The separate system-work declaration applies to CPU and CUDA alike and must be
true before `docs/PAIRED_INFERENCE_COMPARISON.md` accepts a pair.

CPU example:

```bash
export CVI_CPU_PYTHON=/path/to/cpu-environment/bin/python
export CVI_MODEL=/path/to/model.onnx
export CVI_ARTIFACT_A=/path/to/crop-a.png
export CVI_ARTIFACT_B=/path/to/crop-b.png
export CVI_CPU_RUNTIME_POLICY=/path/to/cpu-runtime-policy.json
export CVI_BENCHMARK_RECEIPTS=/path/to/benchmark-receipts

"$CVI_CPU_PYTHON" tools/benchmark_onnx_inference.py \
  --backend CPU \
  --model "$CVI_MODEL" \
  --backend-config configs/deployment/onnx_cpu_backend.example.json \
  --preprocessing configs/pipeline/evidence/image_preprocessing.example.json \
  --artifact "$CVI_ARTIFACT_A" --artifact "$CVI_ARTIFACT_B" \
  --dependency-lock uv.lock \
  --runtime-library-policy "$CVI_CPU_RUNTIME_POLICY" \
  --code-revision REVISION \
  --policy configs/deployment/onnx_inference_benchmark_cpu.example.json \
  --receipt "$CVI_BENCHMARK_RECEIPTS/cpu.json"
```

CUDA example:

```bash
export CVI_CUDA_PYTHON=/path/to/cuda-environment/bin/python
export CVI_CUDA_RUNTIME_POLICY=/path/to/cuda-runtime-policy.json

"$CVI_CUDA_PYTHON" tools/benchmark_onnx_inference.py \
  --backend CUDA \
  --model "$CVI_MODEL" \
  --backend-config configs/deployment/onnx_cuda_backend.example.json \
  --preprocessing configs/pipeline/evidence/image_preprocessing.example.json \
  --artifact "$CVI_ARTIFACT_A" --artifact "$CVI_ARTIFACT_B" \
  --dependency-lock uv.lock \
  --runtime-library-policy "$CVI_CUDA_RUNTIME_POLICY" \
  --code-revision REVISION \
  --policy configs/deployment/onnx_inference_benchmark_cuda.example.json \
  --receipt "$CVI_BENCHMARK_RECEIPTS/cuda.json"
```

Use a scheduler or bounded-job wrapper when required by local policy, but keep
that machine-specific control outside the portable command example.

The two backend configuration examples must be regenerated for the same real
model shape and preprocessing before use. Example values are not performance
targets.

`DISCOVERY_ONLY` receipts are inventory evidence, not admissible measurements.
`tools/freeze_runtime_library_policy.py` converts consistent discovery workers
to a candidate policy; that policy must be inspected and used in a second run
whose runtime decision is `PASS`.
