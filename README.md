# Sentinel

## 1. Problem
ONNX Runtime's QNN Execution Provider silently falls back to the CPU for any part of a model the
Hexagon NPU backend cannot take. A model can "run with QNN" while most of its work runs on the CPU,
and nothing tells you. Quantizing to fix that can quietly break accuracy.

What Sentinel does:
1. Runs the FP32 model on the CPU to get reference outputs and a CPU latency.
2. Scans the graph for symbolic dims, float vs QDQ, and ops (advisory only).
3. Strict probe: QNN EP with `session.disable_cpu_ep_fallback=1`, in a subprocess with a timeout.
4. Placement run: QNN EP with fallback allowed, to measure NPU share, latency and outputs.
5. Gate: NPU share >= `share_min` AND outputs match the FP32 reference (cosine, top-1).
6. If the gate fails, a rule-constrained agent picks one verified fix (`freeze_shapes`,
   `quantize_a8w8`, `quantize_a16w8`), then it re-probes. It keeps the best version and reverts regressions.

## 2. Status

| Component | Status |
|---|---|
| Env check (`scripts/check_env.py`, `sentinel/env.py`) | BUILT, run on x64 dev machine (correctly exits 2: not ARM64) |
| Model/data helpers (`scripts/get_model.py`, `scripts/make_data.py`) | BUILT, run on MobileNetV2 (ONNX Model Zoo) |
| Static scan, CPU baseline | BUILT, run |
| Simulated backend | BUILT, tested |
| Real QNN backend, worker subprocess, probe | BUILT, **not yet run on a Snapdragon device**. On x64 it raises `EnvError` as designed |
| Metrics and gate | BUILT, tested |
| Fix: freeze_shapes (CPU-verified exact) | BUILT, tested, run |
| Fix: QDQ quantization a8w8 / a16w8 | BUILT, tested, run (on CPU) |
| Agent: legal actions, rules policy, decision log | BUILT, tested |
| LLM policy (local OpenAI-compatible, urllib) | BUILT, tested against a fake server only. Disabled by default |
| Orchestrator loop and CLI | BUILT, run end-to-end on the simulated backend |
| HTML report | BUILT, tested |
| On-device results | ROADMAP (next step: P14 on the Snapdragon laptop) |

## 3. Requirements and setup (Windows 11 ARM64, Snapdragon X)
```powershell
python -c "import platform; print(platform.machine())"   # must print ARM64 (native ARM64 Python 3.11+)
pip install -r requirements.txt                            # binary wheels only; stop if anything compiles
python scripts/check_env.py                                # QNNExecutionProvider must be listed; NPU driver entries shown
```
For off-device development on x64, install `onnxruntime` (CPU) instead of `onnxruntime-qnn` and use `--backend sim`.

## 4. Quickstart
```powershell
python scripts/get_model.py --path C:\path\model.onnx --out models/base.onnx     # or --url <your model URL>
python scripts/make_data.py --images C:\path\images --model models/base.onnx --calib 100 --test 100 --out data
python scripts/check_env.py
python -m sentinel.probe --mode strict --model models/base.onnx --inputs data/test_inputs.npz
python -m sentinel run --model models/base.onnx --inputs data/test_inputs.npz --calib data/calib_inputs.npz --dim batch=1
pytest -q
```
Use `--dim NAME=VALUE` with the model's actual symbolic dim name (MobileNetV2 from the ONNX Model Zoo uses `batch_size`).
Off-device: add `--backend sim --no-llm`. Every result it produces carries a SIMULATED banner.

Exit codes: `0` PASS, `1` ATTEMPTS_EXHAUSTED, `2` environment error, `3` DEVICE_UNHEALTHY
(the NPU is unresponsive: reboot, then re-run. Sentinel never reboots automatically).

## 5. Reading `runs/<id>/`
| File | Content |
|---|---|
| `env.json` | environment fingerprint (versions, providers, NPU driver entries) |
| `baseline.json`, `reference_outputs.npz` | CPU FP32 reference outputs and latency |
| `probe_vN.json` | strict-probe result for version N |
| `placement_vN.json`, `outputs_vN.npz` | placement run: share, share_method, latency, providers, outputs |
| `gate_vN.json` | per-output metrics and gate decision |
| `decisions.jsonl` | one line per agent decision: observation, legal actions, choice, rationale, source |
| `models/vN.onnx` | each model version |
| `summary.json` | final status, before/after, attempts, unresolved problems |
| `report.html` | self-contained report rendered only from the files above |

## 6. Results
No on-device results yet. `artifacts/` is reserved for real-backend runs only, copied from
`summary.json` files. Simulated runs are never put there.

Task Manager NPU screenshot: add it here manually after the on-device run. It is intentionally not included.

## 7. Limitations
- Single-input models only (MVP).
- Fix menu: freeze_shapes, quantize_a8w8, quantize_a16w8. Nothing else.
- NPU share methods, in order:
  - `strict_pass`: 1.0 when strict mode succeeds.
  - `epcontext`: 1 − (non-EPContext nodes in the EP-context model / input nodes).
  - `ort_profile_time`: fraction of kernel time on the QNN EP.
  - Otherwise `unavailable`, which fails the gate. The share is never guessed.
- Gate thresholds in `config.toml` are defaults, not tuned values.
- Quantization uses `quantize_static` with QDQ. The installed onnxruntime (1.30.0 on the dev machine)
  has no `quantization.execution_provider.qnn` helper. Calibration size matters: with too few
  calibration samples, test inputs fall outside the calibrated range and the gate rejects the model.
- UNVERIFIED on a device. Every spot is marked `# TODO(verify)` in `sentinel/worker.py` and `sentinel/fixes.py`:
  - QNN provider option names (`backend_path`, `htp_performance_mode`, `profiling_level`, `profiling_file_path`)
  - the EP-context keys (`ep.context_enable`, `ep.context_file_path`) and their output format
  - classic (1.x) vs plugin (2.x) EP registration
  - the QNN QDQ helper

## 8. Roadmap
Failure bisection, op swap, graph split, FLOPs-weighted coverage, compiled-context cache, CI guard,
drift watcher after driver/OS updates, Windows ML path.
