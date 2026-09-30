"""Backend abstraction: the real QNN backend (subprocess) and a SIMULATED backend for off-device work."""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Protocol

import numpy as np

from . import baseline, env, probe
from .scan import scan


@dataclass
class ProbeResult:
    status: str
    error: str | None = None
    compile_time_s: float | None = None
    providers: list = field(default_factory=list)
    backend: str = ""

    def to_dict(self):
        return asdict(self)


@dataclass
class PlacementResult:
    status: str
    error: str | None = None
    share: float | None = None
    share_method: str = "unavailable"
    latency_median_ms: float | None = None
    latency_p95_ms: float | None = None
    outputs_path: str | None = None
    providers: list = field(default_factory=list)
    notes: list = field(default_factory=list)
    backend: str = ""
    extra: dict = field(default_factory=dict)

    def to_dict(self):
        return asdict(self)


class Backend(Protocol):
    name: str

    def run_strict(self, model_path, inputs_path, out_dir, tag) -> ProbeResult: ...

    def run_placement(self, model_path, inputs_path, out_dir, tag) -> PlacementResult: ...


class QnnBackend:
    """Real device backend. All sessions are created in the worker subprocess."""

    name = "QNN"

    def __init__(self, cfg, config_path=None):
        import onnxruntime as ort  # listing providers does not create a session
        if env.QNN_EP not in ort.get_available_providers():
            raise env.EnvError(f"{env.QNN_EP} not available; providers="
                               f"{ort.get_available_providers()}. Use native ARM64 Python with "
                               "onnxruntime-qnn on a Snapdragon device, or --backend sim.")
        self.cfg, self.config_path = cfg, config_path

    def run_strict(self, model_path, inputs_path, out_dir, tag) -> ProbeResult:
        r = probe.run_worker("strict", model_path, inputs_path, out_dir, self.cfg, tag=tag,
                             config_path=self.config_path)
        return ProbeResult(status=r["status"], error=r.get("error"),
                           compile_time_s=r.get("compile_time_s"),
                           providers=r.get("providers", []), backend=self.name)

    def run_placement(self, model_path, inputs_path, out_dir, tag) -> PlacementResult:
        r = probe.run_worker("placement", model_path, inputs_path, out_dir, self.cfg, tag=tag,
                             config_path=self.config_path)
        known = {"status", "error", "share", "share_method", "latency_median_ms",
                 "latency_p95_ms", "outputs_path", "providers", "notes"}
        return PlacementResult(
            status=r["status"], error=r.get("error"), share=r.get("share"),
            share_method=r.get("share_method", "unavailable"),
            latency_median_ms=r.get("latency_median_ms"), latency_p95_ms=r.get("latency_p95_ms"),
            outputs_path=r.get("outputs_path"), providers=r.get("providers", []),
            notes=r.get("notes", []), backend=self.name,
            extra={k: v for k, v in r.items() if k not in known})


class SimulatedBackend:
    """Off-device stand-in. Results are labelled SIMULATED and must never be mixed with real ones.

    Rules: strict fails if the model has symbolic input dims or no QuantizeLinear nodes.
    Share: 0.0 if symbolic dims, 0.3 if float, 1.0 otherwise. Outputs come from running the
    candidate model on CPU plus small deterministic noise when quantized (A16 < A8).
    """

    name = "SIMULATED"
    NOISE = {8: 1e-2, 16: 1e-3}

    def __init__(self, cfg):
        self.cfg = cfg

    def run_strict(self, model_path, inputs_path, out_dir, tag) -> ProbeResult:
        s = scan(model_path)
        if s["symbolic_dims"]:
            names = ", ".join(d["name"] for d in s["symbolic_dims"])
            return ProbeResult("COMPILE_ERROR", f"SIMULATED: symbolic input dims ({names})",
                               None, ["SIMULATED"], self.name)
        if s["is_float"]:
            return ProbeResult("COMPILE_ERROR", "SIMULATED: float graph, no QuantizeLinear nodes",
                               None, ["SIMULATED"], self.name)
        return ProbeResult("OK", None, None, ["SIMULATED"], self.name)

    def run_placement(self, model_path, inputs_path, out_dir, tag) -> PlacementResult:
        s = scan(model_path)
        share = 0.0 if s["symbolic_dims"] else (0.3 if s["is_float"] else 1.0)
        data = baseline.load_inputs(inputs_path)
        try:
            sess = baseline.cpu_session(model_path)
            outs = baseline.run_all(sess, data["x"])
            t = baseline.time_runs(sess, data["x"], self.cfg["timing"]["warmup"],
                                   self.cfg["timing"]["runs"])
        except Exception as e:  # noqa: BLE001
            return PlacementResult("RUNTIME_ERROR", f"{type(e).__name__}: {e}"[:500],
                                   backend=self.name)
        bits = s["activation_bits"]
        if bits:
            rng = np.random.default_rng(0)
            scale = self.NOISE[16 if bits >= 16 else 8]
            outs = {k: v + (scale * (np.std(v) or 1.0) * rng.standard_normal(v.shape)).astype(v.dtype)
                    for k, v in outs.items()}
        out_npz = Path(out_dir) / f"outputs_{tag}.npz"
        baseline.save_outputs(out_npz, outs)
        return PlacementResult("OK", None, share, "simulated", t["latency_median_ms"],
                               t["latency_p95_ms"], str(out_npz), ["SIMULATED"],
                               ["latency measured on host CPU by the simulated backend"],
                               self.name)
