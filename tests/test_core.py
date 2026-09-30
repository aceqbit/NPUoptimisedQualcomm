"""Backend (sim), metrics, freeze fix, agent tests. No NPU needed."""
import json

import numpy as np
import onnx
import pytest

from sentinel import agent, fixes, metrics
from sentinel.backend import SimulatedBackend
from sentinel.scan import scan

# ---------- simulated backend ----------


def test_sim_strict_fails_on_symbolic(tiny, cfg):
    r = SimulatedBackend(cfg).run_strict(tiny["model"], tiny["test"], tiny["dir"], "v0")
    assert r.status == "COMPILE_ERROR" and "symbolic" in r.error and r.backend == "SIMULATED"


def test_sim_placement_share_float_frozen(tiny, cfg):
    out = tiny["dir"] / "f.onnx"
    fixes.freeze_shapes(tiny["model"], out, {"batch": 1}, inputs_npz=tiny["test"], exact_tol=1e-4)
    r = SimulatedBackend(cfg).run_placement(out, tiny["test"], tiny["dir"], "v1")
    assert r.status == "OK" and r.share == 0.3 and r.backend == "SIMULATED"


def test_sim_quantized_passes_strict(tiny, cfg):
    f = tiny["dir"] / "f.onnx"
    q = tiny["dir"] / "q.onnx"
    fixes.freeze_shapes(tiny["model"], f, {"batch": 1}, inputs_npz=tiny["test"], exact_tol=1e-4)
    fixes.quantize(f, q, tiny["calib"], "a8w8", allow_synthetic=True)
    be = SimulatedBackend(cfg)
    assert be.run_strict(q, tiny["test"], tiny["dir"], "v2").status == "OK"
    assert be.run_placement(q, tiny["test"], tiny["dir"], "v2").share == 1.0

# ---------- metrics / gate ----------


def _m(r, c):
    return metrics.compare({"o": r}, {"o": c})


def test_identical_pass(cfg):
    a = np.random.default_rng(0).standard_normal((5, 1, 10)).astype(np.float32)
    assert metrics.gate(_m(a, a.copy()), 1.0, cfg)["passed"]


def test_sign_flip_fails(cfg):
    a = np.random.default_rng(0).standard_normal((5, 1, 10)).astype(np.float32)
    g = metrics.gate(_m(a, -a), 1.0, cfg)
    assert g["status"] == "ACCURACY_FAIL"


def test_nan_fails(cfg):
    a = np.ones((3, 1, 4), np.float32)
    b = a.copy()
    b[0, 0, 0] = np.nan
    assert metrics.gate(_m(a, b), 1.0, cfg)["status"] == "ACCURACY_FAIL"


def test_share_none_fails(cfg):
    a = np.ones((3, 1, 4), np.float32)
    g = metrics.gate(_m(a, a), None, cfg)
    assert not g["passed"] and "share unavailable" in g["reasons"]


def test_top1_flip_with_high_cosine_caught(cfg):
    base = np.full((10, 1, 100), 10.0, np.float32)
    base[:, 0, 0] = 10.02
    cand = base.copy()
    cand[:, 0, 1] = 10.03  # argmax moves from 0 to 1; cosine stays ~1
    m = _m(base, cand)
    assert m["o"]["cosine_min"] > 0.9999
    assert metrics.gate(m, 1.0, cfg)["status"] == "ACCURACY_FAIL"


def test_accuracy_beats_share(cfg):
    a = np.random.default_rng(0).standard_normal((5, 1, 10)).astype(np.float32)
    assert metrics.gate(_m(a, -a), 0.1, cfg)["status"] == "ACCURACY_FAIL"

# ---------- freeze_shapes ----------


def test_freeze_works(tiny, cfg):
    out = tiny["dir"] / "f.onnx"
    r = fixes.freeze_shapes(tiny["model"], out, {"batch": 1}, inputs_npz=tiny["test"],
                            exact_tol=cfg["gate"]["exact_tol"])
    assert r["verified_max_abs_err"] <= cfg["gate"]["exact_tol"]
    assert scan(out)["symbolic_dims"] == []


def test_freeze_missing_dim_raises(tiny):
    with pytest.raises(ValueError, match="batch"):
        fixes.freeze_shapes(tiny["model"], tiny["dir"] / "f.onnx", {"other": 1},
                            inputs_npz=tiny["test"], exact_tol=1e-4)


def test_freeze_corrupted_rejected(tiny, cfg, monkeypatch):
    real = fixes.freeze_graph

    def corrupt(model, dv):
        m = real(model, dv)
        w = onnx.numpy_helper.to_array(m.graph.initializer[0]) * 2
        m.graph.initializer[0].CopyFrom(onnx.numpy_helper.from_array(w, m.graph.initializer[0].name))
        return m

    monkeypatch.setattr(fixes, "freeze_graph", corrupt)
    out = tiny["dir"] / "bad.onnx"
    with pytest.raises(fixes.FixRejected):
        fixes.freeze_shapes(tiny["model"], out, {"batch": 1}, inputs_npz=tiny["test"],
                            exact_tol=cfg["gate"]["exact_tol"])
    assert not out.exists()

# ---------- agent ----------


def _obs(**kw):
    o = {"strict_status": "COMPILE_ERROR", "strict_error": "x", "share": 0.0, "share_method": "s",
         "cos_min": 1.0, "top1_agreement": 1.0, "max_abs_err": 0.0, "latency_median_ms": 1.0,
         "cpu_latency_ms": 1.0, "symbolic_dims": ["batch"], "is_float": True, "attempt": 1,
         "tried": [], "last_gate_status": "SHARE_LOW"}
    o.update(kw)
    return o


def test_legal_order_and_no_repeats():
    assert agent.legal_actions(_obs(), [], True) == ["freeze_shapes", "quantize_a8w8", "stop"]
    h = [{"action": "freeze_shapes", "params": None}]
    assert agent.legal_actions(_obs(symbolic_dims=[]), h, True) == ["quantize_a8w8", "stop"]
    assert agent.legal_actions(_obs(symbolic_dims=[]), h, False) == ["stop"]


def test_a16_only_after_accuracy_fail():
    h = [{"action": "quantize_a8w8", "params": None}]
    o = _obs(symbolic_dims=[], is_float=False, last_gate_status="SHARE_LOW")
    assert agent.legal_actions(o, h, True) == ["stop"]
    o["last_gate_status"] = "ACCURACY_FAIL"
    assert agent.legal_actions(o, h, True) == ["quantize_a16w8", "stop"]
    h.append({"action": "quantize_a16w8", "params": None})
    assert agent.legal_actions(o, h, True) == ["stop"]


def test_observation_sanitizes():
    scan_ = {"symbolic_dims": [], "is_float": True}
    o = agent.observation(scan_, {"status": "COMPILE_ERROR", "error": "a\x00b\x1b" + "z" * 1000},
                          {}, {"status": "SHARE_LOW"}, {}, [], 1, 2.0)
    assert len(o["strict_error"]) == 300 and "\x00" not in o["strict_error"]
    assert set(o) == set(agent.OBS_KEYS)


def test_decision_log_json(tmp_path, cfg):
    log = agent.DecisionLog(tmp_path / "d.jsonl")
    c = agent.decide(_obs(), [], True, cfg, log, use_llm=False)
    assert c == "freeze_shapes"
    rec = [json.loads(l) for l in (tmp_path / "d.jsonl").read_text().splitlines()]
    assert rec[0]["source"] == "rules" and "batch" in rec[0]["rationale"]
