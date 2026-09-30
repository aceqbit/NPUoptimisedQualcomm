"""Orchestrator: baseline -> (scan -> strict -> placement -> gate -> agent -> fix)* -> summary + report."""
from __future__ import annotations

import json
import shutil
from pathlib import Path

from . import agent, env, fixes, metrics, probe
from .backend import QnnBackend, SimulatedBackend
from .baseline import load_inputs, new_run_dir
from .scan import scan

REGRESSION_TOL = 1e-3
UNHEALTHY_MSG = ("NPU unresponsive: reboot the device, then re-run. "
                 "Sentinel never reboots automatically.")
EXIT = {"PASS": 0, "ATTEMPTS_EXHAUSTED": 1, "DEVICE_UNHEALTHY": 3}


def _dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str), encoding="utf-8")


def make_backend(name, cfg, config_path=None):
    if name == "sim":
        return SimulatedBackend(cfg)
    if name == "qnn":
        return QnnBackend(cfg, config_path)
    raise ValueError(f"unknown backend {name!r}")


def evaluate(backend, model, inputs, run_dir, tag, ref_npz, cfg) -> dict:
    s = scan(model)
    strict = backend.run_strict(model, inputs, run_dir, tag).to_dict()
    _dump(run_dir / f"probe_{tag}.json", strict)
    if strict["status"] in ("TIMEOUT", "DEVICE_UNHEALTHY"):
        return {"tag": tag, "model": str(model), "scan": s, "strict": strict, "unhealthy": True}
    place = backend.run_placement(model, inputs, run_dir, tag).to_dict()
    _dump(run_dir / f"placement_{tag}.json", place)
    if place["status"] in ("TIMEOUT", "DEVICE_UNHEALTHY"):
        return {"tag": tag, "model": str(model), "scan": s, "strict": strict, "placement": place,
                "unhealthy": True}
    share = 1.0 if strict["status"] == "OK" else place.get("share")
    share_method = "strict_pass" if strict["status"] == "OK" else place.get("share_method")
    m = metrics.compare(ref_npz, place["outputs_path"]) if place["status"] == "OK" else None
    g = metrics.gate(m, share, cfg)
    if place["status"] != "OK":
        g = {"passed": False, "status": place["status"],
             "reasons": [f"placement run {place['status']}: {agent.sanitize(place.get('error'))}"]}
    agg = metrics.aggregate(m) if m else {"cos_min": None, "max_abs_err": None,
                                          "top1_agreement": None}
    vr = {"tag": tag, "model": str(model), "scan": s, "strict": strict, "placement": place,
          "metrics": m, "gate": g, **agg, "share": share, "share_method": share_method,
          "latency_median_ms": place.get("latency_median_ms"),
          "latency_p95_ms": place.get("latency_p95_ms"), "passed": g["passed"], "unhealthy": False}
    _dump(run_dir / f"gate_{tag}.json", {k: vr[k] for k in ("gate", "metrics", "cos_min",
                                                            "max_abs_err", "top1_agreement",
                                                            "share", "share_method")})
    return vr


def _brief(vr):
    if not vr:
        return None
    keys = ["tag", "share", "share_method", "cos_min", "max_abs_err", "top1_agreement",
            "latency_median_ms", "latency_p95_ms", "passed"]
    out = {k: vr.get(k) for k in keys}
    out["strict_status"] = vr["strict"]["status"]
    out["gate_status"] = vr.get("gate", {}).get("status")
    out["providers"] = (vr.get("placement") or {}).get("providers")
    return out


def run(model, inputs, calib=None, *, backend="qnn", dims=None, use_llm=True, max_attempts=None,
        cfg=None, config_path=None, runs_root=None, backend_obj=None) -> tuple[int, Path]:
    cfg = cfg or env.load_config(config_path)
    max_attempts = max_attempts or cfg["loop"]["max_attempts"]
    be = backend_obj or make_backend(backend, cfg, config_path)
    run_dir = new_run_dir(runs_root or env.REPO_ROOT / "runs")
    models_dir = run_dir / "models"
    models_dir.mkdir()
    _dump(run_dir / "env.json", env.fingerprint())

    ref = probe.run_worker("reference", model, inputs, run_dir, cfg, tag="ref",
                           config_path=config_path)
    if ref["status"] != "OK":
        raise env.EnvError(f"CPU reference run failed: {ref.get('error')}")
    ref_npz = run_dir / "reference_outputs.npz"
    cpu_lat = ref["latency_median_ms"]
    _dump(run_dir / "baseline.json", ref)
    synthetic = load_inputs(inputs)["synthetic"] or (bool(calib) and load_inputs(calib)["synthetic"])

    v0 = models_dir / "v0.onnx"
    shutil.copyfile(model, v0)
    working, float_src = v0, v0
    log = agent.DecisionLog(run_dir / "decisions.jsonl")
    history, versions, best = [], [], None
    status, next_idx = "ATTEMPTS_EXHAUSTED", 1

    for attempt in range(1, max_attempts + 1):
        tag = working.stem
        vr = evaluate(be, working, inputs, run_dir, tag, ref_npz, cfg)
        vr["attempt"] = attempt
        vr["action"] = history[-1]["action"] if history else None
        if vr["unhealthy"]:
            versions.append(vr)
            status = "DEVICE_UNHEALTHY"
            print(UNHEALTHY_MSG)
            break
        vr["regressed"] = bool(best and best["cos_min"] is not None and vr["cos_min"] is not None
                               and vr["cos_min"] < best["cos_min"] - REGRESSION_TOL)
        versions.append(vr)
        if best is None or metrics.score(vr) > metrics.score(best):
            best = vr
        if vr["passed"]:
            status = "PASS"
            break
        if vr["regressed"]:
            working = Path(best["model"])
        if attempt == max_attempts:
            break
        obs = agent.observation(vr["scan"], vr["strict"], vr["placement"], vr["gate"], vr, history,
                                attempt, cpu_lat)
        choice = agent.decide(obs, history, calib is not None, cfg, log, use_llm=use_llm)
        if choice == "stop":
            break
        out = models_dir / f"v{next_idx}.onnx"
        next_idx += 1
        try:
            if choice == "freeze_shapes":
                dv = dims or {d["name"]: 1 for d in vr["scan"]["symbolic_dims"]}
                res = fixes.freeze_shapes(working, out, dv, inputs_npz=inputs,
                                          exact_tol=cfg["gate"]["exact_tol"])
                float_src = out if scan(out)["is_float"] else float_src
            else:
                preset = choice.removeprefix("quantize_")
                # always quantize from the latest float version, never re-quantize a QDQ model
                res = fixes.quantize(float_src, out, calib, preset, allow_synthetic=synthetic)
            history.append({"action": choice, "params": None, "outcome": "applied",
                            "detail": res, "version": out.stem})
            working = out
        except (fixes.FixRejected, ValueError) as e:
            history.append({"action": choice, "params": None, "outcome": "rejected",
                            "detail": str(e)[:300]})

    unresolved = []
    if status != "PASS" and best:
        if best["strict"].get("error"):
            unresolved.append({"strict_error": agent.sanitize(best["strict"]["error"], 500)})
        unresolved += [{"gate_reason": r} for r in best.get("gate", {}).get("reasons", [])]
    if status == "DEVICE_UNHEALTHY":
        unresolved.append({"device": UNHEALTHY_MSG})
    summary = {
        "status": status,
        "exit_code": EXIT[status],
        "backend": be.name,
        "synthetic": bool(synthetic),
        "model": str(model),
        "best_version": best["tag"] if best else None,
        "before": _brief(versions[0]) if versions and not versions[0]["unhealthy"] else None,
        "after": _brief(best),
        "cpu_baseline": {"latency_median_ms": cpu_lat, "latency_p95_ms": ref.get("latency_p95_ms")},
        "attempts": [{"attempt": v["attempt"], "version": v["tag"], "action": v.get("action"),
                      "strict_status": v["strict"]["status"], "share": v.get("share"),
                      "share_method": v.get("share_method"), "cos_min": v.get("cos_min"),
                      "gate_status": v.get("gate", {}).get("status"),
                      "regressed": v.get("regressed", False)} for v in versions],
        "history": history,
        "unresolved": unresolved,
        "thresholds": cfg["gate"],
        "htp_mode": cfg["qnn"]["htp_performance_mode"],
        "env": json.loads((run_dir / "env.json").read_text()),
    }
    _dump(run_dir / "summary.json", summary)
    from . import report
    report.render(run_dir)
    return EXIT[status], run_dir
