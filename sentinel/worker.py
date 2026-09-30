"""Subprocess worker. The ONLY place a QNN session is ever created.

python -m sentinel.worker {strict|placement|reference} --model M --inputs I.npz --out O.json --out-dir D
Catches every exception and writes {"status", "error"} using the Sentinel taxonomy.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
import traceback
from pathlib import Path

from . import baseline, env

QNN_EP = env.QNN_EP


class StageError(Exception):
    def __init__(self, status: str, err: BaseException):
        super().__init__(f"{type(err).__name__}: {err}")
        self.status = status


def make_qnn_session(model_path, *, strict: bool, profiling: bool, ctx_path, cfg,
                     profile_prefix=None):
    """All QNN provider construction lives here.

    Written for the classic provider-bridge QNN EP (onnxruntime-qnn 1.x), where the EP is
    listed by ort.get_available_providers(). # TODO(verify) on the Snapdragon device: if the
    installed onnxruntime-qnn is a 2.x plugin EP, registration differs; adapt only this function.
    """
    import onnxruntime as ort

    if QNN_EP not in ort.get_available_providers():
        raise env.EnvError(f"{QNN_EP} not available; providers={ort.get_available_providers()}")
    so = ort.SessionOptions()
    if strict:
        so.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
    if ctx_path is not None:
        # TODO(verify): EP-context dump keys for the installed version.
        so.add_session_config_entry("ep.context_enable", "1")
        so.add_session_config_entry("ep.context_file_path", str(ctx_path))
    if profiling and profile_prefix is not None:
        so.enable_profiling = True
        so.profile_file_prefix = str(profile_prefix)
    opts = {
        "backend_path": cfg["qnn"]["backend_path"],  # TODO(verify) option name
        "htp_performance_mode": cfg["qnn"]["htp_performance_mode"],  # TODO(verify) option name
    }
    if profiling and profile_prefix is not None:
        opts["profiling_level"] = "basic"  # TODO(verify) option name
        opts["profiling_file_path"] = f"{profile_prefix}_qnn.csv"  # TODO(verify) option name
    providers = [(QNN_EP, opts)] if strict else [(QNN_EP, opts), "CPUExecutionProvider"]
    return ort.InferenceSession(str(model_path), so, providers=providers)


def share_from_epcontext(ctx_path, model_path) -> dict | None:
    """Share = 1 - (non-EPContext nodes left in the generated model / nodes in the input model).

    Nodes not absorbed into an EPContext node are the ones left for another EP (CPU).
    # TODO(verify) against the installed version's EP-context output format.
    """
    import onnx
    p = Path(ctx_path)
    if not p.exists():
        return None
    ctx = onnx.load(str(p), load_external_data=False)
    n_ep = sum(1 for n in ctx.graph.node if n.op_type == "EPContext")
    n_other = len(ctx.graph.node) - n_ep
    n_orig = len(onnx.load(str(model_path), load_external_data=False).graph.node)
    share = 0.0 if n_ep == 0 else max(0.0, (n_orig - n_other) / max(n_orig, 1))
    return {"share": share, "share_method": "epcontext",
            "epcontext_nodes": n_ep, "other_nodes": n_other, "input_nodes": n_orig}


def share_from_ort_profile(profile_json) -> dict | None:
    """Fraction of node kernel time attributed to the QNN EP in ORT's own profile JSON."""
    events = json.loads(Path(profile_json).read_text())
    tot = qnn = 0.0
    for e in events:
        if e.get("cat") == "Node" and e.get("name", "").endswith("_kernel_time"):
            d = float(e.get("dur", 0))
            tot += d
            if e.get("args", {}).get("provider") == QNN_EP:
                qnn += d
    if tot <= 0:
        return None
    return {"share": qnn / tot, "share_method": "ort_profile_time"}


def _create(model, **kw):
    t0 = time.perf_counter()
    try:
        sess = make_qnn_session(model, **kw)
    except env.EnvError:
        raise
    except Exception as e:  # noqa: BLE001
        raise StageError("COMPILE_ERROR", e) from e
    return sess, time.perf_counter() - t0


def do_strict(model, inputs, out_dir, tag, cfg) -> dict:
    data = baseline.load_inputs(inputs)
    sess, ct = _create(model, strict=True, profiling=False, ctx_path=None, cfg=cfg)
    try:
        sess.run(None, baseline.feed(sess, data["x"], 0))
    except Exception as e:  # noqa: BLE001
        raise StageError("RUNTIME_ERROR", e) from e
    return {"status": "OK", "error": None, "compile_time_s": ct, "providers": sess.get_providers(),
            "share": 1.0, "share_method": "strict_pass"}


def do_placement(model, inputs, out_dir, tag, cfg) -> dict:
    out_dir = Path(out_dir)
    data = baseline.load_inputs(inputs)
    ctx_path = out_dir / f"ctx_{tag}.onnx"
    prefix = out_dir / f"profile_{tag}"
    notes = []
    try:
        sess, ct = _create(model, strict=False, profiling=True, ctx_path=ctx_path, cfg=cfg,
                           profile_prefix=prefix)
    except StageError as e:
        notes.append(f"session with ep.context_enable failed ({e}); retried without it")
        sess, ct = _create(model, strict=False, profiling=True, ctx_path=None, cfg=cfg,
                           profile_prefix=prefix)
    try:
        outs = baseline.run_all(sess, data["x"])
        timing = baseline.time_runs(sess, data["x"], cfg["timing"]["warmup"], cfg["timing"]["runs"])
    except Exception as e:  # noqa: BLE001
        raise StageError("RUNTIME_ERROR", e) from e
    profile_file = sess.end_profiling()
    out_npz = out_dir / f"outputs_{tag}.npz"
    baseline.save_outputs(out_npz, outs)

    share = None
    try:
        share = share_from_epcontext(ctx_path, model)
    except Exception as e:  # noqa: BLE001
        notes.append(f"epcontext share failed: {type(e).__name__}: {e}")
    if share is None and profile_file:
        try:
            share = share_from_ort_profile(profile_file)
        except Exception as e:  # noqa: BLE001
            notes.append(f"profile share failed: {type(e).__name__}: {e}")
    share = share or {"share": None, "share_method": "unavailable"}
    return {"status": "OK", "error": None, **share, "compile_time_s": ct, "fresh_compile": True,
            "latency_median_ms": timing["latency_median_ms"],
            "latency_p95_ms": timing["latency_p95_ms"],
            "outputs_path": str(out_npz), "providers": sess.get_providers(),
            "profile_file": profile_file, "notes": notes}


def do_reference(model, inputs, out_dir, tag, cfg) -> dict:
    try:
        return baseline.run_reference(model, inputs, Path(out_dir) / "reference_outputs.npz", cfg)
    except Exception as e:  # noqa: BLE001
        raise StageError("RUNTIME_ERROR", e) from e


MODES = {"strict": do_strict, "placement": do_placement, "reference": do_reference}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.worker")
    ap.add_argument("mode", choices=sorted(MODES))
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--tag", default="v0")
    ap.add_argument("--config", default=str(env.DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    out_dir = Path(a.out_dir or Path(a.out).parent)
    out_dir.mkdir(parents=True, exist_ok=True)
    try:
        res = MODES[a.mode](a.model, a.inputs, out_dir, a.tag, env.load_config(a.config))
    except env.EnvError as e:
        res = {"status": "COMPILE_ERROR", "env_error": True, "error": str(e)[:500]}
    except StageError as e:
        res = {"status": e.status, "error": str(e)[:500]}
    except Exception as e:  # noqa: BLE001
        res = {"status": "RUNTIME_ERROR",
               "error": (f"{type(e).__name__}: {e}\n" + traceback.format_exc())[:500]}
    Path(a.out).write_text(json.dumps(res, indent=2, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
