"""CPU baseline: reference outputs and CPU latency. Also shared CPU-run helpers."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

from . import env

ORT_TO_NP = {
    "tensor(float)": np.float32, "tensor(float16)": np.float16, "tensor(double)": np.float64,
    "tensor(uint8)": np.uint8, "tensor(int8)": np.int8, "tensor(int32)": np.int32,
    "tensor(int64)": np.int64, "tensor(uint16)": np.uint16, "tensor(int16)": np.int16,
}


def load_inputs(path: str | Path) -> dict:
    with np.load(str(path), allow_pickle=False) as d:
        out = {"x": d["x"]}
        out["synthetic"] = bool(d["synthetic"]) if "synthetic" in d.files else False
        out["labels"] = d["labels"] if "labels" in d.files else None
    return out


def cpu_session(model_path: str | Path):
    import onnxruntime as ort
    return ort.InferenceSession(str(model_path), ort.SessionOptions(),
                                providers=["CPUExecutionProvider"])


def feed(sess, x: np.ndarray, i: int) -> dict:
    inp = sess.get_inputs()
    if len(inp) != 1:
        raise ValueError("MVP supports single-input models")
    dtype = ORT_TO_NP.get(inp[0].type, np.float32)
    return {inp[0].name: np.ascontiguousarray(x[i:i + 1]).astype(dtype, copy=False)}


def run_all(sess, x: np.ndarray) -> dict:
    """Run every sample (batch 1). Returns {output_name: array of shape (N, *out_shape)}."""
    names = [o.name for o in sess.get_outputs()]
    per = {n: [] for n in names}
    for i in range(x.shape[0]):
        for n, v in zip(names, sess.run(names, feed(sess, x, i))):
            per[n].append(np.asarray(v))
    return {n: np.stack(v) for n, v in per.items()}


def time_runs(sess, x: np.ndarray, warmup: int, runs: int) -> dict:
    f = feed(sess, x, 0)
    for _ in range(warmup):
        sess.run(None, f)
    ts = []
    for _ in range(runs):
        t0 = time.perf_counter()
        sess.run(None, f)
        ts.append((time.perf_counter() - t0) * 1000.0)
    return {"latency_median_ms": float(np.median(ts)),
            "latency_p95_ms": float(np.percentile(ts, 95)), "runs": runs, "warmup": warmup}


def save_outputs(path: str | Path, outs: dict) -> None:
    np.savez(str(path), **outs)


def new_run_dir(root: str | Path = env.REPO_ROOT / "runs") -> Path:
    root = Path(root)
    stamp = time.strftime("%Y%m%d-%H%M%S")
    d = root / stamp
    k = 1
    while d.exists():
        d = root / f"{stamp}-{k}"
        k += 1
    d.mkdir(parents=True)
    return d


def run_reference(model_path, inputs_path, out_npz, cfg) -> dict:
    data = load_inputs(inputs_path)
    sess = cpu_session(model_path)
    outs = run_all(sess, data["x"])
    save_outputs(out_npz, outs)
    t = time_runs(sess, data["x"], cfg["timing"]["warmup"], cfg["timing"]["runs"])
    return {"status": "OK", "outputs_path": str(out_npz), "providers": sess.get_providers(),
            "samples": int(data["x"].shape[0]), "synthetic": data["synthetic"], **t}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel.baseline")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", default=str(env.DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = env.load_config(a.config)
    run_dir = new_run_dir()
    (run_dir / "env.json").write_text(json.dumps(env.fingerprint(), indent=2))
    res = run_reference(a.model, a.inputs, run_dir / "reference_outputs.npz", cfg)
    res["run_dir"] = str(run_dir)
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
