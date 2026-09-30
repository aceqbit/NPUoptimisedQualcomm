"""Parent-side launchers for the worker subprocess. Never creates a QNN session itself."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

from . import env


def run_worker(mode, model, inputs, out_dir, cfg, *, tag="v0", config_path=None) -> dict:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_json = out_dir / f"_worker_{mode}_{tag}.json"
    if out_json.exists():
        out_json.unlink()
    timeout = cfg["loop"]["probe_timeout_s"]
    cmd = [sys.executable, "-m", "sentinel.worker", mode, "--model", str(model),
           "--inputs", str(inputs), "--out", str(out_json), "--out-dir", str(out_dir),
           "--tag", tag, "--config", str(config_path or env.DEFAULT_CONFIG)]
    try:
        cp = subprocess.run(cmd, timeout=timeout, capture_output=True, text=True, cwd=env.REPO_ROOT)
    except subprocess.TimeoutExpired:
        return {"status": "TIMEOUT", "error": f"worker '{mode}' exceeded {timeout}s and was killed"}
    if not out_json.exists():
        return {"status": "DEVICE_UNHEALTHY",
                "error": f"worker '{mode}' crashed (exit {cp.returncode}) without a result: "
                         f"{(cp.stderr or '')[-400:]}"}
    res = json.loads(out_json.read_text())
    out_json.unlink()
    if res.get("env_error"):
        raise env.EnvError(res.get("error"))
    return res


def main(argv=None) -> int:
    from .baseline import new_run_dir
    ap = argparse.ArgumentParser(prog="python -m sentinel.probe")
    ap.add_argument("--mode", choices=["strict", "placement", "reference"], default="strict")
    ap.add_argument("--model", required=True)
    ap.add_argument("--inputs", required=True)
    ap.add_argument("--config", default=str(env.DEFAULT_CONFIG))
    a = ap.parse_args(argv)
    cfg = env.load_config(a.config)
    out_dir = new_run_dir(env.REPO_ROOT / "runs" / "probes")
    try:
        res = run_worker(a.mode, a.model, a.inputs, out_dir, cfg, config_path=a.config)
    except env.EnvError as e:
        print(json.dumps({"status": "ENV_ERROR", "error": str(e)}, indent=2))
        return 2
    print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
