"""CLI: python -m sentinel run --model M --inputs I.npz [--calib C.npz] [--backend qnn|sim] ..."""
from __future__ import annotations

import argparse
import json
import sys

from . import env


def _dims(items):
    out = {}
    for it in items or []:
        k, _, v = it.partition("=")
        if not v.isdigit():
            raise SystemExit(f"--dim expects NAME=INT, got {it!r}")
        out[k] = int(v)
    return out or None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m sentinel")
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("--model", required=True)
    r.add_argument("--inputs", required=True)
    r.add_argument("--calib")
    r.add_argument("--backend", choices=["qnn", "sim"], default="qnn")
    r.add_argument("--dim", action="append", help="symbolic dim value, e.g. batch=1")
    r.add_argument("--no-llm", action="store_true")
    r.add_argument("--max-attempts", type=int)
    r.add_argument("--config", default=str(env.DEFAULT_CONFIG))
    a = ap.parse_args(argv)

    if a.backend == "sim":
        print("=" * 64 + "\n  SIMULATED BACKEND: results are NOT measured on an NPU\n" + "=" * 64)
    from .loop import run
    try:
        code, run_dir = run(a.model, a.inputs, a.calib, backend=a.backend, dims=_dims(a.dim),
                            use_llm=not a.no_llm, max_attempts=a.max_attempts,
                            config_path=a.config)
    except env.EnvError as e:
        print(f"ENVIRONMENT ERROR: {e}", file=sys.stderr)
        return 2
    summary = json.loads((run_dir / "summary.json").read_text())
    print(json.dumps({k: summary[k] for k in ("status", "backend", "synthetic", "best_version",
                                              "before", "after", "attempts")}, indent=2))
    print(f"run dir: {run_dir}\nreport:  {run_dir / 'report.html'}")
    return code


if __name__ == "__main__":
    sys.exit(main())
