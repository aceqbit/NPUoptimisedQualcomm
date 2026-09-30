"""Fetch or copy an ONNX model, validate it, and print a summary. No model URL is hard-coded."""
import argparse
import json
import shutil
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import onnx  # noqa: E402

from sentinel.scan import graph_inputs, scan  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--url")
    src.add_argument("--path")
    ap.add_argument("--out", default="models/base.onnx")
    a = ap.parse_args(argv)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if a.url:
        print(f"downloading {a.url}")
        with urllib.request.urlopen(a.url, timeout=120) as r, open(out, "wb") as f:
            shutil.copyfileobj(r, f)
    else:
        shutil.copyfile(a.path, out)

    model = onnx.load(str(out))
    onnx.checker.check_model(model)
    if len(graph_inputs(model)) != 1:
        out.unlink()
        print("ERROR: MVP supports single-input models", file=sys.stderr)
        return 2
    s = scan(out)
    print(json.dumps({
        "path": str(out),
        "inputs": s["inputs"],
        "outputs": s["outputs"],
        "symbolic_dims": s["symbolic_dims"],
        "opset": s["opset"],
        "node_count": s["node_count"],
        "has_qdq_nodes": not s["is_float"],
        "size_mb": s["size_mb"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
