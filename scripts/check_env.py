"""Print the environment fingerprint as JSON. Exit 2 if not ARM64."""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from sentinel.env import fingerprint  # noqa: E402

fp = fingerprint()
print(json.dumps(fp, indent=2))
if fp["machine"].upper() not in ("ARM64", "AARCH64"):
    print(f"ERROR: platform.machine() is {fp['machine']!r}, expected ARM64. "
          "Sentinel's QNN path needs native ARM64 Python on a Snapdragon X device.", file=sys.stderr)
    sys.exit(2)
