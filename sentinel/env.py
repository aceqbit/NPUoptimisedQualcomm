"""Environment fingerprint and config loading."""
from __future__ import annotations

import importlib
import importlib.metadata as md
import json
import platform
import subprocess
import sys
import tomllib
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_CONFIG = REPO_ROOT / "config.toml"
QNN_EP = "QNNExecutionProvider"


class EnvError(RuntimeError):
    """Raised when the environment cannot run the requested backend."""


def load_config(path: str | Path | None = None) -> dict:
    with open(path or DEFAULT_CONFIG, "rb") as f:
        return tomllib.load(f)


def _pkg_version(name: str) -> str | None:
    try:
        return md.version(name)
    except md.PackageNotFoundError:
        return None


def _import_status(mod: str) -> str:
    try:
        importlib.import_module(mod)
        return "OK"
    except Exception as e:  # noqa: BLE001 - report exact error
        return f"{type(e).__name__}: {e}"


def npu_devices() -> list | dict:
    """Query PnP devices that look like an NPU. Empty list is allowed."""
    if platform.system() != "Windows":
        return []
    cmd = (
        "Get-PnpDevice | Where-Object {$_.FriendlyName -match 'neural|NPU|hexagon|qualcomm'} "
        "| Select FriendlyName,Status | ConvertTo-Json"
    )
    try:
        out = subprocess.run(
            ["powershell", "-NoProfile", "-Command", cmd],
            capture_output=True, text=True, timeout=60,
        ).stdout.strip()
    except Exception as e:  # noqa: BLE001
        return {"error": f"{type(e).__name__}: {e}"}
    if not out:
        return []
    try:
        data = json.loads(out)
    except json.JSONDecodeError:
        return {"error": "unparseable output", "raw": out[:500]}
    return data if isinstance(data, list) else [data]


def fingerprint() -> dict:
    try:
        import onnxruntime as ort
        providers = ort.get_available_providers()
        ort_version = ort.__version__
    except Exception as e:  # noqa: BLE001
        providers, ort_version = [], f"import failed: {e}"
    return {
        "machine": platform.machine(),
        "processor": platform.processor(),
        "python": sys.version,
        "windows_build": platform.version(),
        "versions": {
            "numpy": _pkg_version("numpy"),
            "onnx": _pkg_version("onnx"),
            "onnxruntime": ort_version,
            "onnxruntime-qnn": _pkg_version("onnxruntime-qnn"),
        },
        "providers": providers,
        "qnn_available": QNN_EP in providers,
        "npu_devices": npu_devices(),
        "imports": {m: _import_status(m) for m in
                    ["numpy", "onnx", "onnxruntime", "jinja2", "rich", "pytest", "PIL"]},
    }
