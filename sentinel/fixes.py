"""Deterministic, verified model transforms. The agent only chooses; this code applies."""
from __future__ import annotations

import os
import tempfile
from pathlib import Path

import numpy as np
import onnx

from . import baseline, env
from .scan import graph_inputs


class FixRejected(RuntimeError):
    """The transform ran but failed its own verification; model_out was removed."""


def freeze_graph(model: onnx.ModelProto, dim_values: dict) -> onnx.ModelProto:
    """Set symbolic input dims to fixed values; clear symbolic dims elsewhere and re-infer."""
    missing = []
    for inp in graph_inputs(model):
        for d in inp.type.tensor_type.shape.dim:
            if d.HasField("dim_value"):
                continue
            name = d.dim_param
            if name not in dim_values:
                missing.append(f"{inp.name}:{name or '?'}")
                continue
            d.ClearField("dim_param")
            d.dim_value = int(dim_values[name])
    if missing:
        raise ValueError(f"no value given for symbolic dims: {', '.join(missing)} "
                         "(pass --dim NAME=VALUE)")
    for out in model.graph.output:
        for d in out.type.tensor_type.shape.dim:
            if not d.HasField("dim_value"):
                v = dim_values.get(d.dim_param)
                d.ClearField("dim_param")
                if v is not None:
                    d.dim_value = int(v)
    del model.graph.value_info[:]
    model = onnx.shape_inference.infer_shapes(model)
    onnx.checker.check_model(model)
    return model


def _max_abs_diff(model_a, model_b, inputs_npz) -> float:
    x = baseline.load_inputs(inputs_npz)["x"]
    ra = baseline.run_all(baseline.cpu_session(model_a), x)
    rb = baseline.run_all(baseline.cpu_session(model_b), x)
    return max(float(np.max(np.abs(ra[k].astype(np.float64) - rb[k].astype(np.float64))))
               for k in ra)


def freeze_shapes(model_in, model_out, dim_values: dict | None, *, inputs_npz, exact_tol: float) -> dict:
    """Exact transform. Verified on CPU against the input model: max_abs_err <= exact_tol."""
    dim_values = dict(dim_values or {"batch": 1})
    model = freeze_graph(onnx.load(str(model_in)), dim_values)
    onnx.save(model, str(model_out))
    err = _max_abs_diff(model_in, model_out, inputs_npz)
    if not err <= exact_tol:
        os.remove(model_out)
        raise FixRejected(f"freeze_shapes changed outputs: max_abs_err={err} > exact_tol={exact_tol}")
    return {"action": "freeze_shapes", "params": dim_values, "verified_max_abs_err": err}


class NpzReader:
    """CalibrationDataReader over calib_inputs.npz (batch 1 per sample)."""

    def __init__(self, x: np.ndarray, input_name: str):
        self._it = iter([{input_name: x[i:i + 1].astype(np.float32)} for i in range(len(x))])

    def get_next(self):
        return next(self._it, None)


PRESETS = {"a8w8": ("QUInt8", "QUInt8"), "a16w8": ("QUInt16", "QUInt8")}


def quantize(model_in, model_out, calib_npz, preset: str = "a8w8", *, allow_synthetic=False) -> dict:
    """QDQ static quantization with calibration data.

    The installed onnxruntime (checked on the dev machine, 1.30.0) has no
    onnxruntime.quantization.execution_provider.qnn helper, so quantize_static(QuantFormat.QDQ)
    is used. # TODO(verify) on device: if onnxruntime-qnn ships get_qnn_qdq_config, prefer it.
    """
    try:
        from onnxruntime.quantization import (CalibrationMethod, QuantFormat, QuantType,
                                              quant_pre_process, quantize_static)
    except Exception as e:  # noqa: BLE001
        raise env.EnvError(
            f"onnxruntime.quantization does not import here ({e}). Workaround: quantize in an x64 "
            "Python env and pass the resulting .onnx to the ARM64 probe.") from e
    data = baseline.load_inputs(calib_npz)
    if data["synthetic"] and not allow_synthetic:
        raise FixRejected("calibration data is SYNTHETIC; pass --allow-synthetic to use it anyway")
    act, wt = PRESETS[preset]
    name = graph_inputs(onnx.load(str(model_in), load_external_data=False))[0].name
    notes = []
    with tempfile.TemporaryDirectory() as td:
        pre = Path(td) / "pre.onnx"
        src = model_in
        try:
            quant_pre_process(str(model_in), str(pre))
            src = pre
        except Exception as e:  # noqa: BLE001
            notes.append(f"quant_pre_process failed, quantizing unpreprocessed model: "
                         f"{type(e).__name__}: {str(e)[:200]}")
        extra = {"ActivationSymmetric": False, "WeightSymmetric": False}
        if act == "QUInt16":
            extra["UseQDQContribOps"] = True  # 16-bit Q/DQ on opsets < 21
        quantize_static(str(src), str(model_out), NpzReader(data["x"], name),
                        quant_format=QuantFormat.QDQ, activation_type=getattr(QuantType, act),
                        weight_type=getattr(QuantType, wt), per_channel=False,
                        calibrate_method=CalibrationMethod.MinMax, extra_options=extra)
    onnx.checker.check_model(onnx.load(str(model_out)))
    res = {"action": f"quantize_{preset}", "params": {"activations": act, "weights": wt},
           "calib_samples": int(len(data["x"])), "notes": notes}
    if data["synthetic"]:
        res["synthetic_calibration"] = True
    return res
