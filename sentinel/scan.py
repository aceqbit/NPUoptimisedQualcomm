"""Static scan of an ONNX model. Advisory only: it never decides anything."""
from __future__ import annotations

import os
from collections import Counter
from pathlib import Path

import onnx
from onnx import helper

QDQ_OPS = {"QuantizeLinear", "DequantizeLinear"}


def graph_inputs(model: onnx.ModelProto) -> list:
    """Real graph inputs (excludes initializers listed as inputs in old opsets)."""
    init = {i.name for i in model.graph.initializer}
    return [i for i in model.graph.input if i.name not in init]


def dims_of(value_info) -> list:
    """Return dims as int or str (symbolic name, '?' if unnamed)."""
    out = []
    for d in value_info.type.tensor_type.shape.dim:
        if d.HasField("dim_value"):
            out.append(d.dim_value)
        else:
            out.append(d.dim_param or "?")
    return out


def elem_dtype(value_info) -> str:
    try:
        return str(helper.tensor_dtype_to_np_dtype(value_info.type.tensor_type.elem_type))
    except Exception:  # noqa: BLE001
        return f"elem_type={value_info.type.tensor_type.elem_type}"


def _all_nodes(graph):
    for n in graph.node:
        yield n
        for a in n.attribute:
            if a.type == onnx.AttributeProto.GRAPH:
                yield from _all_nodes(a.g)
            for g in a.graphs:
                yield from _all_nodes(g)


def activation_bits(model: onnx.ModelProto) -> int | None:
    """Largest bit width among QuantizeLinear zero points (None if not quantized)."""
    inits = {i.name: i for i in model.graph.initializer}
    bits = None
    for n in model.graph.node:
        if n.op_type == "QuantizeLinear" and len(n.input) >= 3 and n.input[2] in inits:
            dt = helper.tensor_dtype_to_np_dtype(inits[n.input[2]].data_type)
            b = dt.itemsize * 8
            bits = b if bits is None else max(bits, b)
    return bits


def scan(model_path: str | Path) -> dict:
    model = onnx.load(str(model_path), load_external_data=False)
    ops = Counter(n.op_type for n in _all_nodes(model.graph))
    symbolic = []
    inputs = graph_inputs(model)
    for inp in inputs:
        for axis, d in enumerate(dims_of(inp)):
            if isinstance(d, str):
                symbolic.append({"input": inp.name, "axis": axis, "name": d})
    opset = next((o.version for o in model.opset_import if o.domain in ("", "ai.onnx")), None)
    return {
        "symbolic_dims": symbolic,
        "is_float": not any(op in ops for op in QDQ_OPS),
        "activation_bits": activation_bits(model),
        "op_counts": dict(sorted(ops.items())),
        "opset": opset,
        "node_count": sum(ops.values()),
        "inputs": [{"name": i.name, "shape": dims_of(i), "dtype": elem_dtype(i)} for i in inputs],
        "outputs": [{"name": o.name, "shape": dims_of(o), "dtype": elem_dtype(o)} for o in model.graph.output],
        "input_dtype": elem_dtype(inputs[0]) if inputs else None,
        "size_mb": round(os.path.getsize(model_path) / 1e6, 3),
    }
