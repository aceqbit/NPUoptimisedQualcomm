import numpy as np
import onnx
import pytest
from onnx import TensorProto, helper, numpy_helper

from sentinel import env


def make_gemm_model(path, batch="batch", in_f=16, out_f=10, seed=0):
    """Tiny float classifier: (batch, in_f) -> Gemm -> Relu -> Gemm -> (batch, out_f) logits."""
    rng = np.random.default_rng(seed)
    w1 = numpy_helper.from_array(rng.standard_normal((in_f, 32)).astype(np.float32) * 0.3, "w1")
    b1 = numpy_helper.from_array(np.zeros(32, np.float32), "b1")
    w2 = numpy_helper.from_array(rng.standard_normal((32, out_f)).astype(np.float32) * 0.3, "w2")
    b2 = numpy_helper.from_array(np.zeros(out_f, np.float32), "b2")
    nodes = [helper.make_node("Gemm", ["x", "w1", "b1"], ["h"]),
             helper.make_node("Relu", ["h"], ["hr"]),
             helper.make_node("Gemm", ["hr", "w2", "b2"], ["logits"])]
    g = helper.make_graph(nodes, "tiny",
                          [helper.make_tensor_value_info("x", TensorProto.FLOAT, [batch, in_f])],
                          [helper.make_tensor_value_info("logits", TensorProto.FLOAT, [batch, out_f])],
                          [w1, b1, w2, b2])
    m = helper.make_model(g, opset_imports=[helper.make_opsetid("", 17)])
    m.ir_version = 8
    onnx.checker.check_model(m)
    onnx.save(m, str(path))
    return path


@pytest.fixture
def cfg():
    return env.load_config()


@pytest.fixture
def tiny(tmp_path):
    model = make_gemm_model(tmp_path / "tiny.onnx")
    rng = np.random.default_rng(1)
    test = tmp_path / "test.npz"
    calib = tmp_path / "calib.npz"
    np.savez(test, x=rng.standard_normal((20, 16)).astype(np.float32), synthetic=np.array(True))
    np.savez(calib, x=rng.standard_normal((400, 16)).astype(np.float32), synthetic=np.array(True))
    return {"model": model, "test": test, "calib": calib, "dir": tmp_path}
