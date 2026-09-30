"""Build disjoint calibration and test input sets (NCHW float32) for a single-input model."""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import onnx  # noqa: E402

from sentinel.scan import dims_of, graph_inputs  # noqa: E402

IMAGENET_MEAN = [0.485, 0.456, 0.406]  # documented default, override with --mean
IMAGENET_STD = [0.229, 0.224, 0.225]  # documented default, override with --std
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def model_chw(model_path):
    inputs = graph_inputs(onnx.load(str(model_path), load_external_data=False))
    if len(inputs) != 1:
        raise SystemExit("ERROR: MVP supports single-input models")
    dims = dims_of(inputs[0])
    if len(dims) != 4 or not all(isinstance(d, int) for d in dims[1:]):
        raise SystemExit(f"ERROR: expected NCHW input with fixed C,H,W; got {dims}")
    return dims[1], dims[2], dims[3]


def load_image(path, h, w, mean, std):
    from PIL import Image
    im = Image.open(path).convert("RGB")
    # resize shorter side to 256/224 * target, then center crop (ImageNet convention)
    scale = max(h, w) * 256 / 224 / min(im.size)
    im = im.resize((max(w, round(im.size[0] * scale)), max(h, round(im.size[1] * scale))),
                   Image.BILINEAR)
    left, top = (im.size[0] - w) // 2, (im.size[1] - h) // 2
    im = im.crop((left, top, left + w, top + h))
    a = np.asarray(im, dtype=np.float32) / 255.0
    a = (a - np.array(mean, np.float32)) / np.array(std, np.float32)
    return a.transpose(2, 0, 1)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--images")
    ap.add_argument("--npy", help="directory of pre-made .npy arrays (C,H,W) if Pillow is unavailable")
    ap.add_argument("--model", required=True)
    ap.add_argument("--calib", type=int, default=100)
    ap.add_argument("--test", type=int, default=100)
    ap.add_argument("--out", default="data")
    ap.add_argument("--mean", type=float, nargs=3, default=IMAGENET_MEAN)
    ap.add_argument("--std", type=float, nargs=3, default=IMAGENET_STD)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--allow-synthetic", action="store_true")
    a = ap.parse_args(argv)

    c, h, w = model_chw(a.model)
    rng = np.random.default_rng(a.seed)
    synthetic = False
    if a.images:
        files = sorted(p for p in Path(a.images).iterdir() if p.suffix.lower() in IMG_EXT)
        arrays = [load_image(p, h, w, a.mean, a.std) for p in files]
    elif a.npy:
        arrays = [np.load(p).astype(np.float32) for p in sorted(Path(a.npy).glob("*.npy"))]
    elif a.allow_synthetic:
        print("WARNING: SYNTHETIC DATA. Never use for accuracy claims.", file=sys.stderr)
        synthetic = True
        arrays = list(rng.standard_normal((a.calib + a.test, c, h, w)).astype(np.float32))
    else:
        print("ERROR: give --images or --npy (or --allow-synthetic for plumbing tests only)",
              file=sys.stderr)
        return 2

    arrays = [x for x in arrays if x.shape == (c, h, w)]
    n = len(arrays)
    if n < 2:
        print(f"ERROR: need at least 2 usable samples, got {n}", file=sys.stderr)
        return 2
    order = rng.permutation(n)
    n_test = min(a.test, n // 2)
    n_calib = min(a.calib, n - n_test)
    test_idx, calib_idx = order[:n_test], order[n_test:n_test + n_calib]
    assert not set(test_idx) & set(calib_idx)
    x = np.stack(arrays).astype(np.float32)

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "calib_inputs.npz", x=x[calib_idx], synthetic=np.array(synthetic))
    np.savez(out / "test_inputs.npz", x=x[test_idx], synthetic=np.array(synthetic))
    print(f"samples available={n} calib={len(calib_idx)} test={len(test_idx)} "
          f"shape=(N,{c},{h},{w}) synthetic={synthetic} out={out}")
    if n_calib < a.calib or n_test < a.test:
        print(f"NOTE: requested calib={a.calib} test={a.test}; fewer images were available.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
