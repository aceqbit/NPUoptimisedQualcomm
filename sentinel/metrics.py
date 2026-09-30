"""Accuracy comparison against the FP32 reference and the pass/fail gate."""
from __future__ import annotations

import math

import numpy as np


def _load(src) -> dict:
    if isinstance(src, dict):
        return src
    with np.load(str(src), allow_pickle=False) as d:
        return {k: d[k] for k in d.files}


def _cos(a: np.ndarray, b: np.ndarray) -> float:
    a = a.astype(np.float64).ravel()
    b = b.astype(np.float64).ravel()
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 and nb == 0:
        return 1.0
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def compare(reference, candidate) -> dict:
    """Per-output metrics. Arrays have shape (N, *per_sample_shape)."""
    ref, cand = _load(reference), _load(candidate)
    if set(ref) == set(cand):
        pairs = [(k, ref[k], cand[k]) for k in ref]
    else:  # fall back to positional matching
        pairs = [(rk, ref[rk], cand[ck]) for rk, ck in zip(ref, cand)]
    out = {}
    for name, r, c in pairs:
        if r.shape != c.shape:
            out[name] = {"cosine_min": None, "max_abs_err": None, "top1_agreement": None,
                         "nonfinite": False, "error": f"shape mismatch {r.shape} vs {c.shape}"}
            continue
        nonfinite = not bool(np.isfinite(c).all())
        cos = [_cos(r[i], c[i]) for i in range(r.shape[0])]
        m = {
            "cosine_min": float(np.nanmin(cos)) if not nonfinite else float("nan"),
            "max_abs_err": float(np.max(np.abs(r.astype(np.float64) - c.astype(np.float64))))
            if not nonfinite else float("inf"),
            "top1_agreement": None,
            "nonfinite": nonfinite,
        }
        per = r.shape[1:]
        if len(per) == 2 and per[0] == 1 and per[1] > 1:  # 2-D classifier logits (1, K)
            m["top1_agreement"] = float(np.mean(r[:, 0].argmax(-1) == c[:, 0].argmax(-1)))
        out[name] = m
    return out


def aggregate(metrics: dict) -> dict:
    cos = [m["cosine_min"] for m in metrics.values()]
    mae = [m["max_abs_err"] for m in metrics.values()]
    top = [m["top1_agreement"] for m in metrics.values() if m["top1_agreement"] is not None]
    bad = any(v is None or (isinstance(v, float) and math.isnan(v)) for v in cos)
    return {
        "cos_min": None if (bad or not cos) else min(cos),
        "max_abs_err": None if (not mae or any(v is None for v in mae)) else max(mae),
        "top1_agreement": min(top) if top else None,
    }


def gate(metrics: dict | None, share: float | None, cfg: dict) -> dict:
    g = cfg["gate"]
    reasons, acc_fail, share_fail = [], False, False
    if not metrics:
        acc_fail = True
        reasons.append("no candidate outputs to compare")
    for name, m in (metrics or {}).items():
        if m.get("error"):
            acc_fail = True
            reasons.append(f"{name}: {m['error']}")
            continue
        if m["nonfinite"]:
            acc_fail = True
            reasons.append(f"{name}: NaN/Inf in outputs")
            continue
        if not (m["cosine_min"] >= g["cos_min"]):
            acc_fail = True
            reasons.append(f"{name}: cosine_min {m['cosine_min']:.6f} < {g['cos_min']}")
        if m["top1_agreement"] is not None and not (m["top1_agreement"] >= g["top1_min"]):
            acc_fail = True
            reasons.append(f"{name}: top1_agreement {m['top1_agreement']:.4f} < {g['top1_min']}")
    if share is None:
        share_fail = True
        reasons.append("share unavailable")
    elif not (share >= g["share_min"]):
        share_fail = True
        reasons.append(f"share {share:.4f} < {g['share_min']}")
    status = "ACCURACY_FAIL" if acc_fail else ("SHARE_LOW" if share_fail else "OK")
    return {"passed": status == "OK", "reasons": reasons, "status": status}


def score(vr: dict) -> tuple:
    """Ordering key for picking the best version (higher is better)."""
    cos = vr.get("cos_min")
    lat = vr.get("latency_median_ms")
    return (bool(vr.get("passed")),
            round(cos, 4) if cos is not None and not math.isnan(cos) else -math.inf,
            vr.get("share") or 0.0,
            -(lat if lat is not None else math.inf))
