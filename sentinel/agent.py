"""Fix-selection agent: whitelisted observation, rule-computed legal actions, rule/LLM choice, log."""
from __future__ import annotations

import json
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

ORDER = ["freeze_shapes", "quantize_a8w8", "quantize_a16w8", "stop"]
OBS_KEYS = ["strict_status", "strict_error", "share", "share_method", "cos_min", "top1_agreement",
            "max_abs_err", "latency_median_ms", "cpu_latency_ms", "symbolic_dims", "is_float",
            "attempt", "tried", "last_gate_status"]
PROMPT_PATH = Path(__file__).parent / "prompts" / "agent_system.txt"
_CTRL = re.compile(r"[\x00-\x08\x0b-\x1f\x7f]")


def sanitize(text, limit=300):
    if text is None:
        return None
    return _CTRL.sub(" ", str(text))[:limit]


def observation(scan, strict, placement, gate, agg, history, attempt, cpu_latency_ms) -> dict:
    return {
        "strict_status": strict.get("status"),
        "strict_error": sanitize(strict.get("error")),
        "share": placement.get("share"),
        "share_method": placement.get("share_method"),
        "cos_min": agg.get("cos_min"),
        "top1_agreement": agg.get("top1_agreement"),
        "max_abs_err": agg.get("max_abs_err"),
        "latency_median_ms": placement.get("latency_median_ms"),
        "cpu_latency_ms": cpu_latency_ms,
        "symbolic_dims": [sanitize(d["name"], 64) for d in scan["symbolic_dims"]],
        "is_float": scan["is_float"],
        "attempt": attempt,
        "tried": [h["action"] for h in history],
        "last_gate_status": gate.get("status"),
    }


def legal_actions(obs, history, calib_available: bool) -> list:
    tried = {(h["action"], json.dumps(h.get("params"), sort_keys=True)) for h in history}
    tried_names = {h["action"] for h in history}
    legal = []
    if obs["symbolic_dims"] and "freeze_shapes" not in tried_names:
        legal.append("freeze_shapes")
    if obs["is_float"] and calib_available and "quantize_a8w8" not in tried_names:
        legal.append("quantize_a8w8")
    if ((not obs["is_float"] or "quantize_a8w8" in tried_names) and calib_available
            and obs.get("last_gate_status") == "ACCURACY_FAIL"
            and "quantize_a16w8" not in tried_names):
        legal.append("quantize_a16w8")
    legal = [a for a in legal if (a, json.dumps(None)) not in tried]
    return legal + ["stop"]


def rationale_for(action, obs) -> str:
    if action == "freeze_shapes":
        return (f"input has symbolic dims {obs['symbolic_dims']}; strict probe status "
                f"{obs['strict_status']}; freezing is exact, so try it before any lossy transform")
    if action == "quantize_a8w8":
        return (f"graph is float (is_float=True), share={obs['share']}, strict status "
                f"{obs['strict_status']}; HTP needs a quantized graph, so try A8W8 QDQ first")
    if action == "quantize_a16w8":
        return (f"last gate status {obs['last_gate_status']} with cos_min={obs['cos_min']}, "
                f"top1={obs['top1_agreement']}; retry with 16-bit activations")
    return (f"no fix left for strict status {obs['strict_status']}, share={obs['share']}, "
            f"cos_min={obs['cos_min']}; tried {obs['tried']}")


def rules_choose(legal) -> str:
    return next(a for a in ORDER if a in legal)


def llm_choose(obs, legal, cfg) -> tuple[dict | None, str | None]:
    """Ask a local OpenAI-compatible endpoint to pick one legal action.

    Returns ({"action", "rationale"}, None) or (None, error). Sends only the whitelisted
    observation and the legal actions list.
    """
    c = cfg.get("llm", {})
    if not c.get("enabled"):
        return None, None
    payload = {
        "model": c.get("model", ""),
        "temperature": 0,
        "messages": [
            {"role": "system", "content": PROMPT_PATH.read_text(encoding="utf-8")},
            {"role": "user", "content": json.dumps(
                {"observation": {k: obs.get(k) for k in OBS_KEYS}, "legal_actions": legal})},
        ],
    }
    req = urllib.request.Request(c["base_url"].rstrip("/") + "/chat/completions",
                                 data=json.dumps(payload).encode(),
                                 headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=c.get("timeout_s", 20)) as r:
            body = json.loads(r.read().decode("utf-8"))
        text = body["choices"][0]["message"]["content"]
    except Exception as e:  # noqa: BLE001
        return None, f"http: {type(e).__name__}: {str(e)[:200]}"
    m = re.search(r"\{.*?\}", text, re.S)
    if not m:
        return None, "no JSON object in reply"
    try:
        reply = json.loads(m.group(0))
    except json.JSONDecodeError as e:
        return None, f"invalid JSON: {e}"
    action, rat = reply.get("action"), reply.get("rationale")
    if action not in legal:
        return None, f"illegal action {sanitize(action, 50)!r}"
    if not isinstance(rat, str) or len(rat) > 300:
        return None, "rationale missing or longer than 300 chars"
    return {"action": action, "rationale": rat}, None


class DecisionLog:
    def __init__(self, path):
        self.path = Path(path)

    def append(self, **rec):
        rec.setdefault("timestamp", time.strftime("%Y-%m-%dT%H:%M:%S"))
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")


def decide(obs, history, calib_available, cfg, log: DecisionLog, use_llm=True) -> str:
    legal = legal_actions(obs, history, calib_available)
    real = [a for a in legal if a != "stop"]
    choice, source, err = None, "rules", None
    if use_llm and len(real) > 1:
        reply, err = llm_choose(obs, legal, cfg)
        if reply:
            choice, rationale, source = reply["action"], reply["rationale"], "llm"
    if choice is None:
        choice = rules_choose(legal)
        rationale = rationale_for(choice, obs)
    rec = {"attempt": obs["attempt"], "observation": obs, "legal_actions": legal,
           "chosen": choice, "rationale": rationale, "source": source}
    if err:
        rec["llm_error"] = err
    log.append(**rec)
    return choice
