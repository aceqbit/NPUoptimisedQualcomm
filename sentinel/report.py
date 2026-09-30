"""Self-contained HTML report rendered from run files only."""
from __future__ import annotations

import json
from pathlib import Path

from jinja2 import Environment

TEMPLATE = """<!doctype html><html><head><meta charset="utf-8"><title>Sentinel report {{ run_id }}</title>
<style>
body{font-family:system-ui,sans-serif;margin:2rem auto;max-width:1100px;padding:0 1rem;color:#1a1a1a}
table{border-collapse:collapse;margin:.5rem 0 1.5rem;width:100%}td,th{border:1px solid #ccc;padding:4px 8px;text-align:left;vertical-align:top;font-size:13px}
th{background:#f2f2f2}.banner{padding:1rem;font-size:1.4rem;font-weight:700;margin-bottom:1rem}
.red{background:#c00;color:#fff}.ok{background:#e7f5e7;border:1px solid #6a6}
.PASS{color:#070;font-weight:700}.fail{color:#b00;font-weight:700}pre{white-space:pre-wrap;background:#f7f7f7;padding:.5rem}
</style></head><body>
<h1>Sentinel report</h1>
{% if s.backend == "SIMULATED" %}<div class="banner red">SIMULATED BACKEND: these numbers were NOT measured on an NPU.</div>{% endif %}
{% if s.synthetic %}<div class="banner red">SYNTHETIC DATA: not valid for accuracy claims.</div>{% endif %}
{% if s.backend != "SIMULATED" and not s.synthetic %}<div class="banner ok">Backend {{ s.backend }}, measured on device.</div>{% endif %}

<h2>1. Environment</h2>
<table>
<tr><th>Machine</th><td>{{ e.machine }}</td></tr>
<tr><th>Windows build</th><td>{{ e.windows_build }}</td></tr>
<tr><th>Python</th><td>{{ e.python }}</td></tr>
<tr><th>onnxruntime</th><td>{{ e.versions.onnxruntime }}</td></tr>
<tr><th>onnxruntime-qnn</th><td>{{ e.versions["onnxruntime-qnn"] }}</td></tr>
<tr><th>Providers</th><td>{{ e.providers | join(", ") }}</td></tr>
<tr><th>NPU driver entries</th><td>{{ e.npu_devices }}</td></tr>
<tr><th>htp_performance_mode (config)</th><td>{{ s.htp_mode }}</td></tr>
</table>

<h2>2. Before and after</h2>
<p>Final status: <span class="{{ 'PASS' if s.status == 'PASS' else 'fail' }}">{{ s.status }}</span>. Best version: {{ s.best_version }}.</p>
<table><tr><th>Metric</th><th>Before (v0)</th><th>After (best)</th></tr>
{% for k in ["strict_status","share","share_method","cos_min","max_abs_err","top1_agreement","latency_median_ms","latency_p95_ms","gate_status","providers"] %}
<tr><th>{{ k }}</th><td>{{ (s.before or {}).get(k) }}</td><td>{{ (s.after or {}).get(k) }}</td></tr>{% endfor %}
<tr><th>CPU baseline latency median / p95 (ms)</th><td colspan="2">{{ s.cpu_baseline.latency_median_ms }} / {{ s.cpu_baseline.latency_p95_ms }}</td></tr>
</table>
{% if slower %}<p class="fail">The final model is slower than the CPU baseline ({{ s.after.latency_median_ms }} ms vs {{ s.cpu_baseline.latency_median_ms }} ms median).</p>{% endif %}
<p>Gate thresholds (config defaults): {{ s.thresholds }}</p>

<h2>3. Attempts</h2>
<table><tr><th>#</th><th>Version</th><th>Action applied</th><th>Strict</th><th>Share</th><th>Share method</th><th>cosine_min</th><th>Gate</th><th>Regressed</th></tr>
{% for a in s.attempts %}<tr><td>{{ a.attempt }}</td><td>{{ a.version }}</td><td>{{ a.action }}</td><td>{{ a.strict_status }}</td><td>{{ a.share }}</td><td>{{ a.share_method }}</td><td>{{ a.cos_min }}</td><td>{{ a.gate_status }}</td><td>{{ a.regressed }}</td></tr>{% endfor %}
</table>
{% if rejected %}<p>Rejected fixes:</p><ul>{% for h in rejected %}<li>{{ h.action }}: {{ h.detail }}</li>{% endfor %}</ul>{% endif %}

<h2>4. Agent decisions</h2>
<table><tr><th>Attempt</th><th>Legal actions</th><th>Chosen</th><th>Rationale</th><th>Source</th><th>LLM error</th></tr>
{% for d in decisions %}<tr><td>{{ d.attempt }}</td><td>{{ d.legal_actions | join(", ") }}</td><td>{{ d.chosen }}</td><td>{{ d.rationale }}</td><td>{{ d.source }}</td><td>{{ d.llm_error or "" }}</td></tr>{% endfor %}
</table>

{% if s.status != "PASS" %}<h2>5. Unresolved problems</h2>
{% for u in s.unresolved %}<pre>{{ u }}</pre>{% endfor %}
<p>Tried: {{ s.history | map(attribute="action") | join(", ") or "nothing" }}</p>{% endif %}
</body></html>"""


def render(run_dir) -> Path:
    run_dir = Path(run_dir)
    s = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    dpath = run_dir / "decisions.jsonl"
    decisions = [json.loads(l) for l in dpath.read_text(encoding="utf-8").splitlines() if l.strip()] \
        if dpath.exists() else []
    after = s.get("after") or {}
    cpu = (s.get("cpu_baseline") or {}).get("latency_median_ms")
    slower = after.get("latency_median_ms") is not None and cpu is not None \
        and after["latency_median_ms"] > cpu
    s["htp_mode"] = s.get("htp_mode") or "see config.toml [qnn]"
    html = Environment(autoescape=True).from_string(TEMPLATE).render(
        s=s, e=s.get("env", {}), decisions=decisions, run_id=run_dir.name, slower=slower,
        rejected=[h for h in s.get("history", []) if h.get("outcome") == "rejected"])
    out = run_dir / "report.html"
    out.write_text(html, encoding="utf-8")
    return out
