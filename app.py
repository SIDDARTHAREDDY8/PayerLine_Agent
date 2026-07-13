"""PayerLine — the click-to-run demo (Gradio, for Hugging Face Spaces).

Serves a pool of pre-rendered real calls (assets/calls/): each is a genuine
end-to-end run — agent ↔ simulated payer, extracted, verified, triaged, and
voiced with Piper — committed with its audio so a click plays instantly. A
fresh call takes minutes to generate and voice, so we do that offline
(build_pool.py), not on every click.

No secrets are needed to serve the demo; everything is pre-rendered.
"""
import base64
import html
import json
import os
import tempfile
from pathlib import Path

import gradio as gr

# Gradio 4.44's API-schema generator crashes on a boolean JSON schema
# (`if "const" in schema` where schema is a bool) — a known bug that aborts the
# Space's startup self-check. Guard that recursive helper to tolerate bools.
# https://github.com/gradio-app/gradio/issues/11722
import gradio_client.utils as _gcu

_orig_jstpt = getattr(_gcu, "_json_schema_to_python_type", None)
if _orig_jstpt is not None:
    def _safe_jstpt(schema, defs=None, _orig=_orig_jstpt):
        if isinstance(schema, bool):
            return "Any"
        return _orig(schema, defs)
    _gcu._json_schema_to_python_type = _safe_jstpt

# The Space is on ZeroGPU hardware, which refuses to start unless it finds a
# @spaces.GPU function — and HF blocks non-PRO accounts from downgrading to CPU.
# This app is CPU-only (LLM + TTS API calls), so define a no-op GPU function
# purely to pass ZeroGPU's startup check. It's never called, so it consumes no
# GPU time; the real work runs on CPU. On non-ZeroGPU hardware `spaces` is absent
# and this is skipped.
try:
    import spaces

    @spaces.GPU
    def _zerogpu_probe():
        return None
except Exception:
    pass

from voice import _clean_for_speech

ASSETS = Path("assets")
CANNED = json.loads((ASSETS / "demo_call.json").read_text())


def _decode_canned_audio() -> str:
    """The pre-recorded call is committed as base64 text (HF Spaces rejects raw
    binaries). Decode it to a temp file once so gr.Audio can serve it."""
    b64 = (ASSETS / "demo_call.m4a.b64").read_text()
    out = Path(tempfile.gettempdir()) / "payerline_demo_call.m4a"
    out.write_bytes(base64.b64decode(b64))
    return str(out)


CANNED_AUDIO = _decode_canned_audio()


def _load_pool():
    """Pre-rendered real calls (transcript + result + triage + audio), served
    instantly on click — generating one live takes minutes, so we don't."""
    pool, d = [], ASSETS / "calls"
    for f in sorted(d.glob("*.json")) if d.exists() else []:
        b = json.loads(f.read_text())
        out = Path(tempfile.gettempdir()) / f"payerline_pool_{f.stem}.m4a"
        out.write_bytes(base64.b64decode(b["audio_b64"]))
        pool.append((b, str(out)))
    return pool


POOL = _load_pool()
_pool_idx = {"i": -1}

FIELD_LABELS = {
    "coverage_active": "Coverage active", "plan_type": "Plan type",
    "copay_specialist": "Specialist copay", "copay_pcp": "PCP copay",
    "deductible_individual": "Deductible", "deductible_met": "Deductible met",
    "oop_max_individual": "Out-of-pocket max", "oop_met": "OOP met",
    "coinsurance_pct": "Coinsurance %", "prior_auth_required": "Prior auth",
    "reference_number": "Reference #", "effective_date": "Effective date",
    "payer": "Payer", "plan_name": "Plan",
}
ROUTE_STYLE = {
    "AUTO_POST": ("#0a7", "✓ AUTO-POST", "straight to the EHR — no human needed"),
    "REVIEW": ("#d80", "⚑ HUMAN REVIEW", "a human glances before it posts"),
    "REDO": ("#c33", "↻ RE-VERIFY", "payer data inconsistent — call again"),
}


def _transcript_html(transcript) -> str:
    rows = []
    for t in transcript:
        who = t["speaker"] if isinstance(t, dict) else t[0]
        text = t["text"] if isinstance(t, dict) else t[1]
        if not text.strip():
            continue
        agent = who.lower() == "agent"
        name = "Agent (RCM)" if agent else "Payer — Dana"
        align = "flex-end" if agent else "flex-start"
        bg = "#2563eb" if agent else "#e5e7eb"
        fg = "#fff" if agent else "#111"
        rows.append(
            f"<div style='display:flex;justify-content:{align};margin:6px 0'>"
            f"<div style='max-width:78%;background:{bg};color:{fg};padding:8px 12px;"
            f"border-radius:14px;font-size:14px;line-height:1.4'>"
            f"<div style='font-size:11px;opacity:.7;margin-bottom:2px'>{name}</div>"
            f"{html.escape(_clean_for_speech(text))}</div></div>")
    return "<div style='padding:4px'>" + "".join(rows) + "</div>"


def _result_md(result: dict) -> str:
    lines = ["| Field | Value |", "| --- | --- |"]
    for key, label in FIELD_LABELS.items():
        if key in result and result[key] is not None:
            v = result[key]
            v = ("Yes" if v is True else "No" if v is False else v)
            lines.append(f"| {label} | {v} |")
    return "\n".join(lines)


def _verify_md(flags) -> str:
    if not flags:
        return "✅ **All required fields captured and internally consistent.**"
    return "\n".join(f"- ⚠️ {f}" for f in flags)


def _triage_html(triage: dict) -> str:
    color, label, sub = ROUTE_STYLE.get(triage["route"], ("#666", triage["route"], ""))
    reason = triage["reasons"][0] if triage.get("reasons") else sub
    return (f"<div style='border-left:5px solid {color};padding:10px 14px;"
            f"background:rgba(0,0,0,.03);border-radius:6px'>"
            f"<span style='color:{color};font-weight:700;font-size:16px'>{label}</span>"
            f"<div style='font-size:13px;opacity:.8;margin-top:3px'>{sub}</div>"
            f"<div style='font-size:12px;opacity:.7;margin-top:4px'>{reason}</div></div>")


def _render_bundle(b: dict, audio_path: str):
    return (audio_path,
            _transcript_html(b["transcript"]), _result_md(b["result"]),
            _verify_md(b["flags"]), _triage_html(b["triage"]),
            f"**Scenario:** {b['scenario']}")


def load_canned():
    return _render_bundle(CANNED, CANNED_AUDIO)


def next_call():
    """Serve the next pre-rendered real call — instant, and a different scenario
    (and triage outcome) each click."""
    if not POOL:
        return load_canned()
    _pool_idx["i"] = (_pool_idx["i"] + 1) % len(POOL)
    b, audio = POOL[_pool_idx["i"]]
    out = list(_render_bundle(b, audio))
    out[-1] = f"**Scenario:** {b['scenario']}  ·  _real call {_pool_idx['i']+1} of {len(POOL)}_"
    return tuple(out)


INTRO = """
# 📞 PayerLine — a voice agent that verifies insurance benefits
An outbound agent calls a (simulated) insurance payer, works an eligibility
checklist, **catches the rep's mistakes**, and hands the EHR clean, structured
data — routing only the risky calls to a human. The call below is **real**: the
agent pushed back when the rep misstated the deductible, so it auto-posts.
"""


with gr.Blocks(title="PayerLine") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            audio = gr.Audio(label="Listen to the call", type="filepath",
                             interactive=False)
            scen = gr.Markdown()
            transcript = gr.HTML(label="Transcript")
        with gr.Column(scale=2):
            gr.Markdown("### Structured result (EHR-ready)")
            result = gr.Markdown()
            gr.Markdown("### Verification layer")
            verify_out = gr.Markdown()
            gr.Markdown("### Triage decision")
            triage_out = gr.HTML()
    gr.Markdown("---")
    run_btn = gr.Button("▶ Play another real call", variant="primary")
    gr.Markdown(
        "_Every call here is a genuine end-to-end run — agent ↔ simulated payer, "
        "extracted, verified, triaged, and voiced. Each click plays a different "
        "one (auto-post, human-review, and re-verify outcomes all show up)._")

    outs = [audio, transcript, result, verify_out, triage_out, scen]
    demo.load(load_canned, outputs=outs)
    run_btn.click(next_call, outputs=outs)


if __name__ == "__main__":
    # HF Spaces proxies 0.0.0.0:7860 — binding to localhost isn't reachable there.
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
