"""PayerLine — the live, click-to-run demo (Gradio, for Hugging Face Spaces).

Two modes, one code path (demo_pipeline.generate):
  • A real pre-recorded call plays instantly and free — no API calls (assets/).
  • "Run a live call" generates a fresh one on demand, behind a daily cap so a
    burst of visitors can't drain the API keys.

Secrets on the Space (Settings → Variables and secrets):
  ANTHROPIC_API_KEY, ELEVENLABS_API_KEY   — required for live runs
  MAX_LIVE_RUNS_PER_DAY                    — optional, defaults to 25
"""
import base64
import html
import json
import os
import tempfile
from datetime import date
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

import voice
from scenarios import SCENARIOS
from voice import _clean_for_speech

ASSETS = Path("assets")
CANNED = json.loads((ASSETS / "demo_call.json").read_text())
MAX_LIVE = int(os.environ.get("MAX_LIVE_RUNS_PER_DAY", "25"))
_runs: dict[str, int] = {}


def _decode_canned_audio() -> str:
    """The pre-recorded call is committed as base64 text (HF Spaces rejects raw
    binaries). Decode it to a temp file once so gr.Audio can serve it."""
    b64 = (ASSETS / "demo_call.m4a.b64").read_text()
    out = Path(tempfile.gettempdir()) / "payerline_demo_call.m4a"
    out.write_bytes(base64.b64decode(b64))
    return str(out)


CANNED_AUDIO = _decode_canned_audio()

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


def run_live(scenario_idx):
    keys = os.environ.get("ANTHROPIC_API_KEY") and (
        os.environ.get("ELEVENLABS_API_KEY") or os.environ.get("ELEVEN_API_KEY"))
    if not keys:
        raise gr.Error("Live runs need ANTHROPIC_API_KEY and ELEVENLABS_API_KEY "
                       "set as Space secrets. The pre-recorded call above is real.")
    today = str(date.today())
    if _runs.get(today, 0) >= MAX_LIVE:
        raise gr.Error(f"Daily live-run cap reached ({MAX_LIVE}). The pre-recorded "
                       "call above is a real run — try the live button tomorrow.")
    _runs[today] = _runs.get(today, 0) + 1

    # 1) The reasoning + reliability pipeline (the real substance).
    from demo_pipeline import run_pipeline
    try:
        b = run_pipeline(int(scenario_idx))
    except Exception as e:
        _runs[today] -= 1                       # a failed run shouldn't burn the cap
        raise gr.Error(f"Live call failed — {type(e).__name__}: {e}")

    # 2) Voice, but degrade gracefully — if the ElevenLabs quota is out, still show
    #    the call as text rather than erroring. Serve from Gradio's temp dir.
    audio_out = str(Path(tempfile.gettempdir()) / "payerline_live.wav")
    audio_path, note = None, ""
    try:
        audio_path = str(voice.render(b["transcript"], audio_out, engine="eleven"))
    except Exception as e:
        if "quota" in str(e).lower():
            note = ("  ·  🔇 ElevenLabs voice quota is used up this month — showing "
                    "the fresh call as text. The pre-recorded call above has real audio.")
        else:
            raise gr.Error(f"Voice render failed — {type(e).__name__}: {e}")

    left = MAX_LIVE - _runs[today]
    out = list(_render_bundle(b, audio_path))
    out[-1] = f"**Scenario:** {b['scenario']}  ·  _{left} live runs left today_{note}"
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
    gr.Markdown("---\n### Run a fresh call live")
    with gr.Row():
        scenario_dd = gr.Dropdown(
            [(s["name"], i) for i, s in enumerate(SCENARIOS)], value=2,
            label="Scenario", scale=3)
        run_btn = gr.Button("▶ Run a live call", variant="primary", scale=1)
    gr.Markdown(
        "_Each live run makes a fresh call (new every time) and spends API "
        "credits, so it's capped per day. The pre-recorded call above is free._")

    outs = [audio, transcript, result, verify_out, triage_out, scen]
    demo.load(load_canned, outputs=outs)
    run_btn.click(run_live, inputs=[scenario_dd], outputs=outs)


if __name__ == "__main__":
    # HF Spaces proxies 0.0.0.0:7860 — binding to localhost isn't reachable there.
    demo.queue(default_concurrency_limit=2).launch(
        server_name="0.0.0.0", server_port=int(os.environ.get("PORT", 7860)))
