---
title: PayerLine
emoji: 📞
colorFrom: indigo
colorTo: blue
sdk: gradio
sdk_version: 5.50.0
app_file: app.py
pinned: false
license: mit
---

# 📞 PayerLine — a voice agent that verifies insurance benefits

[![Live demo](https://img.shields.io/badge/%F0%9F%A4%97%20Live%20demo-PayerLine-blue)](https://siddarthareddy8-payerline.hf.space)

**An outbound AI agent calls a (simulated) insurance payer, works an eligibility
checklist, catches the rep's mistakes, and hands the EHR clean structured data —
routing only the risky calls to a human.**

### ▶ [Try it live — no signup](https://siddarthareddy8-payerline.hf.space)

A real payer call plays instantly, in real voices: hear the agent push back when
the rep claims $2,000 of a $1,000 deductible has been met, then auto-post the
corrected record. One button runs a fresh call live.

I built this after studying VoiceAdmin because I wanted to *show* rather than tell —
a small, runnable prototype of the hardest, most valuable part of an outbound
payer-call product: **run the call, extract accurate structured benefits, catch the
rep's mistakes, and hand the EHR clean data.** It runs against a *simulated* payer
(no real PHI, no live payer lines), which also means every call has a known ground
truth — so accuracy is **measured, not claimed.**

## What it does

```
┌─────────┐   phone-style turns   ┌──────────────┐
│  Agent  │◄─────────────────────►│  Payer sim   │  (IVR gate → live rep "Dana",
│ (RCM)   │                       │  + ground    │   knows the true benefits,
└────┬────┘                       │  truth)      │   sometimes wrong on purpose)
     │ transcript                 └──────────────┘
     ▼
  extract ──► EligibilityResult (Pydantic)
     │
     ├─► verify()  ── deterministic reliability checks (the "pushback" layer)
     └─► to_fhir() ── EHR-ready CoverageEligibilityResponse write-back
```

- **`agent.py`** — the outbound agent. Authenticates through the IVR, works an
  eligibility checklist one turn at a time, and is instructed to *push back* when
  a rep's numbers are internally inconsistent.
- **`payer_sim.py`** — a simulated payer (IVR + live rep) with a hidden true record.
  Some scenarios make the rep terse or wrong to stress-test the agent.
- **`verifier.py`** — deterministic checks (missing fields, deductible-met >
  deductible, inactive coverage w/ copays, etc.). This is the part that keeps a
  wrong copay from silently reaching billing — the reliability moat competitors
  (e.g. Infinitus's real-time correction) sell hardest.
- **`schema.py`** — structured output + a FHIR mapping, because the deliverable
  isn't "the call happened," it's clean data the EHR ingests without re-keying.
- **`triage.py`** — routes each finished call: **AUTO_POST** (confident + consistent
  → straight to the EHR), **REVIEW** (a field is missing or low-confidence → a human
  glances), or **REDO** (payer data inconsistent → re-verify). This is the lever that
  lets a small team stand behind millions of calls — humans only touch what the system
  is unsure about.
- **`review_queue.py`** — runs every scenario and reports the **auto-post rate** (75%
  in the current set) alongside field accuracy. The founder-facing scale story.
- **`eval_run.py`** — accuracy harness: scores every extracted field vs. ground
  truth across scenarios. How you'd catch a regression before it hits a payer.
- **`voice.py`** — the voice layer. Generates a **fresh call each run** and
  renders it to audio with real, human-sounding voices (ElevenLabs, one voice per
  speaker) — or the offline macOS `say` voice when no key is set. Audio is
  stitched from PCM via stdlib `wave`, so no ffmpeg is required.

## Run it

No API key? The verification layer and the FHIR write-back are pure logic, and
run against a scripted adversarial call:

```bash
pip install -r requirements.txt
python demo_offline.py       # no key needed — shows the naive vs. corrected record
python voice.py --scripted   # no key needed — hear the scripted call (offline `say`)
```

For **real, human-sounding voices** and a **fresh call each run**, add
`ELEVENLABS_API_KEY` (and `ANTHROPIC_API_KEY`) to `.env` — see `.env.example` —
then:

```bash
python voice.py              # generates a new call, speaks it with real voices
```

The full loop (live agent ↔ simulated payer ↔ LLM extraction) needs a key:

```bash
export ANTHROPIC_API_KEY=sk-...        # any LLM works; swap the client in llm.py

python run_demo.py 2            # runs the ADVERSARIAL scenario (rep misstates deductible)
python run_demo.py 2 --speak    # ...and hear it, live, turn by turn
python run_demo.py 2 --render   # ...and save it to call.wav
python eval_run.py              # full accuracy + verification report across scenarios
python review_queue.py          # per-call triage routes + the auto-post rate
```

Watch scenario 2: the rep claims $2,000 of a $1,000 deductible is met. A naive
agent records it; here the agent pushes back and the verifier flags the
inconsistency.

## Why this maps to VoiceAdmin specifically

| VoiceAdmin does | This prototype demonstrates |
|---|---|
| Outbound payer calls (claim status / eligibility) | The eligibility call, end to end |
| Structured data back to Epic/Cerner | `EligibilityResult` → FHIR write-back |
| HIPAA / accuracy at scale | A verification layer + a measurable eval harness |

## Honest limitations (what I'd build next, inside the company)

- **Voice is TTS-only, and there's no telephony.** `voice.py` speaks both sides
  (macOS `say`, offline) so a call is audible, but the loop is still text in the
  middle: no STT, no phone line. Wiring `call.py` to Twilio/LiveKit/Vapi is the
  seam — though note the payer here is a *simulator*, so dialing out would prove
  little. The real next step is STT: transcribing the audio back and re-running
  extraction would measure what the speech layer costs in accuracy, which is
  where a production system actually bleeds. I scoped this to the reasoning +
  reliability layer on purpose; that's the hard part.
- IVR navigation is simulated, not DTMF against real payer trees.
- The eval set is 4 scenarios to keep it readable; the harness scales to hundreds.
- Verification rules are hand-written; next step is learning them from labeled
  call outcomes (the knowledge-graph direction).

*Built as a conversation starter, not a product. Happy to walk through the design
choices — or pair on the real telephony integration.*
