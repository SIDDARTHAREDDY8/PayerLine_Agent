"""One call, end to end — the whole pipeline as a single function.

Both the web app (app.py) and the pre-recorded-asset builder (build_demo_asset.py)
call this, so the live button and the shipped demo are literally the same code
path. Returns everything the UI shows: transcript, structured record, per-field
confidence, verification flags, triage decision, and the rendered audio file.
"""
from call import run_call
from payer_sim import PayerSim
from scenarios import SCENARIOS

import voice


def run_pipeline(scenario_idx: int = 2) -> dict:
    """The reasoning + reliability half: call → extract → verify → triage.
    No audio, so it works even when the TTS quota is exhausted."""
    from agent import extract_scored          # imported lazily; needs an API key
    from triage import triage
    from verifier import verify

    sc = SCENARIOS[scenario_idx]
    payer = PayerSim(sc["truth"], behavior=sc["behavior"])
    transcript = run_call(payer)

    result, conf, evidence = extract_scored(transcript)
    flags = verify(result)
    decision = triage(result, conf, flags=flags)

    return {
        "scenario": sc["name"],
        "transcript": transcript,
        "result": result.model_dump(exclude_none=True),
        "confidence": conf,
        "flags": flags,
        "triage": {"route": decision.route, "reasons": decision.reasons},
    }


def generate(scenario_idx: int = 2, audio_out: str = "call.wav",
             engine: str = "auto") -> dict:
    """Full bundle incl. rendered audio — used to build the pre-recorded asset."""
    bundle = run_pipeline(scenario_idx)
    bundle["audio"] = str(voice.render(bundle["transcript"], audio_out, engine=engine))
    return bundle
