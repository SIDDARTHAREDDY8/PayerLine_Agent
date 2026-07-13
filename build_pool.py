"""Pre-render a pool of real calls so the live button is instant.

Generating a fresh 16-turn LLM conversation and voicing it takes minutes — far
too slow to do on every click. Instead we generate a handful offline (one per
scenario, covering auto-post / review / re-verify), voice each with Piper, and
commit them. At click time the app just serves the next one: instant, varied,
and every call is a genuine end-to-end run.

    python build_pool.py            # writes assets/calls/NN.json (audio inline)
"""
import base64
import json
import subprocess
from pathlib import Path

import voice
from agent import extract_scored
from call import run_call
from hard_scenarios import STRESS_SCENARIOS
from payer_sim import PayerSim
from triage import triage
from verifier import verify

OUT = Path("assets/calls")


def pipeline_for(sc: dict) -> dict:
    payer = PayerSim(sc["truth"], behavior=sc["behavior"])
    transcript = run_call(payer)
    result, conf, _ = extract_scored(transcript)
    flags = verify(result)
    decision = triage(result, conf, flags=flags)
    return {
        "scenario": sc["name"],
        "transcript": transcript,
        "result": result.model_dump(exclude_none=True),
        "flags": flags,
        "triage": {"route": decision.route, "reasons": decision.reasons},
    }


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    for i, sc in enumerate(STRESS_SCENARIOS):
        print(f"\n[{i+1}/{len(STRESS_SCENARIOS)}] {sc['name']}")
        bundle = pipeline_for(sc)

        wav = OUT / f"{i:02d}.wav"
        m4a = OUT / f"{i:02d}.m4a"
        voice.render(bundle["transcript"], str(wav), engine="piper")
        subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(m4a)],
                       check=True)
        bundle["audio_b64"] = base64.b64encode(m4a.read_bytes()).decode()
        wav.unlink(missing_ok=True)
        m4a.unlink(missing_ok=True)

        (OUT / f"{i:02d}.json").write_text(json.dumps(bundle))
        print(f"  → {OUT/f'{i:02d}.json'}  route={bundle['triage']['route']}  "
              f"audio={len(bundle['audio_b64'])//1000}k")

    print(f"\nwrote {len(STRESS_SCENARIOS)} calls to {OUT}/")


if __name__ == "__main__":
    main()
