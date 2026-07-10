"""Run ONE call end to end and print the transcript, structured output, FHIR
payload, and verification flags.

    python run_demo.py           # first scenario
    python run_demo.py 2         # third scenario (adversarial)
"""
import json
import sys

from agent import extract_scored
from call import run_call
from payer_sim import PayerSim
from scenarios import SCENARIOS
from triage import triage
from verifier import REQUIRED, verify


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("-")]
    flags = {a for a in sys.argv[1:] if a.startswith("-")}
    idx = int(args[0]) if args else 0
    if not 0 <= idx < len(SCENARIOS):
        print(f"No scenario {idx}. Pick 0-{len(SCENARIOS)-1}:")
        for i, s in enumerate(SCENARIOS):
            print(f"  {i}  {s['name']}")
        sys.exit(1)
    sc = SCENARIOS[idx]

    # --speak: hear the call as it happens.  --render: save it as call.mp3.
    on_turn = None
    if "--speak" in flags:
        from voice import speak
        on_turn = speak

    print(f"\n{'='*70}\nSCENARIO: {sc['name']}\n{'='*70}")
    payer = PayerSim(sc["truth"], behavior=sc["behavior"])

    print("\n--- LIVE CALL ---")
    transcript = run_call(payer, verbose=True, on_turn=on_turn)

    if "--render" in flags:
        from voice import render
        print("\n--- RENDERING AUDIO ---")
        render(transcript)

    print("\n\n--- STRUCTURED RESULT (EHR-ready) ---")
    result, conf, evidence = extract_scored(transcript)
    print(result.model_dump_json(indent=2, exclude_none=True))

    print("\n--- FIELD CONFIDENCE (required fields) ---")
    for name, label in REQUIRED:
        c = conf.get(name, 0.0)
        bar = "█" * round(c * 10) + "·" * (10 - round(c * 10))
        val = getattr(result, name)
        print(f"  {label:26} {bar} {c:>4.0%}  {val if val is not None else '—'}")

    print("\n--- VERIFICATION LAYER ---")
    flags = verify(result)
    if flags:
        for f in flags:
            print(f"  ⚠  {f}")
    else:
        print("  ✓ All required fields captured and internally consistent.")

    print("\n--- TRIAGE DECISION ---")
    d = triage(result, conf, flags=flags)
    print(f"  ROUTE: {d.route}")
    for r in d.reasons:
        print(f"    • {r}")

    print("\n--- FHIR CoverageEligibilityResponse (write-back payload) ---")
    print(json.dumps(result.to_fhir(), indent=2))


if __name__ == "__main__":
    main()
