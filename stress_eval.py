"""Stress test: does the pipeline survive a real phone line?

For every scenario it runs the call, then re-hears the rep's answers the way a
production agent would — spoken, pushed through a narrowband noisy telephone
channel, and transcribed by a local STT model — and re-extracts from *that*.
Two things get measured:

  1. Field accuracy THROUGH voice vs. on clean text — how much the speech layer
     actually costs (spoiler from the recon: numbers survive; words don't).
  2. Triage routing vs. the safe outcome — whether every unsafe call (a
     terminated plan, an impossible number) gets flagged before it reaches the
     EHR, instead of auto-posting wrong data.

Free + offline: macOS `say`, scipy, faster-whisper. No ElevenLabs, no API for
the audio path. (The agent + extractor still use the LLM.)

    python stress_eval.py            # all scenarios, through the phone channel
    python stress_eval.py --json out.json
"""
import argparse
import json

from agent import extract, extract_scored
from call import run_call
from hard_scenarios import STRESS_SCENARIOS
from payer_sim import PayerSim
from triage import triage
from verifier import verify
from voice_channel import hear_transcript


def field_match(exp, act) -> bool:
    if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
        return abs(float(exp) - float(act)) < 0.01
    return exp == act


def score(result, expected) -> tuple[int, int, list]:
    c = misses = 0
    bad = []
    for k, v in expected.items():
        if field_match(v, getattr(result, k)):
            c += 1
        else:
            misses += 1
            bad.append(f"{k}: expected {v}, heard {getattr(result, k)}")
    return c, misses, bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", help="write full results to this path")
    args = ap.parse_args()

    rows, results = [], []
    tot_fields = tot_clean = tot_heard = 0
    route_ok = flagged_ok = flagged_total = 0

    print("\nRunning the phone-line stress test — speak → degrade → STT → extract.")
    print("(First run downloads the ~150 MB STT model.)\n")

    for sc in STRESS_SCENARIOS:
        payer = PayerSim(sc["truth"], behavior=sc["behavior"])
        transcript = run_call(payer)

        # Baseline: extract from the clean text transcript.
        clean = extract(transcript)
        # Through voice: re-hear the rep, then extract from what STT produced.
        heard_tx = hear_transcript(transcript)
        heard, conf, _ = extract_scored(heard_tx)

        cc, _, _ = score(clean, sc["expected"])
        hc, hmiss, bad = score(heard, sc["expected"])
        n = len(sc["expected"])
        tot_fields += n; tot_clean += cc; tot_heard += hc

        decision = triage(heard, conf, flags=verify(heard))
        want = sc["expect_route"]
        ok = decision.route == want
        route_ok += ok
        # Safety = an unsafe call (one that must NOT auto-post) never auto-posts.
        unsafe = want in ("REVIEW", "REDO")
        if unsafe:
            flagged_total += 1
            flagged_ok += decision.route in ("REVIEW", "REDO")

        reason = decision.reasons[0] if decision.reasons else ""
        rows.append((sc["name"], cc, hc, n, decision.route, want, ok, bad, reason))
        results.append({"scenario": sc["name"], "clean": cc, "heard": hc, "n": n,
                        "route": decision.route, "expect_route": want,
                        "reason": reason, "misses_through_voice": bad})

    # Safety failure = a call that should have been stopped but auto-posted.
    slipped = sum(1 for r in results
                  if r["expect_route"] in ("REVIEW", "REDO") and r["route"] == "AUTO_POST")
    auto = sum(1 for r in results if r["route"] == "AUTO_POST")

    print("=" * 86)
    print(f"{'SCENARIO':42}{'clean':>7}{'voice':>7}{'  route':>12}")
    print("=" * 86)
    for name, cc, hc, n, route, want, ok, bad, reason in rows:
        print(f"{name[:42]:42}{f'{cc}/{n}':>7}{f'{hc}/{n}':>7}{route:>12}")
        print(f"    ↳ {reason}")
        for b in bad:
            print(f"    ⚠ voice-introduced error — {b}  (caught: {'yes' if route!='AUTO_POST' else 'NO'})")
    print("-" * 86)
    cp = f"{100*tot_clean//tot_fields}%" if tot_fields else "n/a"
    hp = f"{100*tot_heard//tot_fields}%" if tot_fields else "n/a"
    print(f"Field accuracy — clean text {tot_clean}/{tot_fields} ({cp})   ·   "
          f"through phone + STT {tot_heard}/{tot_fields} ({hp})")
    print(f"Unsafe calls that slipped through to auto-post: {slipped}   "
          f"(caught {flagged_ok}/{flagged_total})")
    print(f"Auto-posted with no human: {auto}/{len(rows)}   ·   "
          f"routed to a human: {len(rows)-auto}/{len(rows)}")
    print("=" * 86)

    if args.json:
        summary = {"clean_accuracy": [tot_clean, tot_fields],
                   "voice_accuracy": [tot_heard, tot_fields],
                   "route_correct": [route_ok, len(rows)],
                   "unsafe_flagged": [flagged_ok, flagged_total],
                   "scenarios": results}
        with open(args.json, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
