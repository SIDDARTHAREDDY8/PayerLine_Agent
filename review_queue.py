"""The review queue — one pass that runs every scenario end to end and shows,
for each call: field accuracy, the triage route, and why.

Then the headline a founder cares about: the auto-post rate — the share of calls
that need zero human attention.

    python review_queue.py
"""
from agent import extract_scored
from call import run_call
from payer_sim import PayerSim
from scenarios import SCENARIOS
from triage import triage
from verifier import verify

ICON = {"AUTO_POST": "✓ auto-post", "REVIEW": "⚑ human review", "REDO": "↻ re-verify"}


def field_match(exp, act) -> bool:
    if isinstance(exp, (int, float)) and isinstance(act, (int, float)):
        return abs(float(exp) - float(act)) < 0.01
    return exp == act


def pct(correct: int, total: int) -> str:
    return f"{100*correct//total}%" if total else "n/a"


def main():
    total = correct = auto = 0
    rows = []

    for sc in SCENARIOS:
        payer = PayerSim(sc["truth"], behavior=sc["behavior"])
        transcript = run_call(payer, verbose=False)
        result, conf, _ = extract_scored(transcript)
        decision = triage(result, conf, flags=verify(result))

        c = sum(field_match(v, getattr(result, k)) for k, v in sc["expected"].items())
        n = len(sc["expected"])
        total += n
        correct += c
        if decision.route == "AUTO_POST":
            auto += 1
        rows.append((sc["name"], c, n, decision))

    print("\n" + "=" * 74)
    print("REVIEW QUEUE".ljust(46) + "acc".rjust(7) + "   route")
    print("=" * 74)
    for name, c, n, d in rows:
        print(f"{name[:44]:44} {c}/{n:<4} {pct(c, n):>4}  {ICON[d.route]}")
        print(f"    └ {d.reasons[0]}")

    rate = round(100 * auto / len(rows))
    print("-" * 74)
    print(f"Overall field accuracy: {correct}/{total} ({pct(correct, total)})")
    print(f"Auto-post rate: {auto}/{len(rows)} calls ({rate}%) need NO human.")
    print(f"→ A team reviews only the other {100-rate}% — the calls the system "
          f"isn't sure about.")
    print("=" * 74)


if __name__ == "__main__":
    main()
