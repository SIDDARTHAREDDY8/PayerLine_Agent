"""Accuracy eval harness — the part that shows engineering maturity.

Runs every scenario, extracts the structured record, and scores each field
against ground truth. Reports per-scenario and overall field-level accuracy.
This is how you'd catch a regression before it ships to a payer.

    python eval_run.py
"""
from agent import extract
from call import run_call
from payer_sim import PayerSim
from scenarios import SCENARIOS
from verifier import verify


def field_match(expected, actual) -> bool:
    if isinstance(expected, (int, float)) and isinstance(actual, (int, float)):
        return abs(float(expected) - float(actual)) < 0.01
    return expected == actual


def pct(correct: int, total: int) -> str:
    return f"{100*correct//total}%" if total else "n/a"


def main():
    total_fields = total_correct = 0

    for sc in SCENARIOS:
        payer = PayerSim(sc["truth"], behavior=sc["behavior"])
        transcript = run_call(payer, verbose=False)
        record = extract(transcript)
        result = record.model_dump()

        correct = wrong = 0
        misses = []
        for field, exp in sc["expected"].items():
            if field_match(exp, result.get(field)):
                correct += 1
            else:
                wrong += 1
                misses.append(f"{field}: expected {exp}, got {result.get(field)}")

        n = correct + wrong
        total_fields += n
        total_correct += correct
        flags = verify(record)

        print(f"\n{sc['name']}")
        print(f"  field accuracy: {correct}/{n}  ({pct(correct, n)})")
        for m in misses:
            print(f"    ✗ {m}")
        print(f"  verification flags raised: {len(flags)}")
        for f in flags:
            print(f"    ⚠ {f}")

    print(f"\n{'='*60}")
    print(f"OVERALL FIELD ACCURACY: {total_correct}/{total_fields}  "
          f"({pct(total_correct, total_fields)})")


if __name__ == "__main__":
    main()
