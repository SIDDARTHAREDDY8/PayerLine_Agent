"""The reliability layer — the part competitors make their moat.

A raw LLM will happily record whatever the rep said. Production RCM can't: a
wrong copay or a missed reference number costs money. This runs deterministic
checks over the extracted record and returns human-readable flags. In a real
system these flags would trigger a re-ask on the live call or route to a human.
"""
from schema import EligibilityResult

REQUIRED = [
    ("coverage_active", "active coverage status"),
    ("plan_type", "plan type"),
    ("copay_specialist", "specialist copay"),
    ("deductible_individual", "individual deductible"),
    ("deductible_met", "deductible met"),
    ("oop_max_individual", "out-of-pocket max"),
    ("reference_number", "payer call reference number"),
]


def verify(r: EligibilityResult) -> list[str]:
    flags: list[str] = []

    for field, label in REQUIRED:
        if getattr(r, field) is None:
            flags.append(f"MISSING · {label} was never confirmed by the rep")

    if r.deductible_individual is not None and r.deductible_met is not None:
        if r.deductible_met > r.deductible_individual:
            flags.append(
                f"INCONSISTENT · deductible met (${r.deductible_met}) exceeds "
                f"deductible (${r.deductible_individual}) — re-verify")

    if r.oop_max_individual is not None and r.oop_met is not None:
        if r.oop_met > r.oop_max_individual:
            flags.append(
                f"INCONSISTENT · OOP met (${r.oop_met}) exceeds OOP max "
                f"(${r.oop_max_individual}) — re-verify")

    # `is not None`, not truthiness — a $0 copay is a real, common benefit.
    if r.coverage_active is False and (r.copay_specialist is not None
                                       or r.copay_pcp is not None):
        flags.append("INCONSISTENT · coverage reported inactive but copays "
                     "returned — re-confirm active status before billing")

    if r.coinsurance_pct is not None and not (0 <= r.coinsurance_pct <= 100):
        flags.append(f"INVALID · coinsurance {r.coinsurance_pct}% out of range")

    return flags
