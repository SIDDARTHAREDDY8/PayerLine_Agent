"""Harder scenarios for the stress test — where the *rep* is the failure source.

The clean demo set (scenarios.py) shows the happy path. These add calls a real
payer line throws at you: a terminated plan whose rep still quotes copays, and a
rep who states an impossible number and won't back down. The agent can't always
fix these live — so the deterministic verification + triage layer has to catch
them before the data reaches billing.

Each scenario carries `expect_route`: the triage decision that keeps bad data
out of the EHR. The stress test scores routing against it.
"""
from scenarios import SCENARIOS

# Tag the clean set with the safe routing outcome.
_ROUTES = ["AUTO_POST", "AUTO_POST", "AUTO_POST", "REVIEW"]
CLEAN = [{**s, "expect_route": r} for s, r in zip(SCENARIOS, _ROUTES)]

HARD = [
    {
        "name": "Terminated plan — rep still quotes copays",
        "expect_route": "REDO",
        "behavior": "This member's coverage TERMINATED on 2026-06-30. When asked "
                    "if coverage is active, say no, it ended last month. But you're "
                    "a busy rep — if asked about copays, absent-mindedly quote them "
                    "anyway: specialist $45, PCP $25. Give reference 4471-BC.",
        "truth": {
            "payer": "BlueCross BlueShield", "plan_name": "BCBS PPO Gold",
            "plan_type": "PPO", "coverage_active": False, "effective_date": "2025-01-01",
            "copay_pcp": 25, "copay_specialist": 45,
            "deductible_individual": 1500, "deductible_met": 1500,
            "oop_max_individual": 6000, "oop_met": 3000,
            "coinsurance_pct": 20, "prior_auth_required": True,
            "reference_number": "4471-BC",
        },
        "expected": {
            "coverage_active": False, "plan_type": "PPO",
            "copay_specialist": 45,
        },
    },
    {
        "name": "Impossible deductible — rep won't correct it",
        "expect_route": "REDO",
        "behavior": "Be cooperative, BUT when asked how much of the deductible has "
                    "been met, say $2,500 has been met on a $1,500 deductible. If the "
                    "caller pushes back that it can't exceed the deductible, INSIST "
                    "you're reading it correctly and do NOT change the number. Give "
                    "reference 9920-AE.",
        "truth": {
            "payer": "Aetna", "plan_name": "Aetna PPO Select",
            "plan_type": "PPO", "coverage_active": True, "effective_date": "2026-01-01",
            "copay_pcp": 20, "copay_specialist": 40,
            "deductible_individual": 1500, "deductible_met": 2500,   # what the rep insists
            "oop_max_individual": 5000, "oop_met": 1200,
            "coinsurance_pct": 20, "prior_auth_required": False,
            "reference_number": "9920-AE",
        },
        "expected": {
            "plan_type": "PPO", "deductible_individual": 1500,
            "copay_specialist": 40,
        },
    },
]

STRESS_SCENARIOS = CLEAN + HARD
