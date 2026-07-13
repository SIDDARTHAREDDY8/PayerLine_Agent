"""The VoiceAdmin-style agent: calls a payer, authenticates, and works a
checklist to verify eligibility, then hands back structured data.

Kept deliberately close to how a real RCM rep runs the call so the transcript
reads like the real thing.
"""
from llm import MODEL_AGENT, chat, parse_json
from schema import EligibilityResult

CALLER_CONTEXT = {
    "provider": "Riverside Family Medicine",
    "npi": "1548291057",
    "member_id": "ZK884120931",
    "patient_dob": "1984-03-12",
    "service": "specialist office visit",
}

AGENT_SYSTEM = f"""You are an outbound revenue-cycle voice agent calling an
insurance payer to verify a patient's eligibility and benefits. Speak in short,
natural phone sentences — one turn at a time. You are efficient and polite.

Your caller identity (provide when the IVR/rep asks):
- Provider: {CALLER_CONTEXT['provider']}, NPI {CALLER_CONTEXT['npi']}
- Member ID: {CALLER_CONTEXT['member_id']}, patient DOB {CALLER_CONTEXT['patient_dob']}
- Reason for call: verify benefits for a {CALLER_CONTEXT['service']}

Work this CHECKLIST, asking for whatever is still unknown:
  1. Capture the call reference number.
  2. Is coverage active today? Plan type (HMO/PPO/etc.)? Effective date?
  3. Specialist copay and PCP copay.
  4. Individual deductible AND how much has been met.
  5. Individual out-of-pocket max AND how much has been met.
  6. Coinsurance %.
  7. Is prior authorization required for the specialist visit?

Rules:
- Ask about only 1-2 items per turn, like a real call.
- If an answer is internally inconsistent (e.g. deductible met exceeds the
  deductible), politely push back and re-confirm — do NOT accept it silently.
- When every checklist item is captured, say a brief goodbye and end your final
  message with the token [[END_CALL]].
"""

EXTRACT_SYSTEM = """You extract a structured eligibility record from a call
transcript. Output ONLY a JSON object with these keys (use null when the payer
never stated it — do not guess):
member_id, payer, plan_name, plan_type, coverage_active (bool), group_number,
effective_date, copay_pcp (number), copay_specialist (number),
deductible_individual (number), deductible_met (number),
oop_max_individual (number), oop_met (number), coinsurance_pct (number),
prior_auth_required (bool), reference_number, notes.
Numbers must be plain numbers (no $ or %)."""


class Agent:
    def utterance(self, transcript: list[dict]) -> str:
        # From the agent's POV, the payer is the 'user'. Skip empty turns — the
        # API rejects empty message content.
        msgs = [{"role": "user" if t["speaker"] == "payer" else "assistant",
                 "content": t["text"]} for t in transcript if t["text"].strip()]
        if not msgs or msgs[0]["role"] != "user":  # agent opens the call
            msgs.insert(0, {"role": "user", "content": "[Call connected — payer line ringing]"})
        return chat(AGENT_SYSTEM, msgs, model=MODEL_AGENT, max_tokens=200)


def extract(transcript: list[dict]) -> EligibilityResult:
    convo = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in transcript)
    raw = chat(EXTRACT_SYSTEM, [{"role": "user", "content": convo}],
               model=MODEL_AGENT, max_tokens=700, temperature=0)
    return EligibilityResult(**parse_json(raw))


# ---- confidence-scored extraction (feeds the triage / human-review queue) ----

SCORED_FIELDS = [
    "member_id", "payer", "plan_name", "plan_type", "coverage_active",
    "group_number", "effective_date", "copay_pcp", "copay_specialist",
    "deductible_individual", "deductible_met", "oop_max_individual", "oop_met",
    "coinsurance_pct", "prior_auth_required", "reference_number",
]

SCORED_SYSTEM = """You extract a structured eligibility record from a payer call
transcript AND rate your confidence in each field.

Return ONLY JSON of this exact shape:
{
  "fields": {
    "<field>": {"value": <value or null>, "confidence": <0.0-1.0>,
                "evidence": "<short verbatim quote from the transcript, or ''>"},
    ...
  },
  "notes": "<one sentence on anything notable (e.g. a correction), or ''>"
}

Include exactly these fields: member_id, payer, plan_name, plan_type,
coverage_active(bool), group_number, effective_date, copay_pcp(number),
copay_specialist(number), deductible_individual(number), deductible_met(number),
oop_max_individual(number), oop_met(number), coinsurance_pct(number),
prior_auth_required(bool), reference_number.

Rules:
- Use null and confidence 0 when the payer never stated the value. Never guess.
- confidence 1.0 = stated explicitly and unambiguously; 0.6-0.8 = hedged,
  partial, or inferred; below that = shaky.
- evidence must be a real quote from the transcript or '' — never fabricate one.
- Numbers are plain (no $ or %)."""


def extract_scored(transcript: list[dict]):
    """Returns (EligibilityResult, confidence_by_field, evidence_by_field).

    Falls back to the flat extractor if the scored JSON won't parse — messy real
    (or STT-transcribed) transcripts occasionally make the model emit invalid
    JSON, and a verification pipeline must never crash on one bad call."""
    convo = "\n".join(f"{t['speaker'].upper()}: {t['text']}" for t in transcript)
    try:
        data = parse_json(chat(SCORED_SYSTEM, [{"role": "user", "content": convo}],
                               model=MODEL_AGENT, max_tokens=1500, temperature=0))
        fields = data.get("fields", {})
        values, conf, evidence = {}, {}, {}
        for f in SCORED_FIELDS:
            cell = fields.get(f) or {}
            values[f] = cell.get("value")
            conf[f] = float(cell.get("confidence") or 0.0)
            evidence[f] = cell.get("evidence") or ""
        result = EligibilityResult(**values, notes=(data.get("notes") or None))
        return result, conf, evidence
    except (ValueError, TypeError):
        result = extract(transcript)                    # flat, far simpler JSON
        d = result.model_dump()
        conf = {f: (1.0 if d.get(f) is not None else 0.0) for f in SCORED_FIELDS}
        return result, conf, {f: "" for f in SCORED_FIELDS}
