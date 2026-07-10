"""Thin Anthropic wrapper + JSON helpers.

One place to swap models. Agent and payer-sim use different models on purpose so
the agent isn't 'talking to itself' with identical behavior.
"""
import json
import os
import re

from anthropic import Anthropic, BadRequestError


def _load_dotenv(path=".env"):
    """Minimal .env loader (no dependency) so `python run_demo.py` just works."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, val = line.partition("=")
            os.environ.setdefault(key.strip(), val.strip().strip('"').strip("'"))


_load_dotenv()
_client = Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

MODEL_AGENT = os.environ.get("MODEL_AGENT", "claude-sonnet-5")
MODEL_SIM = os.environ.get("MODEL_SIM", "claude-haiku-4-5-20251001")


# `temperature` is deprecated on newer models (claude-sonnet-5 returns 400).
# Older ones still honor it, and the payer sim relies on it for turn variety —
# so send it, and remember the models that refuse.
_NO_TEMPERATURE: set[str] = set()


def chat(system: str, messages: list[dict], model: str, max_tokens: int = 800,
         temperature: float | None = None) -> str:
    send_temp = temperature is not None and model not in _NO_TEMPERATURE
    kwargs = {"temperature": temperature} if send_temp else {}
    try:
        resp = _client.messages.create(
            model=model, system=system, messages=messages,
            max_tokens=max_tokens, **kwargs)
    except BadRequestError as e:
        if not send_temp or "temperature" not in str(e):
            raise
        _NO_TEMPERATURE.add(model)
        resp = _client.messages.create(
            model=model, system=system, messages=messages, max_tokens=max_tokens)
    return "".join(b.text for b in resp.content if b.type == "text").strip()


def parse_json(text: str) -> dict:
    """Best-effort: strip code fences, grab the outermost JSON object."""
    text = re.sub(r"^```(?:json)?|```$", "", text.strip(), flags=re.MULTILINE).strip()
    start, end = text.find("{"), text.rfind("}")
    if start == -1 or end == -1:
        raise ValueError(f"No JSON object found in model output:\n{text}")
    return json.loads(text[start:end + 1])
