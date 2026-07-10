"""Orchestrates one agent<->payer call and returns the transcript."""
from agent import Agent
from payer_sim import PayerSim

MAX_TURNS = 16
END_TOKEN = "[[END_CALL]]"


def run_call(payer: PayerSim, verbose: bool = False, on_turn=None) -> list[dict]:
    """`on_turn(speaker, text)` fires as each turn lands — the hook the voice
    layer uses to speak a call as it happens."""
    agent = Agent()
    transcript: list[dict] = []

    def emit(speaker, text):
        transcript.append({"speaker": speaker, "text": text})
        if verbose:
            print(f"{'' if speaker == 'payer' else chr(10)}  {speaker.upper()}: {text}")
        if on_turn and text:
            on_turn(speaker, text)

    for _ in range(MAX_TURNS):
        text = agent.utterance(transcript)
        ended = END_TOKEN in text
        emit("agent", text.replace(END_TOKEN, "").strip())
        if ended:
            break

        emit("payer", payer.reply(transcript))

    return transcript
