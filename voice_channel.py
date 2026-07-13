"""A simulated payer phone line: text → speech → telephone degradation → STT.

The demo's honest gap was that it ran on clean text. This closes it: the rep's
answers are spoken (macOS `say`), pushed through a narrowband, noisy telephone
channel, and transcribed back with a local speech-to-text model — the exact path
a production agent hears. Everything here is free and offline: `say` for TTS,
scipy for the channel, faster-whisper for STT.

Only the PAYER's turns go through this — in a real call the agent's own
utterances are known text; only the rep's voice is heard over the line.
"""
import subprocess
import tempfile
import wave
from pathlib import Path

import numpy as np
from scipy.signal import butter, lfilter

# Telephone reality: ~300–3400 Hz passband, and line noise. 12 dB SNR is a
# mediocre-but-plausible payer line — bad enough to garble words, and (as the
# measurement shows) still good enough to keep the digits.
_BAND = (300, 3400)
_SNR_DB = 12
_SAY_VOICE = "Samantha"

_model = None


def _stt_model():
    global _model
    if _model is None:
        from faster_whisper import WhisperModel
        _model = WhisperModel("base.en", device="cpu", compute_type="int8")
    return _model


def _say(text: str, path: Path, rate: int = 16000) -> None:
    subprocess.run(["say", "-v", _SAY_VOICE, "-o", str(path),
                    f"--data-format=LEI16@{rate}", text], check=True)


def _load(path: Path):
    with wave.open(str(path)) as w:
        a = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
        return a, w.getframerate()


def _telephone(a: np.ndarray, sr: int, snr_db: float = _SNR_DB, seed: int = 0) -> np.ndarray:
    b, al = butter(4, [_BAND[0] / (sr / 2), _BAND[1] / (sr / 2)], btype="band")
    a = lfilter(b, al, a)
    power = np.mean(a ** 2) + 1e-12
    noise = np.random.RandomState(seed).normal(0, np.sqrt(power / 10 ** (snr_db / 10)), len(a))
    a = a + noise
    return a / (np.max(np.abs(a)) + 1e-9) * 0.9


def hear(text: str, seed: int = 0) -> str:
    """Speak `text`, run it through the telephone channel, and transcribe it back.
    Returns what the agent's STT would actually receive."""
    if not text.strip():
        return text
    with tempfile.TemporaryDirectory() as tmp:
        clean, degraded = Path(tmp) / "c.wav", Path(tmp) / "d.wav"
        _say(text, clean)
        a, sr = _load(clean)
        deg = (_telephone(a, sr, seed=seed) * 32767).astype(np.int16)
        with wave.open(str(degraded), "wb") as w:
            w.setnchannels(1); w.setsampwidth(2); w.setframerate(sr)
            w.writeframes(deg.tobytes())
        segs, _ = _stt_model().transcribe(str(degraded), language="en")
        return " ".join(s.text for s in segs).strip()


def hear_transcript(transcript: list[dict]) -> list[dict]:
    """Return a copy of the call as the agent HEARD it: payer turns through the
    phone channel + STT, agent turns left as the known text they were."""
    heard = []
    for i, t in enumerate(transcript):
        if t["speaker"] == "payer" and t["text"].strip():
            heard.append({"speaker": "payer", "text": hear(t["text"], seed=i)})
        else:
            heard.append(dict(t))
    return heard
