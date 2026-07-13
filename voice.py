"""The voice layer — makes a call audible instead of just readable.

Voice is the product; text turns were only ever a stand-in. This renders a
call to speech with a distinct voice per speaker and stitches the turns into
one audio file you can play, embed, or hand to someone who won't read the code.

Two engines:
  • ElevenLabs (default when ELEVENLABS_API_KEY is set) — real, human-sounding
    voices. Audio is stitched from PCM via stdlib `wave`, so no ffmpeg needed.
  • macOS `say` (automatic fallback) — robotic but free, offline, zero setup.

By default it generates a FRESH call each run (live agent ↔ simulated payer),
so no two runs sound alike. `--scripted` replays the fixed adversarial call
(no ANTHROPIC key needed).

Deliberately NOT telephony. The payer is a simulator, so dialing a real number
would prove nothing the text loop doesn't — see the README's limitations. This
makes the loop hearable; the telephony seam stays `call.py`.

    python voice.py                 # fresh call, real voices -> call.wav, then plays
    python voice.py --scripted      # replay the scripted adversarial call
    python voice.py --engine say    # force the offline robotic voice
    python voice.py --scenario 0    # pick which scenario to generate
    python run_demo.py 2 --speak    # speak a live call, turn by turn
"""
import argparse
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import urllib.error
import urllib.request
import wave
from pathlib import Path

# ── engine config ───────────────────────────────────────────────────────────
_RATE = 22050                       # ElevenLabs pcm_22050 and `say` both emit this
_GAP_SECONDS = 0.35                 # a beat between turns so it isn't a run-on

# `say` voices (offline fallback). Two clearly different so you can tell speakers apart.
SAY_VOICES = {"agent": "Samantha", "payer": "Tessa"}

# ElevenLabs voice IDs. These two "premade" voices work on the free API tier —
# most "library" voices (Aria, Rachel, …) return 402 unless you upgrade.
# Override per speaker with ELEVEN_VOICE_AGENT / ELEVEN_VOICE_PAYER.
ELEVEN_VOICES = {
    "agent": "ErXwobaYiN019PkySvjV",   # Antoni — the RCM caller
    "payer": "EXAVITQu4vr4xnSDxMaL",   # Sarah  — Dana, the payer rep
}
ELEVEN_MODEL = "eleven_multilingual_v2"
ELEVEN_URL = "https://api.elevenlabs.io/v1/text-to-speech/{vid}?output_format=pcm_22050"

# Piper — self-hosted neural TTS: free, offline, no key, no quota. This is what
# powers the live button on the deployed Space so it works on every click.
# Models are the shared piper-voices (medium = 22.05 kHz, matching _RATE).
PIPER_VOICES = {
    "agent": "en_US-ryan-medium",     # the RCM caller (male)
    "payer": "en_US-amy-medium",      # Dana, the payer rep (female)
}
_piper_cache: dict = {}


def _env(key: str, default: str = "") -> str:
    """os.environ first, then a bare read of .env (so we don't import llm/anthropic)."""
    if key in os.environ:
        return os.environ[key]
    try:
        for line in Path(".env").read_text().splitlines():
            line = line.strip()
            if line and not line.startswith("#") and line.split("=", 1)[0].strip() == key:
                return line.split("=", 1)[1].strip().strip('"').strip("'")
    except FileNotFoundError:
        pass
    return default


def _api_key() -> str:
    return _env("ELEVENLABS_API_KEY") or _env("ELEVEN_API_KEY")


def _piper_available() -> bool:
    try:
        import piper  # noqa: F401
        return True
    except Exception:
        return False


def resolve_engine(requested: str = "auto") -> str:
    """Pick a TTS engine. 'piper' is free/offline (the Space's live engine);
    'eleven' needs a key; 'say' is the macOS offline fallback."""
    if requested == "say":
        return "say"
    if requested == "piper":
        if _piper_available():
            return "piper"
        return "eleven" if _api_key() else "say"
    if requested in ("eleven", "auto") and _api_key():
        return "eleven"
    if requested == "eleven":
        sys.exit("No ELEVENLABS_API_KEY found. Add it to .env (see .env.example), "
                 "or use --engine say / --engine piper.")
    return "say"


def _require(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        sys.exit(f"missing {', '.join(missing)} — this is a macOS demo (needs `say`).")


def _clean_for_speech(text: str) -> str:
    """Strip markdown so TTS reads words, not symbols — the sim sometimes emits
    **bold**, `code`, and '- ' bullet lists that otherwise get voiced literally."""
    text = re.sub(r"[*_`#]+", "", text)                 # bold/italic/code/heading marks
    text = re.sub(r"^\s*[-•]\s+", "", text, flags=re.M)  # leading list bullets
    text = re.sub(r"[ \t]+", " ", text)                 # collapse runs of spaces
    text = re.sub(r"\n{2,}", ". ", text).replace("\n", ", ")  # newlines → spoken pauses
    return text.strip()


def _spell(token: str) -> str:
    """Space out a code so TTS reads it character by character, not as a number
    (NPI 1548291057 → 'one five four eight…', not 'one billion…')."""
    return " ".join(c for c in token if c not in "-/")


def _speechify(text: str) -> str:
    """What actually gets sent to TTS: markdown stripped, and identifiers spelled
    out digit-by-digit while dollar amounts, percentages, and years stay numbers.

    An identifier is a long pure-digit run (an NPI), an alphanumeric code (a
    member ID like ZK884120931), or a digit run joined by - or / (a reference
    number like 7712-UH). Short bare numbers ($60, 1000, 20%) are left alone."""
    text = _clean_for_speech(text)

    def fix(m):
        tok = m.group()
        digits = sum(c.isdigit() for c in tok)
        alpha = any(c.isalpha() for c in tok)
        sep = "-" in tok or "/" in tok
        if (digits >= 7 and not alpha) or (alpha and digits >= 3) or (sep and digits >= 2):
            return _spell(tok)
        return tok

    return re.sub(r"[A-Za-z0-9][A-Za-z0-9\-/]*[A-Za-z0-9]", fix, text)


def _normalize(transcript) -> list[tuple[str, str]]:
    """Accept run_call's [{'speaker','text'}] or the scripted [(SPEAKER, text)]."""
    turns = []
    for t in transcript:
        speaker, text = (t["speaker"], t["text"]) if isinstance(t, dict) else t
        text = text.strip()
        if text:
            turns.append((speaker.lower(), text))
    return turns


# ── per-turn synthesis: every engine writes a 16-bit mono PCM wav to `path` ──

def _say_clip(text: str, speaker: str, path: Path) -> None:
    subprocess.run(["say", "-v", SAY_VOICES.get(speaker, "Samantha"), "-o", str(path),
                    f"--data-format=LEI16@{_RATE}", _speechify(text)], check=True)


def _eleven_pcm(text: str, speaker: str, api_key: str) -> bytes:
    vid = _env(f"ELEVEN_VOICE_{speaker.upper()}") or ELEVEN_VOICES.get(speaker, ELEVEN_VOICES["agent"])
    body = json.dumps({
        "text": _speechify(text),
        "model_id": _env("ELEVEN_MODEL") or ELEVEN_MODEL,
        # mid stability + speaker boost reads as a natural, unhurried phone voice
        "voice_settings": {"stability": 0.45, "similarity_boost": 0.8,
                           "style": 0.0, "use_speaker_boost": True},
    }).encode()
    req = urllib.request.Request(
        ELEVEN_URL.format(vid=vid), data=body, method="POST",
        headers={"xi-api-key": api_key, "Content-Type": "application/json",
                 "Accept": "audio/pcm"})
    try:
        with urllib.request.urlopen(req, timeout=60) as r:
            return r.read()
    except urllib.error.HTTPError as e:
        detail = e.read().decode("utf-8", "replace")[:300]
        raise RuntimeError(f"ElevenLabs {e.code}: {detail}") from None


def _piper_voice(speaker: str):
    """Lazy-load (and cache) a Piper voice, downloading the model on first use."""
    name = _env(f"PIPER_VOICE_{speaker.upper()}") or PIPER_VOICES.get(speaker, PIPER_VOICES["agent"])
    if name not in _piper_cache:
        from huggingface_hub import hf_hub_download
        from piper import PiperVoice
        lang = name.split("-")[0]                       # e.g. en_US-amy-medium -> en_US
        quality = name.rsplit("-", 1)[-1]               # -> medium
        voice = name.split("-")[1]                      # -> amy
        base = f"{lang[:2]}/{lang}/{voice}/{quality}/{name}"
        onnx = hf_hub_download("rhasspy/piper-voices", base + ".onnx")
        hf_hub_download("rhasspy/piper-voices", base + ".onnx.json")
        _piper_cache[name] = PiperVoice.load(onnx)
    return _piper_cache[name]


def _piper_clip(text: str, speaker: str, path: Path) -> None:
    voice = _piper_voice(speaker)
    with wave.open(str(path), "wb") as wf:              # piper writes 22.05 kHz 16-bit mono
        voice.synthesize_wav(_speechify(text), wf)


def _write_wav(path: Path, pcm: bytes) -> None:
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(_RATE)
        w.writeframes(pcm)


def synth_clip(text: str, speaker: str, path: Path, engine: str, api_key: str = "") -> None:
    if engine == "eleven":
        _write_wav(path, _eleven_pcm(text, speaker, api_key))
    elif engine == "piper":
        _piper_clip(text, speaker, path)
    else:
        _say_clip(text, speaker, path)


# ── live turn-by-turn (used by run_demo --speak) ────────────────────────────

_LIVE_ENGINE = None  # resolved lazily so importing this module is cheap


def speak(speaker: str, text: str) -> None:
    """Say one turn out loud, blocking until it finishes. Engine picked once."""
    global _LIVE_ENGINE
    if _LIVE_ENGINE is None:
        _LIVE_ENGINE = resolve_engine("auto")
    key = _api_key()
    with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tf:
        clip = Path(tf.name)
    try:
        synth_clip(text, speaker, clip, _LIVE_ENGINE, key)
        subprocess.run(["afplay", str(clip)], check=True)
    finally:
        clip.unlink(missing_ok=True)


# ── whole-call render ───────────────────────────────────────────────────────

def _to_mp3(wav: Path, mp3: Path) -> bool:
    if shutil.which("ffmpeg") is None:
        return False
    done = subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                           "-c:a", "libmp3lame", "-q:a", "4", str(mp3)],
                          capture_output=True)
    return done.returncode == 0 and mp3.exists()


def render(transcript, out: str = "call.wav", engine: str = "auto") -> Path:
    """Synthesize every turn and stitch into one file, with silence between turns.

    Stdlib stitching: each clip is 16-bit mono PCM in a wav, concatenated via the
    `wave` module. `.mp3` output compresses at the end iff a working ffmpeg exists;
    otherwise you get the `.wav` and a note, never a crash.
    """
    engine = resolve_engine(engine)
    if engine == "say":
        _require("say")
    api_key = _api_key()

    turns = _normalize(transcript)
    if not turns:
        sys.exit("Nothing to render — the transcript is empty.")

    out_path = Path(out)
    wav_path = out_path.with_suffix(".wav")
    gap = b"\x00" * (2 * int(_RATE * _GAP_SECONDS))

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        with wave.open(str(wav_path), "wb") as out_wav:
            out_wav.setnchannels(1)
            out_wav.setsampwidth(2)
            out_wav.setframerate(_RATE)
            for i, (speaker, text) in enumerate(turns):
                clip = tmp / f"{i:03d}.wav"
                synth_clip(text, speaker, clip, engine, api_key)
                with wave.open(str(clip), "rb") as w:
                    out_wav.writeframes(w.readframes(w.getnframes()))
                out_wav.writeframes(gap)
                print(f"  {speaker:>5} [{engine}] {text[:56]}…")

    final = wav_path
    if out_path.suffix == ".mp3":
        if _to_mp3(wav_path, out_path):
            wav_path.unlink()
            final = out_path
        else:
            print("\n  (ffmpeg unavailable — keeping the .wav)")

    print(f"\nwrote {final} ({final.stat().st_size:,} bytes, {len(turns)} turns, {engine})")
    return final


def _live_transcript(scenario_idx: int):
    """Generate a fresh call. Needs ANTHROPIC_API_KEY (imports the agent stack)."""
    from call import run_call
    from payer_sim import PayerSim
    from scenarios import SCENARIOS
    sc = SCENARIOS[scenario_idx]
    print(f"Generating a fresh call — {sc['name']}\n")
    payer = PayerSim(sc["truth"], behavior=sc["behavior"])
    return run_call(payer, verbose=True)


def main():
    ap = argparse.ArgumentParser(description="Render a payer call to real audio.")
    ap.add_argument("-o", "--out", default="call.wav",
                    help="output path (.wav, or .mp3 if ffmpeg works)")
    ap.add_argument("--engine", choices=["auto", "eleven", "say"], default="auto",
                    help="voice engine (default: ElevenLabs if a key is set, else say)")
    ap.add_argument("--scripted", action="store_true",
                    help="replay the fixed adversarial call instead of a fresh one")
    ap.add_argument("--scenario", type=int, default=2,
                    help="which scenario to generate live (default 2, adversarial)")
    ap.add_argument("--no-play", action="store_true", help="don't play when done")
    args = ap.parse_args()

    engine = resolve_engine(args.engine)
    if engine == "say" and args.engine == "auto":
        print("No ElevenLabs key — falling back to the offline `say` voice.")
        print("For real voices, add ELEVENLABS_API_KEY to .env (see .env.example).\n")

    if args.scripted or not _env("ANTHROPIC_API_KEY"):
        from sample_call import SAMPLE_TRANSCRIPT
        if not args.scripted:
            print("No ANTHROPIC_API_KEY — replaying the scripted call.\n")
        transcript = SAMPLE_TRANSCRIPT
    else:
        transcript = _live_transcript(args.scenario)

    print()
    out = render(transcript, args.out, engine=engine)
    if not args.no_play:
        subprocess.run(["afplay", str(out)], check=True)


if __name__ == "__main__":
    main()
