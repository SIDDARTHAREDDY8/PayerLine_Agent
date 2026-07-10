"""The voice layer — makes a call audible instead of just readable.

Voice is the product; text turns were only ever a stand-in. This renders a
transcript to real speech using macOS `say` (offline, no API key, no account)
with a distinct voice per speaker, and stitches the turns into one MP3 you can
play, embed, or hand to someone who won't read the code.

Deliberately NOT telephony. The payer here is a simulator, so dialing a real
number would prove nothing the text loop doesn't already prove — see the
limitations section of the README. This makes the existing loop hearable; the
telephony seam stays `call.py`.

    python voice.py                 # render the scripted offline call -> call.mp3
    python voice.py --play          # ...and play it when done
    python run_demo.py 2 --speak    # speak a live generated call, turn by turn
"""
import argparse
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

# Two clearly different voices so you can tell who's talking with your eyes shut.
VOICES = {"agent": "Samantha", "payer": "Tessa"}

# `say` infers the container from the extension; pinning the data format keeps
# every clip identical, which is what ffmpeg's concat demuxer requires.
_FORMAT = "LEF32@22050"
_GAP_SECONDS = 0.35


def _require(*tools: str) -> None:
    missing = [t for t in tools if shutil.which(t) is None]
    if missing:
        sys.exit(f"missing {', '.join(missing)} — the voice layer needs "
                 f"macOS `say` and ffmpeg (brew install ffmpeg).")


def _normalize(transcript) -> list[tuple[str, str]]:
    """Accept run_call's [{'speaker','text'}] or demo_offline's [(SPEAKER, text)]."""
    turns = []
    for t in transcript:
        speaker, text = (t["speaker"], t["text"]) if isinstance(t, dict) else t
        text = text.strip()
        if text:
            turns.append((speaker.lower(), text))
    return turns


def speak(speaker: str, text: str) -> None:
    """Say one turn out loud, blocking until it finishes."""
    subprocess.run(["say", "-v", VOICES.get(speaker, "Samantha"), text], check=True)


def _clip(text: str, voice: str, path: Path) -> None:
    subprocess.run(["say", "-v", voice, "-o", str(path),
                    f"--data-format={_FORMAT}", text], check=True)


def _silence(path: Path) -> None:
    subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "lavfi",
                    "-i", "anullsrc=r=22050:cl=mono", "-t", str(_GAP_SECONDS),
                    "-c:a", "pcm_f32le", str(path)], check=True)


def render(transcript, out: str = "call.mp3") -> Path:
    """Render every turn to speech and stitch them into one MP3, with a beat of
    silence between turns so it sounds like a conversation, not a run-on."""
    _require("say", "ffmpeg")
    turns = _normalize(transcript)
    if not turns:
        sys.exit("Nothing to render — the transcript is empty.")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        gap = tmp / "gap.wav"
        _silence(gap)

        clips = []
        for i, (speaker, text) in enumerate(turns):
            clip = tmp / f"{i:03d}.wav"
            _clip(text, VOICES.get(speaker, "Samantha"), clip)
            clips += [clip, gap]
            print(f"  {speaker:>5} ({VOICES.get(speaker, '?'):8}) {text[:58]}…")

        listing = tmp / "clips.txt"
        listing.write_text("".join(f"file '{c}'\n" for c in clips))

        out_path = Path(out)
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-f", "concat",
                        "-safe", "0", "-i", str(listing),
                        "-c:a", "libmp3lame", "-q:a", "4", str(out_path)],
                       check=True)

    size = out_path.stat().st_size
    print(f"\nwrote {out_path} ({size:,} bytes, {len(turns)} turns)")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="Render a payer call to audio.")
    ap.add_argument("-o", "--out", default="call.mp3", help="output mp3 path")
    ap.add_argument("--play", action="store_true", help="play it when done")
    args = ap.parse_args()

    # The scripted adversarial call — no API key needed to hear the pushback.
    from demo_offline import SAMPLE_TRANSCRIPT

    print("Rendering the ADVERSARIAL call (rep misstates the deductible)…\n")
    out = render(SAMPLE_TRANSCRIPT, args.out)
    if args.play:
        subprocess.run(["afplay", str(out)], check=True)


if __name__ == "__main__":
    main()
