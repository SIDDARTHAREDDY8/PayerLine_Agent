"""Give the GitHub Pages showcase a voice.

Pages is static, so the audio has to travel inside the HTML. This reads the
scripted call turns straight out of showcase.html, voices each with Piper
(applying the same number-normalization as the app), encodes them to small MP3
data URIs, and injects them into the `// @@AUDIO@@` slot so each clip plays in
sync with its chat bubble. Re-run after editing the transcript.

    python build_showcase_audio.py        # needs piper-tts + ffmpeg
"""
import base64
import re
import subprocess
import tempfile
from pathlib import Path

import voice

HTML = Path("showcase.html")


def turns_from_html(src: str):
    # Each entry looks like {who:"agent", label:"…", text:"…", …}; text has no
    # embedded double-quotes, so a simple capture is safe.
    return re.findall(r'\{who:"(agent|payer)",\s*label:"[^"]*",\s*text:"([^"]*)"', src)


def mp3_data_uri(speaker: str, text: str) -> str:
    with tempfile.TemporaryDirectory() as tmp:
        wav, mp3 = Path(tmp) / "c.wav", Path(tmp) / "c.mp3"
        voice.synth_clip(text, speaker, wav, engine="piper")
        subprocess.run(["ffmpeg", "-y", "-loglevel", "error", "-i", str(wav),
                        "-ac", "1", "-ar", "22050", "-b:a", "32k", str(mp3)], check=True)
        b64 = base64.b64encode(mp3.read_bytes()).decode()
    return "data:audio/mpeg;base64," + b64


def main():
    src = HTML.read_text()
    turns = turns_from_html(src)
    if not turns:
        raise SystemExit("No turns found in showcase.html.")

    uris = []
    for i, (who, text) in enumerate(turns):
        uris.append(mp3_data_uri(who, text))
        print(f"  [{i+1}/{len(turns)}] {who:5} {len(uris[-1])//1000}k  {text[:48]}…")

    array = "const CALL_AUDIO = [\n" + "".join(f'    "{u}",\n' for u in uris) + "  ];  // @@AUDIO@@"
    new = re.sub(r"const CALL_AUDIO = \[[\s\S]*?\];  // @@AUDIO@@", array, src, count=1)
    HTML.write_text(new)
    total = sum(len(u) for u in uris)
    print(f"\ninjected {len(uris)} clips into showcase.html ({total//1000}k base64 total)")


if __name__ == "__main__":
    main()
