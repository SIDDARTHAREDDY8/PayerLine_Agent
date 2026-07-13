"""Pre-render one real call so the deployed demo plays instantly and for free.

Runs the live pipeline once (spends a little API credit), compresses the audio
to a small .m4a with macOS `afconvert` (no ffmpeg), and writes a JSON bundle the
web app loads to show the transcript, result, flags, and triage with zero API
calls. Re-run whenever you want a fresh canned call.

    python build_demo_asset.py            # scenario 2 (adversarial) -> assets/
    python build_demo_asset.py 0          # a different scenario
"""
import base64
import json
import subprocess
import sys
from pathlib import Path

from demo_pipeline import generate

ASSETS = Path("assets")


def main():
    idx = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    ASSETS.mkdir(exist_ok=True)

    wav = ASSETS / "demo_call.wav"
    bundle = generate(scenario_idx=idx, audio_out=str(wav), engine="eleven")

    # compress wav -> m4a (AAC) with the built-in macOS encoder; small enough to
    # commit, and browsers/Gradio play it natively.
    m4a = ASSETS / "demo_call.m4a"
    subprocess.run(["afconvert", "-f", "m4af", "-d", "aac", str(wav), str(m4a)],
                   check=True)
    wav.unlink(missing_ok=True)

    # Commit the audio as base64 text — HF Spaces rejects raw binaries not in LFS.
    b64 = ASSETS / "demo_call.m4a.b64"
    b64.write_text(base64.b64encode(m4a.read_bytes()).decode())
    m4a.unlink(missing_ok=True)                     # raw .m4a is gitignored
    bundle["audio"] = b64.name

    (ASSETS / "demo_call.json").write_text(json.dumps(bundle, indent=2))
    print(f"\nwrote {b64} ({b64.stat().st_size:,} bytes)")
    print(f"wrote {ASSETS/'demo_call.json'}")
    print(f"route: {bundle['triage']['route']} · flags: {len(bundle['flags'])}")


if __name__ == "__main__":
    main()
