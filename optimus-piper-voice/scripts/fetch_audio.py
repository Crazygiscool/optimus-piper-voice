import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"


def download_audio(url, output_name):
    """
    Downloads audio from a URL (e.g. YouTube) and converts it to Piper-ready WAV.
    Piper requirements: 22050 Hz, mono, 16-bit PCM.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)

    # Template saves the file exactly as 'output_name' (avoids yt-dlp 'word-separator' error).
    output_template = str(RAW_DIR / f"{output_name}.%(ext)s")

    command = [
        "yt-dlp",
        "-x",  # Extract audio
        "--audio-format",
        "wav",
        "-o",
        output_template,
        "--postprocessor-args",
        "ffmpeg:-ar 22050 -ac 1 -sample_fmt s16",
        url,
    ]

    print(f"--- Fetching audio from: {url} ---")
    try:
        subprocess.run(command, check=True)
        print(f"--- Success! Check {RAW_DIR} for {output_name}.wav ---")
    except subprocess.CalledProcessError as e:
        print(f"--- Error downloading: {e} ---")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(f"Usage: {Path(__file__).name} <url> [name]")
        sys.exit(1)
    url = sys.argv[1]
    name = sys.argv[2] if len(sys.argv) > 2 else "optimus"
    download_audio(url, name)