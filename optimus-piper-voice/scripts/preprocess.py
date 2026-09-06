import os
import subprocess
import sys
from pathlib import Path

# --- CONFIGURATION ---
ROOT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT_DIR / "data"
WAVS_DIR = DATA_DIR / "wavs"
METADATA_FILE = DATA_DIR / "metadata.csv"
OUTPUT_DIR = ROOT_DIR / "checkpoints"
PIPER_PYTHON = ROOT_DIR / "piper-source" / "src" / "python"


def run_preprocess():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Symlink wavs into checkpoints
    target_wavs = OUTPUT_DIR / "wavs"

    if target_wavs.is_symlink() and not target_wavs.exists():
        target_wavs.unlink()

    if not target_wavs.exists():
        if not WAVS_DIR.exists():
            print(f"ERROR: Source wavs directory not found at {WAVS_DIR}")
            return
        print(f"Creating symlink: {WAVS_DIR} -> {target_wavs}")
        os.symlink(str(WAVS_DIR.resolve()), str(target_wavs))
    else:
        print("Symlink already exists, skipping...")

    # 2. Run Piper Preprocess
    print("--- Starting Piper Phonemization ---")

    env = os.environ.copy()
    env["PYTHONPATH"] = f"{PIPER_PYTHON}:{env.get('PYTHONPATH', '')}"

    cmd = [
        sys.executable,
        "-m",
        "piper_train.preprocess",
        "--input-dir",
        str(DATA_DIR),
        "--output-dir",
        str(OUTPUT_DIR),
        "--dataset-format",
        "ljspeech",
        "--language",
        "en-us",
        "--sample-rate",
        "22050",
    ]

    try:
        subprocess.run(cmd, check=True, cwd=str(ROOT_DIR), env=env)
        print(f"\nSUCCESS: dataset.jsonl.gz has been created in {OUTPUT_DIR}")
    except subprocess.CalledProcessError as e:
        print(f"\nERROR: Preprocessing failed. {e}")


if __name__ == "__main__":
    run_preprocess()
