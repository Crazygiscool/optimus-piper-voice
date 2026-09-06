import csv
import sys
from pathlib import Path

from faster_whisper import WhisperModel
from pydub import AudioSegment
from tqdm import tqdm

# --- CONFIG ---
ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
WAVS_DIR = ROOT_DIR / "data" / "wavs"
METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"

# Whisper size for transcription. 'medium' is a good default on CPU.
MODEL_SIZE = sys.argv[1] if len(sys.argv) > 1 else "medium"

WAVS_DIR.mkdir(parents=True, exist_ok=True)

print(f"Loading Whisper {MODEL_SIZE} model on CPU (float32 fallback)...")
model = WhisperModel(MODEL_SIZE, device="cpu", compute_type="float32", cpu_threads=4)


def find_first_free_index(prefix="optimus_"):
    """Continue the ID counter past any already-exported clips."""
    highest = -1
    for wav in WAVS_DIR.glob(f"{prefix}*.wav"):
        stem = wav.stem[len(prefix):]
        if stem.isdigit():
            highest = max(highest, int(stem))
    return highest + 1


def process_file(audio_path, counter):
    """Transcribe one raw file and slice valid segments into data/wavs.
    Returns the updated clip counter."""
    print(f"Loading {audio_path.name}...")
    audio = AudioSegment.from_file(audio_path)

    print("Transcribing and segmenting (this will take a while on CPU)...")
    segments, info = model.transcribe(str(audio_path), vad_filter=True, beam_size=5)

    total_duration = round(info.duration, 2)
    pbar = tqdm(total=total_duration, unit="sec", desc="Slicing Audio")

    with open(METADATA_PATH, "a", encoding="utf-8", newline="") as f:
        writer = csv.writer(f, delimiter="|")

        for segment in segments:
            pbar.update(segment.end - pbar.n)

            # Piper sweet spot: 1.5s to 12s
            duration = segment.end - segment.start
            if duration < 1.5 or duration > 12.0:
                continue

            base_name = f"optimus_{counter:04d}"
            counter += 1
            wav_path = WAVS_DIR / f"{base_name}.wav"

            start_ms = segment.start * 1000
            end_ms = segment.end * 1000
            clip = audio[start_ms:end_ms]

            # Piper specs: 22050 Hz, mono, 16-bit PCM
            clip = clip.set_channels(1).set_frame_rate(22050).set_sample_width(2)
            clip.export(str(wav_path), format="wav")

            # Format: ID|Text
            writer.writerow([base_name, segment.text.strip()])

    pbar.close()
    print(f"Processed {audio_path.name}: snippets so far -> {counter}")
    return counter


def create_dataset():
    raw_files = sorted(RAW_DIR.glob("*.wav"))
    if not raw_files:
        print(f"ERROR: No audio found in {RAW_DIR}. Run scripts/fetch_audio.py first.")
        return

    counter = find_first_free_index()
    for raw_file in raw_files:
        counter = process_file(raw_file, counter)

    print(f"\nSuccess! Dataset in {WAVS_DIR} + {METADATA_PATH}")
    print(f"Total clips so far: {counter}")


if __name__ == "__main__":
    create_dataset()