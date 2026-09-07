#!/usr/bin/env python3
"""Filter pre-cut voice clips for piper training.

For each file in data/raw/ (mp3/wav/m4a/flac):
  - transcribe with faster-whisper (word timestamps, no_speech_prob, avg_logprob)
  - trim to the speech span (kills score/SFX run-in/run-out)
  - score background suspicion and flag clips needing a human listen
  - export accepted clips to data/wavs/ + rows in data/metadata.csv
  - copy flagged clips to data/review/ and write data/filter_report.csv

Only unambiguous cases are dropped automatically (too short / no speech).
Everything else lands in data/review/ for a manual listen before training.

Requires: .venv (faster-whisper, librosa, pydub) + ffmpeg.
"""
import argparse
import csv
import re
import shutil
import subprocess
import sys
from pathlib import Path

import librosa as L
import numpy as np
from faster_whisper import WhisperModel
from pydub import AudioSegment

ROOT_DIR = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT_DIR / "data" / "raw"
WAVS_DIR = ROOT_DIR / "data" / "wavs"
REVIEW_DIR = ROOT_DIR / "data" / "review"
SEPARATED_DIR = ROOT_DIR / "data" / "separated"
REPORT_PATH = ROOT_DIR / "data" / "filter_report.csv"

SR = 22050
HOP = 512
PAD_MS = 150
EXTENSIONS = ("*.mp3", "*.wav", "*.m4a", "*.flac", "*.ogg")

REPORT_FIELDS = [
    "source", "id", "text", "dur_s", "n_words", "logprob", "no_speech_prob",
    "min_word_prob", "mean_word_prob", "flatness", "bg_rms_ratio",
    "filename_match", "flags",
]


def load_np(audio: AudioSegment) -> np.ndarray:
    raw = np.frombuffer(audio.set_channels(1).set_frame_rate(SR).get_array_of_samples(),
                        dtype=np.int16).astype(np.float32) / 32768.0
    return raw.astype(np.float64) if raw.dtype != np.float64 else raw


def clip_metrics(y: np.ndarray, speech_mask: np.ndarray) -> dict:
    """Spectral/background features. speech_mask: bool per hop frame (True = inside a word)."""
    if len(y) < SR // 5:
        return {"flatness": float("nan"), "bg_rms_ratio": float("nan")}
    flat = L.feature.spectral_flatness(y=y, hop_length=HOP)[0]
    rms = L.feature.rms(y=y, hop_length=HOP)[0]
    n = min(len(flat), len(rms), len(speech_mask))
    flat, rms, mask = flat[:n], rms[:n], speech_mask[:n]
    speech_flatness = float(np.nanmean(flat[mask])) if mask.any() else float("nan")
    if mask.any() and (~mask).any():
        bg = float(np.mean(rms[~mask]))
        sp = float(np.median(rms[mask]))
        bg_ratio = (bg / sp) if sp > 1e-6 else float("nan")
    else:
        bg_ratio = float("nan")
    return {"flatness": speech_flatness, "bg_rms_ratio": bg_ratio}


def word_speech_mask(audio: AudioSegment, word_ts) -> np.ndarray:
    """Boolean speech mask (hop frames) built from whisper word timestamps."""
    dur_frames = int(np.ceil(len(audio) / 1000.0 * SR / HOP))
    mask = np.zeros(dur_frames, dtype=bool)
    for _, start, end in word_ts:
        s = int(start * SR / HOP)
        e = int(end * SR / HOP)
        mask[max(0, s):min(dur_frames, e + 1)] = True
    return mask


def filename_match_score(rel_name: str, text: str) -> float:
    """How well the filename's words appear in the transcript (0..1)."""
    fw = set(re.sub(r"[^a-z0-9 ]", " ", rel_name.lower()).split()) - {"s", "t"}
    tw = set(re.sub(r"[^a-z0-9 ]", " ", text.lower()).split())
    fw.discard("")
    if not fw:
        return float("nan")
    return len(fw & tw) / len(fw)


def main():
    ap = argparse.ArgumentParser(description="Filter pre-cut voice clips")
    ap.add_argument("--whisper", default="medium",
                    choices=["tiny", "small", "medium", "large-v3"], help="Whisper model size")
    ap.add_argument("--pad-ms", type=int, default=PAD_MS, help="Padding around speech (ms)")
    ap.add_argument("--logprob-bad", type=float, default=-0.9,
                    help="Below this avg word logprob -> flag 'low-confidence'")
    ap.add_argument("--no-speech-flag", type=float, default=0.35,
                    help="Above this no_speech_prob -> flag")
    ap.add_argument("--bg-flag", type=float, default=0.4,
                    help="Above this bg_rms_ratio -> flag 'background bleed'")
    ap.add_argument("--min-dur", type=float, default=1.5,
                    help="Drop clips shorter than this after trim (s)")
    ap.add_argument("--append", action="store_true",
                    help="Append to existing metadata.csv instead of overwriting")
    ap.add_argument("--separate-test", nargs="+", metavar="FILE",
                    help="Demucs A/B test: separate these raw clips into data/separated/ and exit")
    args = ap.parse_args()

    if args.separate_test:
        SEPARATED_DIR.mkdir(parents=True, exist_ok=True)
        for name in args.separate_test:
            src = RAW_DIR / name
            if not src.exists():
                print(f"ERROR: {src} not found")
                sys.exit(1)
            print(f"--- Separating {name} (htdemucs, vocals) ---")
            subprocess.check_call([
                sys.executable, "-m", "demucs", "--two-stems", "vocals", "--out",
                str(SEPARATED_DIR), "--jobs", "1", str(src),
            ])
            print(f"    -> {SEPARATED_DIR}/htdemucs/{src.stem}/vocals.wav")
        print("Done. Compare vocals.wav vs the original in data/raw/.")
        sys.exit(0)

    raw_files = sorted(f for pat in EXTENSIONS for f in RAW_DIR.glob(pat))
    if not raw_files:
        print(f"ERROR: No audio in {RAW_DIR}. Run scripts/fetch_audio.py first.")
        sys.exit(1)

    print(f"Loading Whisper {args.whisper} on CPU...")
    model = WhisperModel(args.whisper, device="cpu", compute_type="float32", cpu_threads=4)

    WAVS_DIR.mkdir(parents=True, exist_ok=True)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)

    # continue the optimus_NNNN counter past existing exports
    highest = -1
    for wav in WAVS_DIR.glob("optimus_*.wav"):
        stem = wav.stem.removeprefix("optimus_")
        if stem.isdigit():
            highest = max(highest, int(stem))
    counter = highest + 1

    rows, dropped = [], []
    for src in raw_files:
        print(f"\n== {src.name} ==")
        audio = AudioSegment.from_file(src)
        segments, info = model.transcribe(
            str(src), vad_filter=True, beam_size=5, word_timestamps=True,
            condition_on_previous_text=False,
        )
        segs = list(segments)

        if not segs:
            dropped.append((src.name, "NO SPEECH"))
            continue

        text = " ".join(s.text.strip() for s in segs if s.text).strip()
        no_speech_prob = float(np.mean([s.no_speech_prob if s.no_speech_prob is not None else 1.0
                                       for s in segs]))
        logprob = float(np.mean([s.avg_logprob for s in segs]))
        words = [(w, w.start, w.end) for s in segs if s.words for w in s.words]
        n_words = len(words)

        first, last = words[0][1], words[-1][2]
        start_ms = max(0, int((first * 1000) - args.pad_ms))
        end_ms = min(len(audio), int((last * 1000) + args.pad_ms))
        trimmed = audio[start_ms:end_ms]
        dur = len(trimmed) / 1000.0

        if dur < args.min_dur:
            dropped.append((src.name, f"TOO SHORT after trim ({dur:.1f}s)"))
            continue

        y = load_np(trimmed)
        mask = word_speech_mask(trimmed, words)
        feats = clip_metrics(y, mask)
        flatness = feats["flatness"]
        bg_ratio = feats["bg_rms_ratio"]

        w_probs = [w.probability if w.probability is not None else 0.5 for w, _, _ in words]
        min_wp = float(np.min(w_probs)) if w_probs else float("nan")
        mean_wp = float(np.mean(w_probs)) if w_probs else float("nan")
        fname_match = filename_match_score(src.stem, text)

        flags = []
        if logprob < args.logprob_bad:
            flags.append(f"low-confidence (logprob {logprob:.2f})")
        if no_speech_prob > args.no_speech_flag:
            flags.append(f"whisper no-speech {no_speech_prob:.2f}")
        if not np.isnan(bg_ratio) and bg_ratio > args.bg_flag:
            flags.append(f"background bleed (bg/spk {bg_ratio:.2f})")
        if not np.isnan(flatness) and flatness > 0.15:
            flags.append(f"noisy spectrum (flatness {flatness:.2f})")
        if not np.isnan(fname_match) and fname_match < 0.5:
            flags.append(f"transcript!=filename ({fname_match:.0%})")

        base_id = f"optimus_{counter:04d}"
        counter += 1
        wav_path = WAVS_DIR / f"{base_id}.wav"
        trimmed.set_channels(1).set_frame_rate(SR).set_sample_width(2).export(str(wav_path), format="wav")

        rows.append({
            "source": src.name, "id": base_id, "text": text, "dur_s": round(dur, 2),
            "n_words": n_words, "logprob": round(logprob, 3), "no_speech_prob": round(no_speech_prob, 3),
            "min_word_prob": round(min_wp, 3) if not np.isnan(min_wp) else "",
            "mean_word_prob": round(mean_wp, 3) if not np.isnan(mean_wp) else "",
            "flatness": round(flatness, 4) if not np.isnan(flatness) else "",
            "bg_rms_ratio": round(bg_ratio, 3) if not np.isnan(bg_ratio) else "",
            "filename_match": round(fname_match, 2) if not np.isnan(fname_match) else "",
            "flags": "; ".join(flags),
        })
        print(f"  kept {wav_path.name} ({dur:.1f}s)  logprob={logprob:.2f} no_speech={no_speech_prob:.2f}"
              f" bg={bg_ratio:.2f} flags={flags or '-'}")
        if flags:
            review_copy = REVIEW_DIR / wav_path.name
            shutil.copy2(wav_path, review_copy)
            print(f"  -> FLAGGED: copied to {review_copy} for review")

    # write report + metadata
    with open(REPORT_PATH, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=REPORT_FIELDS)
        w.writeheader()
        w.writerows(sorted(rows, key=lambda r: r.get("dur_s", 0), reverse=True))
    mode = "a" if args.append else "w"
    with open(ROOT_DIR / "data" / "metadata.csv", mode, encoding="utf-8", newline="") as f:
        for r in rows:
            f.write(f"{r['id']}|{r['text']}\n")

    print("\n=== SUMMARY ===")
    print(f"kept: {len(rows)}  dropped: {len(dropped)}")
    for name, why in dropped:
        print(f"  DROP {name}: {why}")
    flagged = [r for r in rows if r["flags"]]
    print(f"flagged for review: {len(flagged)} (copies in data/review/, see {REPORT_PATH.name})")
    print("\nNext: listen to data/review/, fix transcripts in data/metadata.csv,\n"
          "then run scripts/preprocess.py and scripts/train.sh.")
    print("Tip: scripts/test.sh  (after a few epochs) compares base vs latest.")


if __name__ == "__main__":
    main()