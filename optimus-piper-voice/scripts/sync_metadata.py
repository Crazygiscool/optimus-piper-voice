#!/usr/bin/env python3
"""Sync data/metadata.csv with the wavs actually present in data/wavs.

After manually deleting bad clips (e.g. ones with background voice bleed),
run this to drop their rows from the transcript file so preprocess/train only
see surviving clips.
"""
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
WAVS_DIR = ROOT_DIR / "data" / "wavs"
METADATA_PATH = ROOT_DIR / "data" / "metadata.csv"

existing = {p.stem for p in WAVS_DIR.glob("*.wav")}
kept, removed = [], []
for line in METADATA_PATH.read_text(encoding="utf-8").splitlines():
    line = line.strip()
    if not line:
        continue
    clip_id = line.split("|", 1)[0]
    if clip_id in existing:
        kept.append(line)
    else:
        removed.append(line)

METADATA_PATH.write_text("\n".join(kept) + "\n", encoding="utf-8")
print(f"metadata.csv: {len(kept)} rows kept, {len(removed)} removed")
for line in removed:
    print(f"  - {line.split('|', 1)[0]}")