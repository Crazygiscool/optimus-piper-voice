#!/usr/bin/env python3
"""Test script: synthesizes a phrase using base and latest checkpoints."""
import argparse
import json
import glob
import os
import sys
from pathlib import Path

import torch
import soundfile as sf
import piper_phonemize

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PIPER_PYTHON = PROJECT_ROOT / "piper-source" / "src" / "python"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUT_DIR = PROJECT_ROOT / "test_results"
CONFIG_PATH = CHECKPOINTS_DIR / "config.json"
BASE_CKPT = CHECKPOINTS_DIR / "optimus-final.ckpt"
DEFAULT_PHRASE = "Freedom is the right of all sentient beings."

sys.path.insert(0, str(PIPER_PYTHON))
from piper_train.vits.lightning import VitsModel


def find_latest_checkpoint():
    ckpts = glob.glob(str(CHECKPOINTS_DIR / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt"))
    ckpts = [c for c in ckpts if not Path(c).name.startswith("best-gen-loss-")]
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def find_best_loss_checkpoint():
    ckpts = glob.glob(str(CHECKPOINTS_DIR / "lightning_logs" / "version_*" / "checkpoints" / "best-gen-loss-*.ckpt"))
    if not ckpts:
        return None
    return max(ckpts, key=os.path.getmtime)


def resolve_checkpoint(spec: str):
    """Resolve a --checkpoint value: existing path, path relative to the project,
    or a basename/glob searched across lightning_logs version dirs (newest match)."""
    p = Path(spec)
    for candidate in (p, PROJECT_ROOT / p):
        if candidate.exists():
            return str(candidate.resolve())
    pattern = p.name if p.is_absolute() else spec
    matches = glob.glob(
        str(CHECKPOINTS_DIR / "lightning_logs" / "version_*" / "checkpoints" / pattern)
    )
    if not matches and "*" not in spec:
        matches = [
            m
            for m in glob.glob(
                str(CHECKPOINTS_DIR / "lightning_logs" / "version_*" / "checkpoints" / "*.ckpt")
            )
            if Path(m).name == spec
        ]
    if not matches:
        return None
    return max(matches, key=os.path.getmtime)


def text_to_phoneme_ids(text, language="en-us"):
    phonemes = piper_phonemize.phonemize_espeak(text, language)
    flat = [p for seq in phonemes for p in seq]
    return piper_phonemize.phoneme_ids_espeak(flat)


def synthesize(ckpt_path, label, phrase, language="en-us"):
    print(f"--- {label} ---")
    print(f"    {ckpt_path}")

    model = VitsModel.load_from_checkpoint(ckpt_path, dataset=None)
    model_g = model.model_g
    model_g.eval()

    with torch.no_grad():
        model_g.dec.remove_weight_norm()

        ids = text_to_phoneme_ids(phrase, language)
        x = torch.LongTensor([ids])
        x_lengths = torch.LongTensor([len(ids)])
        scales = torch.FloatTensor([0.667, 1.0, 0.8])

        audio, _, _, _ = model_g.infer(
            x, x_lengths,
            noise_scale=scales[0],
            length_scale=scales[1],
            noise_scale_w=scales[2],
        )
        audio = audio[0, 0].cpu().numpy()

    out_path = OUT_DIR / f"{label}.wav"
    sf.write(str(out_path), audio, 22050)
    duration = len(audio) / 22050
    print(f"    Saved: {out_path.name} ({duration:.2f}s)")
    return out_path


def main():
    parser = argparse.ArgumentParser(description="Test TTS model with a phrase")
    parser.add_argument("-p", "--phrase", default=DEFAULT_PHRASE, help="Text to synthesize")
    parser.add_argument("-l", "--language", default="en-us", help="Language code (default: en-us)")
    parser.add_argument(
        "-b",
        "--best",
        action="store_true",
        help="Use the lowest-loss checkpoint (best-gen-loss-*) instead of the latest",
    )
    parser.add_argument(
        "-c",
        "--checkpoint",
        default=None,
        help="Path or glob name of a specific checkpoint (e.g. 'epoch=36*', 'best-gen-loss-*'); overrides --best",
    )
    args = parser.parse_args()

    phrase = args.phrase
    language = args.language

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    with open(CONFIG_PATH) as f:
        config = json.load(f)

    print(f'Phrase: "{phrase}"')
    print(f"Output: {OUT_DIR}/\n")

    if BASE_CKPT.exists():
        synthesize(BASE_CKPT, "base_model", phrase, language)
    else:
        print(f"Base model not found: {BASE_CKPT}\n")

    if args.checkpoint:
        if args.best:
            print("(--best ignored; --checkpoint takes precedence)")
        synth_ckpt = resolve_checkpoint(args.checkpoint)
        label = Path(synth_ckpt).stem if synth_ckpt else "custom"
        if not synth_ckpt:
            print(f"\nCould not resolve checkpoint: {args.checkpoint}")
    else:
        synth_ckpt = find_best_loss_checkpoint() if args.best else find_latest_checkpoint()
        label = "best" if args.best else "latest"

    if synth_ckpt and Path(synth_ckpt).resolve() != Path(BASE_CKPT).resolve():
        synthesize(synth_ckpt, label, phrase, language)
    elif args.checkpoint:
        pass
    elif args.best:
        print("\nNo best-loss checkpoint found (best-gen-loss-*).")
    else:
        print("\nNo newer checkpoint found than base model.")

    print("\nDone. Listen to results in test_results/")


if __name__ == "__main__":
    main()
