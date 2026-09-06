# optimus-piper-voice

A Piper TTS voice clone of **Optimus Prime** (Peter Cullen), fine-tuned on
`rhasspy/piper`'s `en_US-lessac-medium` base model. Template: `ahsoka-piper-voice/`.

> Copyright note: the training audio is dialogue from Paramount/Hasbro productions.
> This project is for research/experimentation; the model may be subject to takedown.

## Pipeline

Run from `optimus-piper-voice/` (all scripts require `.venv` + `piper-source`):

1. Setup: `scripts/setup.sh` (requires `uv`) — clones rhasspy/piper, installs deps, builds Cython.
2. Base model: download `en_US-lessac-medium` checkpoint →
   `checkpoints/lessac-medium.ckpt` (846 MB, from
   `https://huggingface.co/datasets/rhasspy/piper-checkpoints/resolve/main/en/en_US/lessac/medium/epoch=2164-step=1355540.ckpt`).
3. Fetch: `scripts/fetch_audio.py "<youtube-url>" <name>` → `data/raw/*.wav` (22050 Hz mono s16).
4. Slice: `scripts/slice_audio.py [whisper-model]` → transcribes with faster-whisper,
   slices 1.5–12 s, writes `data/metadata.csv` (`ID|Text`) + `data/wavs/*.wav`.
   **Manually QA the text** (proper nouns like Autobots/Decepticons/Cybertron/Megatron;
   drop clips with music/SFX or non-Optimus speech).
5. Preprocess: `scripts/preprocess.py` → phonemizes into `checkpoints/`,
   writes `config.json` + `dataset.jsonl` (cached audio in gitignored `checkpoints/cache/`),
   symlinks `data/wavs → checkpoints/wavs`.
6. Train: `scripts/train.sh` — CPU fine-tune (lr 1e-5, batch 4). Auto-resumes the newest
   `lightning_logs` checkpoint, or bootstraps from `lessac-medium.ckpt` on first run.
7. Test: `scripts/test.sh [-p "phrase"]` → compares base vs latest into `test_results/*.wav`.
8. Publish: `scripts/publish.sh` (after `huggingface-cli login`) → exports newest ckpt to ONNX,
   uploads to `crazygiscool/optimus-piper-voice`, commits `published/`.

## Notes

- `checkpoints/*.ckpt`, `checkpoints/cache/`, `checkpoints/lightning_logs/`,
  `published/*.onnx`, `test_results/`, `data/raw/`, `piper-source/`, `.venv/` are gitignored.
- The committed `checkpoints/dataset.jsonl` embeds absolute wav paths — re-run
  `scripts/preprocess.py` after moving/cloning the repo.
- Modern-era sources only (Bayverse films + Transformers: Prime) for a consistent voice.