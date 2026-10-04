---
license: cc-by-nc-4.0
language:
  - en
base_model:
  - speaches-ai/piper-en_US-lessac-medium
pipeline_tag: text-to-speech
tags:
  - piper
  - text-to-speech
  - voice-cloning
---

# Optimus Prime Piper Voice

An unofficial English text-to-speech voice model fine-tuned from Piper's
`en_US-lessac-medium` voice to resemble the Optimus Prime character voice
associated with Peter Cullen. It exports as a Piper-compatible ONNX model and
uses the VITS architecture.

This project is not affiliated with or endorsed by Peter Cullen, Hasbro,
Paramount, or the Transformers franchise rights holders.

## Model Details

- **Developer:** Crazygiscool
- **Task:** English text-to-speech
- **Architecture:** Piper VITS
- **Base model:** `speaches-ai/piper-en_US-lessac-medium`
- **Sample rate:** 22,050 Hz
- **Phonemizer:** eSpeak, `en-us`
- **Artifacts:** ONNX model, Piper JSON config, and PyTorch training checkpoint
- **License stated for this model:** CC BY-NC 4.0

The license above does not grant rights to the Transformers character, any
performer's voice or likeness, or the copyrighted recordings used to create the
training clips. See **Rights and Responsible Use** before using or sharing
outputs.

## Intended Use

For non-commercial research, education, and clearly labeled fan demonstrations,
provided the user has the rights and permissions needed for their use. Generated
audio should be disclosed as synthetic and unofficial.

## Out-of-Scope Use

- Commercial use without all necessary permissions.
- Deceptive impersonation, fraud, or falsely attributing generated speech to a
  real person or rights holder.
- Suggesting that the model or its outputs are official, endorsed, or performed
  by Peter Cullen or the Transformers rights holders.
- Uses that violate applicable law, platform rules, or the rights of others.

## Rights and Responsible Use

The training clips were selected from dialogue in Transformers films and
television, including Bayverse films and *Transformers: Prime*. Those recordings
are copyrighted by their respective rights holders. The audio clips and training
dataset are not included in this repository. The model's CC BY-NC 4.0 label is
not a representation that the uploader owns or has cleared rights to the source
recordings, character, trademarks, or voice likeness. Users are responsible for
obtaining any permissions required for their intended use; the model may be
subject to takedown or other rights claims.

## Limitations

- The model synthesizes English (`en-us`) only and may mispronounce names,
  unusual spellings, or words outside its training domain.
- Prosody, emotion, and pronunciation may differ from the character or source
  performer; outputs are not recordings of the performer.
- Evaluation was limited to project-level listening comparisons against the
  base voice. No standardized benchmark, independent evaluation, or quantitative
  quality metric is reported.
- The model may reproduce biases or artifacts present in the base voice and
  training material.

## Usage

Download `optimus-final.onnx` and its companion
`optimus-final.onnx.json`, install a compatible Piper runtime, then synthesize
audio with:

```bash
printf '%s\n' 'Freedom is the right of all sentient beings.' | \
  piper --model optimus-final.onnx \
        --config optimus-final.onnx.json \
        --output_file output.wav
```

The output is 22,050 Hz mono audio. Check the Piper runtime's documentation for
installation and version-specific options.

## Training

The model was fine-tuned from the Lessac medium checkpoint using Piper's VITS
training pipeline. The project configuration records a learning rate of
`1e-5`, batch size `4`, CPU training, eSpeak `en-us` phonemization, and a
22,050 Hz sample rate. Training audio was sliced into clips, transcribed, and
manually reviewed for text and unwanted music, sound effects, or non-target
speech. Exact dataset size, training duration, and quantitative evaluation
results are not reported. The audio and dataset metadata are not distributed
with the model.

## Source

- [Project repository](https://github.com/Crazygiscool/optimus-piper-voice)
- [Base voice](https://huggingface.co/speaches-ai/piper-en_US-lessac-medium)