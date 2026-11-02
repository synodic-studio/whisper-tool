# whisper-tool

Local audio transcription CLI using faster-whisper or Apple Silicon MLX Whisper. Runs entirely offline after models are cached.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/Developer/whisper-tool
uv sync
```

For the optional Apple Silicon backend, install its exactly pinned packages and cache the exactly pinned large-v3 model before transcription:

```bash
uv sync --extra mlx
uv run hf download mlx-community/whisper-large-v3-mlx \
  --revision 49e6aa286ad60c14352c404340ded53710378a11
```

MLX transcription resolves that immutable revision from the local Hugging Face cache only. It never downloads or follows the repository's moving `main` revision during transcription.

## Usage

```bash
# Plain text output
uv run transcribe recording.m4a

# JSON with segment timestamps
uv run transcribe recording.m4a --json

# Word-level timestamps (implies --json)
uv run transcribe recording.m4a --words

# Batch transcription to a directory
uv run transcribe *.m4a --output-dir ./transcripts/

# Use a larger model for better accuracy
uv run transcribe recording.m4a --model large-v3

# Use Apple Silicon acceleration with the pinned MLX large-v3 model
uv run transcribe recording.m4a --model large-v3 --backend mlx
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--backend` | `faster-whisper` | `faster-whisper`, `mlx`, or `auto` |
| `--model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--compute-type` | `auto` | `int8` for faster-whisper or `float16` for MLX; explicit `int8`, `float16`, `float32` |
| `--json` | off | Output JSON with segment timestamps and metadata |
| `--words` | off | Include word-level timestamps (implies `--json`) |
| `--output-dir` | stdout | Write per-file results to a directory |

`auto` selects MLX only for `large-v3` on Darwin arm64 when `mlx-whisper` is importable and hotwords are not requested. All other cases retain faster-whisper. The MLX backend requires `mlx-whisper==0.4.3`, `mlx==0.32.4`, the immutable model revision above, and float16 inference; it fails closed if the installed runtime differs. Passing an explicit non-float16 compute type to MLX is an error. MLX JSON includes an exact `transcription_binding` object for evidence provenance, while `language_probability` is `null` because mlx-whisper does not return the detected-language probability. MLX does not support `--hotwords` and fails explicitly rather than silently ignoring it.

## Output formats

**Plain text** (default) prints the transcription to stdout. When multiple files are given, each is preceded by a `--- filename ---` divider.

**JSON** (`--json`) returns structured output:

```json
{
  "file": "recording.m4a",
  "language": "en",
  "language_probability": 0.98,
  "duration_seconds": 62.4,
  "text": "Full transcribed text...",
  "segments": [
    { "start": 0.0, "end": 4.8, "text": "First segment..." }
  ],
  "stats": {
    "segment_count": 12,
    "word_count": null,
    "transcription_ms": 3200
  }
}
```

With `--words`, each segment includes a `words` array containing per-word start/end times and confidence probabilities. Multiple files produce a JSON array.

## Models

Larger models are slower but more accurate. The `base` default is a good starting point. Bump to `small` or `medium` for noisy audio or accented speech. `large-v3` gives the best quality at significant compute cost.
