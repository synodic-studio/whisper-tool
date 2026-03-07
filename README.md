# whisper-tool

Local audio transcription CLI using [faster-whisper](https://github.com/SYSTRAN/faster-whisper). Runs entirely offline on your machine.

## Setup

Requires Python 3.12+ and [uv](https://docs.astral.sh/uv/).

```bash
cd ~/Developer/whisper-tool
uv sync
```

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
```

## Options

| Flag | Default | Description |
|---|---|---|
| `--model` | `base` | Model size: `tiny`, `base`, `small`, `medium`, `large-v3` |
| `--compute-type` | `int8` | Precision: `int8`, `float16`, `float32` |
| `--json` | off | Output JSON with segment timestamps and metadata |
| `--words` | off | Include word-level timestamps (implies `--json`) |
| `--output-dir` | stdout | Write per-file results to a directory |

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
