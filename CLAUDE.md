# CLAUDE.md — whisper-tool

## What This Is
Local audio transcription CLI using faster-whisper. Runs entirely offline on the local machine with no cloud dependencies.

## Tech Stack
- Language: Python 3.12+
- Library: faster-whisper (SYSTRAN)
- Package manager: uv
- Build backend: hatchling
- CLI entry point: `transcribe` (mapped to `whisper_tool.cli:main`)

## Build & Test
```bash
uv sync              # Install dependencies
uv run transcribe recording.m4a                    # Plain text output
uv run transcribe recording.m4a --json             # JSON with timestamps
uv run transcribe recording.m4a --words            # Word-level timestamps
uv run transcribe *.m4a --output-dir ./transcripts/ # Batch mode
uv run transcribe recording.m4a --model large-v3   # Larger model
```

## Project Structure
```
src/whisper_tool/
  __init__.py
  cli.py          # CLI entry point, argument parsing, transcription logic
pyproject.toml     # Project metadata, dependencies, script entry points
uv.lock            # Locked dependencies
CODEOWNERS         # Review gate configuration
```

## Key Conventions
- Single-module package under `src/whisper_tool/`
- Default model is `base` (int8 compute type) — bump to `small`/`medium`/`large-v3` for noisy audio
- Output modes: plain text (default), JSON with segment timestamps (`--json`), word-level timestamps (`--words`)
- Batch processing outputs per-file results to `--output-dir`
- No tests currently in repo
