# Beads Issues — Candidates Worth Doing

Curated from this repo's open/in-progress beads before beads was removed (2026-07-10).
Dropped as regenerable: test-suite/CI/ruff/CLAUDE.md/gitignore/type-annotation/version-pin
audit tasks, stale-branch cleanup. Bead IDs kept.

## Real bug worth fixing
- `WhisperModel` is instantiated inside `transcribe_file()`, which runs per file → in batch mode the model reloads from disk N times (loading is the most expensive step). Instantiate once and reuse across files. *(m4t)*
