"""Local audio transcription CLI using faster-whisper."""

import argparse
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel


def transcribe_file(
    audio_path: Path,
    model_size: str = "base",
    compute_type: str = "int8",
    word_timestamps: bool = False,
) -> dict:
    """Transcribe a single audio file. Returns dict with text, segments, and metadata."""
    model = WhisperModel(model_size, compute_type=compute_type)
    start = time.monotonic()
    segments_iter, info = model.transcribe(
        str(audio_path), word_timestamps=word_timestamps
    )

    segments = []
    full_text_parts = []
    total_words = 0

    for seg in segments_iter:
        seg_data = {
            "start": round(seg.start, 2),
            "end": round(seg.end, 2),
            "text": seg.text.strip(),
        }
        if word_timestamps and seg.words:
            seg_data["words"] = [
                {
                    "start": round(w.start, 2),
                    "end": round(w.end, 2),
                    "word": w.word.strip(),
                    "probability": round(w.probability, 3),
                }
                for w in seg.words
            ]
            total_words += len(seg.words)
        segments.append(seg_data)
        full_text_parts.append(seg.text.strip())

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "file": str(audio_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(info.duration, 2),
        "text": " ".join(full_text_parts),
        "segments": segments,
        "stats": {
            "segment_count": len(segments),
            "word_count": total_words if word_timestamps else None,
            "transcription_ms": elapsed_ms,
        },
    }


def main():
    parser = argparse.ArgumentParser(
        prog="transcribe",
        description="Transcribe audio files using faster-whisper (local, offline).",
    )
    parser.add_argument("files", nargs="+", type=Path, help="Audio file(s) to transcribe")
    parser.add_argument(
        "--model",
        default="base",
        choices=["tiny", "base", "small", "medium", "large-v3"],
        help="Whisper model size (default: base)",
    )
    parser.add_argument(
        "--compute-type",
        default="int8",
        choices=["int8", "float16", "float32"],
        help="Compute type (default: int8)",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        dest="json_output",
        help="Output full JSON with timestamps",
    )
    parser.add_argument(
        "--words",
        action="store_true",
        help="Include word-level timestamps (implies --json)",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Write output to files in this directory instead of stdout",
    )

    args = parser.parse_args()

    if args.words:
        args.json_output = True

    results = []
    for audio_path in args.files:
        if not audio_path.exists():
            print(f"Error: {audio_path} not found", file=sys.stderr)
            sys.exit(1)

        result = transcribe_file(
            audio_path,
            model_size=args.model,
            compute_type=args.compute_type,
            word_timestamps=args.words,
        )
        results.append(result)

        if args.output_dir:
            args.output_dir.mkdir(parents=True, exist_ok=True)
            out_name = audio_path.stem + (".json" if args.json_output else ".txt")
            out_path = args.output_dir / out_name
            with open(out_path, "w") as f:
                if args.json_output:
                    json.dump(result, f, indent=2, ensure_ascii=False)
                else:
                    f.write(result["text"] + "\n")
            print(f"{audio_path.name} -> {out_path}", file=sys.stderr)
        elif args.json_output:
            # Defer JSON output to print all at once
            pass
        else:
            if len(args.files) > 1:
                print(f"--- {audio_path.name} ---")
            print(result["text"])

    if args.json_output and not args.output_dir:
        if len(results) == 1:
            json.dump(results[0], sys.stdout, indent=2, ensure_ascii=False)
        else:
            json.dump(results, sys.stdout, indent=2, ensure_ascii=False)
        print()


if __name__ == "__main__":
    main()
