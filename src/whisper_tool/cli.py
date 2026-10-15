"""Local audio transcription CLI using faster-whisper."""

import argparse
import json
import sys
import time
from pathlib import Path

from faster_whisper import WhisperModel
from faster_whisper.audio import decode_audio

SAMPLE_RATE = 16_000


def transcribe_file(
    audio_path: Path,
    model_size: str = "base",
    compute_type: str = "int8",
    word_timestamps: bool = False,
    initial_prompt: str | None = None,
    hotwords: str | None = None,
    condition_on_previous_text: bool = True,
    independent_window_seconds: float | None = None,
    independent_window_overlap_seconds: float = 0.0,
) -> dict:
    """Transcribe a single audio file. Returns dict with text, segments, and metadata."""
    model = WhisperModel(model_size, compute_type=compute_type)
    start = time.monotonic()
    segments = []
    full_text_parts = []
    total_words = 0
    info = None

    def append_segments(segments_iter, *, offset: float = 0.0) -> None:
        nonlocal total_words
        for seg in segments_iter:
            seg_data = {
                "start": round(seg.start + offset, 2),
                "end": round(seg.end + offset, 2),
                "text": seg.text.strip(),
            }
            if word_timestamps and seg.words:
                seg_data["words"] = [
                    {
                        "start": round(w.start + offset, 2),
                        "end": round(w.end + offset, 2),
                        "word": w.word.strip(),
                        "probability": round(w.probability, 3),
                    }
                    for w in seg.words
                ]
                total_words += len(seg.words)
            segments.append(seg_data)
            full_text_parts.append(seg.text.strip())

    transcription_options = {
        "word_timestamps": word_timestamps,
        "initial_prompt": initial_prompt,
        "hotwords": hotwords,
        "condition_on_previous_text": condition_on_previous_text,
    }
    if independent_window_seconds is None:
        segments_iter, info = model.transcribe(str(audio_path), **transcription_options)
        append_segments(segments_iter)
        duration_seconds = info.duration
    else:
        audio = decode_audio(str(audio_path), sampling_rate=SAMPLE_RATE)
        window_samples = max(1, round(independent_window_seconds * SAMPLE_RATE))
        overlap_samples = round(independent_window_overlap_seconds * SAMPLE_RATE)
        stride_samples = window_samples - overlap_samples
        if stride_samples <= 0:
            raise ValueError(
                "independent window overlap must be smaller than its window"
            )
        for sample_start in range(0, len(audio), stride_samples):
            segments_iter, window_info = model.transcribe(
                audio[sample_start : sample_start + window_samples],
                **transcription_options,
            )
            if info is None:
                info = window_info
            append_segments(segments_iter, offset=sample_start / SAMPLE_RATE)
        duration_seconds = len(audio) / SAMPLE_RATE
    if info is None:
        raise ValueError("audio contains no independently decodable windows")

    elapsed_ms = int((time.monotonic() - start) * 1000)

    return {
        "file": str(audio_path),
        "language": info.language,
        "language_probability": round(info.language_probability, 3),
        "duration_seconds": round(duration_seconds, 2),
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
    parser.add_argument(
        "files", nargs="+", type=Path, help="Audio file(s) to transcribe"
    )
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
    parser.add_argument(
        "--initial-prompt",
        help="Optional spelling/context prompt supplied to Whisper",
    )
    parser.add_argument(
        "--hotwords",
        help="Optional words or phrases to bias in every decode window",
    )
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
        help="Decode each window without carrying prior decoded text forward",
    )
    parser.add_argument(
        "--independent-window-seconds",
        type=float,
        help="Decode fixed-size audio windows independently",
    )
    parser.add_argument(
        "--independent-window-overlap-seconds",
        type=float,
        default=0.0,
        help="Overlap independently decoded windows by this many seconds",
    )

    args = parser.parse_args()
    if (
        args.independent_window_seconds is not None
        and args.independent_window_seconds <= 0
    ):
        parser.error("--independent-window-seconds must be greater than zero")
    if args.independent_window_overlap_seconds < 0:
        parser.error("--independent-window-overlap-seconds cannot be negative")
    if (
        args.independent_window_seconds is None
        and args.independent_window_overlap_seconds
    ):
        parser.error(
            "--independent-window-overlap-seconds requires independent windows"
        )
    if (
        args.independent_window_seconds is not None
        and args.independent_window_overlap_seconds >= args.independent_window_seconds
    ):
        parser.error(
            "--independent-window-overlap-seconds must be smaller than its window"
        )

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
            initial_prompt=args.initial_prompt,
            hotwords=args.hotwords,
            condition_on_previous_text=args.condition_on_previous_text,
            independent_window_seconds=args.independent_window_seconds,
            independent_window_overlap_seconds=args.independent_window_overlap_seconds,
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
