"""Validated MLX Whisper adapter for Apple Silicon transcription."""

from __future__ import annotations

import importlib
import importlib.metadata
import math
from copy import deepcopy
from numbers import Real
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
TIMESTAMP_TOLERANCE_SECONDS = 0.05
MLX_LARGE_V3_REPOSITORY = "mlx-community/whisper-large-v3-mlx"
MLX_LARGE_V3_REVISION = "49e6aa286ad60c14352c404340ded53710378a11"
MLX_WHISPER_VERSION = "0.4.3"
MLX_VERSION = "0.32.4"
MLX_TRANSCRIPTION_BINDING = {
    "backend": "mlx",
    "precision": "float16",
    "packages": {
        "mlx": MLX_VERSION,
        "mlx-whisper": MLX_WHISPER_VERSION,
    },
    "model": {
        "name": "large-v3",
        "repository": MLX_LARGE_V3_REPOSITORY,
        "revision": MLX_LARGE_V3_REVISION,
    },
}


def _finite_number(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, Real):
        raise ValueError(f"MLX Whisper returned non-numeric {name}")
    number = float(value)
    if not math.isfinite(number):
        raise ValueError(f"MLX Whisper returned non-finite {name}")
    return number


def _bounded_timestamp(value: object, name: str, duration: float) -> float:
    timestamp = _finite_number(value, name)
    if timestamp < 0 or timestamp > duration + TIMESTAMP_TOLERANCE_SECONDS:
        raise ValueError(f"MLX Whisper returned out-of-bounds {name}")
    return timestamp


def _probability(value: object) -> float:
    probability = _finite_number(value, "word probability")
    if not 0 <= probability <= 1:
        raise ValueError("MLX Whisper returned out-of-bounds word probability")
    return probability


def _adapt_words(
    words: object,
    duration: float,
    offset: float,
) -> list[dict]:
    if not isinstance(words, list):
        raise ValueError("MLX Whisper omitted word timestamps")
    adapted = []
    previous_end: float | None = None
    for word in words:
        if not isinstance(word, dict) or not isinstance(word.get("word"), str):
            raise ValueError("MLX Whisper returned malformed word data")
        start = _bounded_timestamp(word.get("start"), "word start", duration)
        end = _bounded_timestamp(word.get("end"), "word end", duration)
        if end < start:
            raise ValueError("MLX Whisper returned reversed word timestamps")
        if (
            previous_end is not None
            and start < previous_end - TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise ValueError("MLX Whisper returned overlapping word timestamps")
        previous_end = end
        adapted.append(
            {
                "start": round(start + offset, 2),
                "end": round(end + offset, 2),
                "word": word["word"].strip(),
                "probability": round(_probability(word.get("probability")), 3),
            }
        )
    return adapted


def _adapt_segments(
    payload: object,
    *,
    duration: float,
    offset: float,
    word_timestamps: bool,
) -> list[dict]:
    if not isinstance(payload, dict) or not isinstance(payload.get("segments"), list):
        raise ValueError("MLX Whisper returned malformed transcription data")
    adapted = []
    previous_end: float | None = None
    for segment in payload["segments"]:
        if not isinstance(segment, dict) or not isinstance(segment.get("text"), str):
            raise ValueError("MLX Whisper returned malformed segment data")
        start = _bounded_timestamp(segment.get("start"), "segment start", duration)
        end = _bounded_timestamp(segment.get("end"), "segment end", duration)
        if end < start:
            raise ValueError("MLX Whisper returned reversed segment timestamps")
        if (
            previous_end is not None
            and start < previous_end - TIMESTAMP_TOLERANCE_SECONDS
        ):
            raise ValueError("MLX Whisper returned overlapping segment timestamps")
        previous_end = end
        result = {
            "start": round(start + offset, 2),
            "end": round(end + offset, 2),
            "text": segment["text"].strip(),
        }
        if word_timestamps:
            result["words"] = _adapt_words(segment.get("words"), duration, offset)
        adapted.append(result)
    return adapted


def _validate_runtime_versions() -> None:
    expected = {"mlx-whisper": MLX_WHISPER_VERSION, "mlx": MLX_VERSION}
    for distribution, expected_version in expected.items():
        try:
            installed_version = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError as exc:
            raise RuntimeError(
                f"MLX backend requires {distribution}=={expected_version}"
            ) from exc
        if installed_version != expected_version:
            raise RuntimeError(
                f"MLX backend requires {distribution}=={expected_version}; "
                f"found {installed_version}"
            )


def _load_modules() -> tuple[Any, Any, Any]:
    _validate_runtime_versions()
    try:
        mlx_whisper = importlib.import_module("mlx_whisper")
        mlx_audio = importlib.import_module("mlx_whisper.audio")
        huggingface_hub = importlib.import_module("huggingface_hub")
    except ImportError as exc:
        raise RuntimeError(
            "MLX backend requires the pinned 'mlx' optional dependency"
        ) from exc
    return mlx_whisper, mlx_audio, huggingface_hub


def transcribe_mlx(
    audio_path: Path,
    *,
    model_size: str,
    word_timestamps: bool,
    initial_prompt: str | None,
    hotwords: str | None,
    condition_on_previous_text: bool,
    independent_window_seconds: float | None,
    independent_window_overlap_seconds: float,
) -> dict:
    """Transcribe through MLX while retaining the existing JSON result contract."""
    if model_size != "large-v3":
        raise ValueError("MLX backend supports only the pinned large-v3 model")
    if hotwords is not None:
        raise ValueError("MLX backend does not support hotwords")
    if independent_window_seconds is not None and (
        not math.isfinite(independent_window_seconds) or independent_window_seconds <= 0
    ):
        raise ValueError("independent window duration must be finite and positive")
    if (
        not math.isfinite(independent_window_overlap_seconds)
        or independent_window_overlap_seconds < 0
    ):
        raise ValueError("independent window overlap must be finite and nonnegative")
    if independent_window_seconds is None and independent_window_overlap_seconds != 0:
        raise ValueError("independent window overlap requires a window duration")
    if (
        independent_window_seconds is not None
        and independent_window_overlap_seconds >= independent_window_seconds
    ):
        raise ValueError("independent window overlap must be smaller than its window")

    mlx_whisper, mlx_audio, huggingface_hub = _load_modules()
    model_path = huggingface_hub.snapshot_download(
        repo_id=MLX_LARGE_V3_REPOSITORY,
        revision=MLX_LARGE_V3_REVISION,
        local_files_only=True,
    )
    audio = mlx_audio.load_audio(str(audio_path))
    duration_seconds = len(audio) / SAMPLE_RATE
    if not math.isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("audio contains no independently decodable windows")

    transcription_options = {
        "path_or_hf_repo": model_path,
        "word_timestamps": word_timestamps,
        "initial_prompt": initial_prompt,
        "condition_on_previous_text": condition_on_previous_text,
        "verbose": None,
        "fp16": True,
    }
    windows: list[tuple[object, float, float]] = []
    if independent_window_seconds is None:
        windows.append((audio, 0.0, duration_seconds))
    else:
        window_samples = max(1, round(independent_window_seconds * SAMPLE_RATE))
        overlap_samples = round(independent_window_overlap_seconds * SAMPLE_RATE)
        stride_samples = window_samples - overlap_samples
        if stride_samples <= 0:
            raise ValueError(
                "independent window overlap must be smaller than its window"
            )
        for sample_start in range(0, len(audio), stride_samples):
            window = audio[sample_start : sample_start + window_samples]
            windows.append(
                (
                    window,
                    sample_start / SAMPLE_RATE,
                    len(window) / SAMPLE_RATE,
                )
            )

    segments = []
    full_text_parts: list[str] = []
    language = None
    for window, offset, window_duration in windows:
        payload = mlx_whisper.transcribe(window, **transcription_options)
        if not isinstance(payload, dict):
            raise ValueError("MLX Whisper returned malformed transcription data")
        window_language = payload.get("language")
        if not isinstance(window_language, str) or not window_language:
            raise ValueError("MLX Whisper returned malformed language data")
        if language is None:
            language = window_language
        elif window_language != language:
            raise ValueError("MLX Whisper returned inconsistent window languages")
        adapted = _adapt_segments(
            payload,
            duration=window_duration,
            offset=offset,
            word_timestamps=word_timestamps,
        )
        segments.extend(adapted)
        full_text_parts.extend(segment["text"] for segment in adapted)

    if language is None:
        raise ValueError("audio contains no independently decodable windows")
    return {
        "language": language,
        # mlx-whisper detects a language but does not expose its probability.
        "language_probability": None,
        "duration_seconds": round(duration_seconds, 2),
        "text": " ".join(full_text_parts),
        "segments": segments,
        "word_count": (
            sum(len(segment.get("words", [])) for segment in segments)
            if word_timestamps
            else None
        ),
        "transcription_binding": deepcopy(MLX_TRANSCRIPTION_BINDING),
    }
