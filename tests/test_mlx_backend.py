"""Hermetic contracts for the optional MLX Whisper backend."""

from pathlib import Path
from types import SimpleNamespace

import pytest

from whisper_tool import cli, mlx_backend


def _payload(*, probability: float = 0.9, end: float = 0.5) -> dict:
    return {
        "language": "en",
        "segments": [
            {
                "start": 0.25,
                "end": end,
                "text": " Huécar ",
                "words": [
                    {
                        "start": 0.25,
                        "end": end,
                        "word": " Huécar",
                        "probability": probability,
                    }
                ],
            }
        ],
    }


def _fake_modules(monkeypatch, payload=None, seconds: int = 2):
    calls = []
    downloads = []
    audio = list(range(mlx_backend.SAMPLE_RATE * seconds))

    def transcribe(window, **options):
        calls.append((window, options))
        return _payload() if payload is None else payload

    def snapshot_download(**options):
        downloads.append(options)
        return "/immutable/mlx-large-v3"

    modules = (
        SimpleNamespace(transcribe=transcribe),
        SimpleNamespace(load_audio=lambda _path: audio),
        SimpleNamespace(snapshot_download=snapshot_download),
    )
    monkeypatch.setattr(mlx_backend, "_load_modules", lambda: modules)
    return calls, downloads


def test_auto_selects_mlx_only_for_eligible_apple_large_v3():
    assert (
        cli.select_backend(
            "auto",
            model_size="large-v3",
            hotwords=None,
            system="Darwin",
            machine="arm64",
            mlx_available=True,
        )
        == "mlx"
    )
    ineligible = [
        {"model_size": "base"},
        {"hotwords": "Huécar"},
        {"system": "Linux"},
        {"machine": "x86_64"},
        {"mlx_available": False},
    ]
    defaults = {
        "model_size": "large-v3",
        "hotwords": None,
        "system": "Darwin",
        "machine": "arm64",
        "mlx_available": True,
    }
    for overrides in ineligible:
        assert cli.select_backend("auto", **(defaults | overrides)) == "faster-whisper"


def test_explicit_backend_selection_does_not_probe_platform():
    assert cli.select_backend(
        "faster-whisper", model_size="large-v3", hotwords=None
    ) == ("faster-whisper")
    assert cli.select_backend("mlx", model_size="large-v3", hotwords=None) == "mlx"


def test_mlx_adapts_words_and_independent_window_offsets(monkeypatch, capsys):
    calls, downloads = _fake_modules(monkeypatch)

    result = mlx_backend.transcribe_mlx(
        Path("audio.mp3"),
        model_size="large-v3",
        word_timestamps=True,
        initial_prompt="Huécar",
        hotwords=None,
        condition_on_previous_text=False,
        independent_window_seconds=1,
        independent_window_overlap_seconds=0,
    )

    assert downloads == [
        {
            "repo_id": mlx_backend.MLX_LARGE_V3_REPOSITORY,
            "revision": mlx_backend.MLX_LARGE_V3_REVISION,
            "local_files_only": True,
        }
    ]
    assert len(calls) == 2
    assert all(len(window) == mlx_backend.SAMPLE_RATE for window, _ in calls)
    assert all(options["initial_prompt"] == "Huécar" for _, options in calls)
    assert all(options["condition_on_previous_text"] is False for _, options in calls)
    assert all(options["word_timestamps"] is True for _, options in calls)
    assert all(options["verbose"] is None for _, options in calls)
    assert all(options["fp16"] is True for _, options in calls)
    assert [segment["start"] for segment in result["segments"]] == [0.25, 1.25]
    assert [segment["words"][0]["start"] for segment in result["segments"]] == [
        0.25,
        1.25,
    ]
    assert result["segments"][0]["words"][0]["probability"] == 0.9
    assert result["word_count"] == 2
    assert result["language_probability"] is None
    assert result["transcription_binding"] == mlx_backend.MLX_TRANSCRIPTION_BINDING
    assert capsys.readouterr().err == ""


def test_mlx_transcribe_file_preserves_result_shape(monkeypatch):
    fake_result = {
        "language": "en",
        "language_probability": None,
        "duration_seconds": 2.0,
        "text": "Huécar",
        "segments": [{"start": 0.0, "end": 1.0, "text": "Huécar"}],
        "word_count": None,
        "transcription_binding": mlx_backend.MLX_TRANSCRIPTION_BINDING,
    }
    monkeypatch.setattr(
        mlx_backend, "transcribe_mlx", lambda *_args, **_kwargs: fake_result
    )

    result = cli.transcribe_file(
        Path("audio.mp3"),
        model_size="large-v3",
        backend="mlx",
    )

    assert set(result) == {
        "file",
        "language",
        "language_probability",
        "duration_seconds",
        "text",
        "segments",
        "transcription_binding",
        "stats",
    }
    assert set(result["stats"]) == {
        "segment_count",
        "word_count",
        "transcription_ms",
    }


@pytest.mark.parametrize(
    ("payload", "message"),
    [
        (_payload(probability=float("nan")), "non-finite word probability"),
        (_payload(probability=1.01), "out-of-bounds word probability"),
        (_payload(end=3.0), "out-of-bounds segment end"),
    ],
)
def test_mlx_rejects_invalid_evidence(monkeypatch, payload, message):
    _fake_modules(monkeypatch, payload=payload, seconds=2)

    with pytest.raises(ValueError, match=message):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="large-v3",
            word_timestamps=True,
            initial_prompt=None,
            hotwords=None,
            condition_on_previous_text=True,
            independent_window_seconds=None,
            independent_window_overlap_seconds=0,
        )


def test_mlx_rejects_missing_word_probability(monkeypatch):
    payload = _payload()
    del payload["segments"][0]["words"][0]["probability"]
    _fake_modules(monkeypatch, payload=payload)

    with pytest.raises(ValueError, match="non-numeric word probability"):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="large-v3",
            word_timestamps=True,
            initial_prompt=None,
            hotwords=None,
            condition_on_previous_text=True,
            independent_window_seconds=None,
            independent_window_overlap_seconds=0,
        )


def test_mlx_rejects_out_of_bounds_word_timestamp(monkeypatch):
    payload = _payload()
    payload["segments"][0]["words"][0]["end"] = 3.0
    _fake_modules(monkeypatch, payload=payload)

    with pytest.raises(ValueError, match="out-of-bounds word end"):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="large-v3",
            word_timestamps=True,
            initial_prompt=None,
            hotwords=None,
            condition_on_previous_text=True,
            independent_window_seconds=None,
            independent_window_overlap_seconds=0,
        )


@pytest.mark.parametrize(
    ("window", "overlap", "message"),
    [
        (float("nan"), 0, "duration must be finite and positive"),
        (1, float("inf"), "overlap must be finite and nonnegative"),
        (None, 0.5, "overlap requires a window duration"),
        (1, 1, "overlap must be smaller than its window"),
    ],
)
def test_mlx_rejects_invalid_window_bounds_before_loading(
    monkeypatch, window, overlap, message
):
    monkeypatch.setattr(
        mlx_backend,
        "_load_modules",
        lambda: pytest.fail("invalid inputs must fail before importing MLX"),
    )
    with pytest.raises(ValueError, match=message):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="large-v3",
            word_timestamps=False,
            initial_prompt=None,
            hotwords=None,
            condition_on_previous_text=True,
            independent_window_seconds=window,
            independent_window_overlap_seconds=overlap,
        )


def test_mlx_rejects_unbound_model_and_unsupported_hotwords(monkeypatch):
    monkeypatch.setattr(
        mlx_backend,
        "_load_modules",
        lambda: pytest.fail("unsupported inputs must fail before importing MLX"),
    )
    with pytest.raises(ValueError, match="only the pinned large-v3"):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="base",
            word_timestamps=False,
            initial_prompt=None,
            hotwords=None,
            condition_on_previous_text=True,
            independent_window_seconds=None,
            independent_window_overlap_seconds=0,
        )
    with pytest.raises(ValueError, match="does not support hotwords"):
        mlx_backend.transcribe_mlx(
            Path("audio.mp3"),
            model_size="large-v3",
            word_timestamps=False,
            initial_prompt=None,
            hotwords="Huécar",
            condition_on_previous_text=True,
            independent_window_seconds=None,
            independent_window_overlap_seconds=0,
        )


def test_mlx_binding_is_exact_and_runtime_versions_are_enforced(monkeypatch):
    assert mlx_backend.MLX_TRANSCRIPTION_BINDING == {
        "backend": "mlx",
        "precision": "float16",
        "packages": {"mlx": "0.32.4", "mlx-whisper": "0.4.3"},
        "model": {
            "name": "large-v3",
            "repository": "mlx-community/whisper-large-v3-mlx",
            "revision": "49e6aa286ad60c14352c404340ded53710378a11",
        },
    }
    versions = {"mlx-whisper": "0.4.3", "mlx": "0.32.4"}
    monkeypatch.setattr(
        mlx_backend.importlib.metadata, "version", lambda package: versions[package]
    )
    mlx_backend._validate_runtime_versions()

    versions["mlx"] = "0.32.5"
    with pytest.raises(RuntimeError, match=r"requires mlx==0\.32\.4; found 0\.32\.5"):
        mlx_backend._validate_runtime_versions()


def test_mlx_rejects_incompatible_compute_type_before_loading(monkeypatch):
    monkeypatch.setattr(
        mlx_backend,
        "transcribe_mlx",
        lambda *_args, **_kwargs: pytest.fail("compute type must fail first"),
    )
    with pytest.raises(ValueError, match="requires --compute-type float16"):
        cli.transcribe_file(
            Path("audio.mp3"),
            model_size="large-v3",
            backend="mlx",
            compute_type="int8",
        )


def test_mlx_accepts_silence_and_upstream_word_segment_skew(monkeypatch):
    silent = {"language": "en", "segments": []}
    _fake_modules(monkeypatch, payload=silent)
    result = mlx_backend.transcribe_mlx(
        Path("audio.mp3"),
        model_size="large-v3",
        word_timestamps=True,
        initial_prompt=None,
        hotwords=None,
        condition_on_previous_text=True,
        independent_window_seconds=None,
        independent_window_overlap_seconds=0,
    )
    assert result["text"] == ""
    assert result["segments"] == []

    outside = _payload()
    outside["segments"][0]["words"][0]["start"] = 0.0
    _fake_modules(monkeypatch, payload=outside)
    result = mlx_backend.transcribe_mlx(
        Path("audio.mp3"),
        model_size="large-v3",
        word_timestamps=True,
        initial_prompt=None,
        hotwords=None,
        condition_on_previous_text=True,
        independent_window_seconds=None,
        independent_window_overlap_seconds=0,
    )
    assert result["segments"][0]["words"][0]["start"] == 0.0
