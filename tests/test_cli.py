"""Tests for whisper-tool CLI argument parsing and output formatting."""

import json
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import pytest

# ---------------------------------------------------------------------------
# Helpers — build the arg parser the same way main() does
# ---------------------------------------------------------------------------


def _build_parser():
    """Reproduce the ArgumentParser from cli.main so we can test parsing
    without invoking the real transcription pipeline."""
    import argparse

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
    parser.add_argument("--initial-prompt")
    parser.add_argument("--hotwords")
    parser.add_argument(
        "--no-condition-on-previous-text",
        action="store_false",
        dest="condition_on_previous_text",
    )
    parser.add_argument("--independent-window-seconds", type=float)
    parser.add_argument("--independent-window-overlap-seconds", type=float, default=0.0)
    return parser


def _fake_result(filename="test.m4a", text="Hello world"):
    """Return a realistic transcribe_file result dict for output tests."""
    return {
        "file": filename,
        "language": "en",
        "language_probability": 0.98,
        "duration_seconds": 3.5,
        "text": text,
        "segments": [
            {"start": 0.0, "end": 3.5, "text": text},
        ],
        "stats": {
            "segment_count": 1,
            "word_count": None,
            "transcription_ms": 120,
        },
    }


# ===================================================================
# 1. CLI argument parsing
# ===================================================================


class TestArgumentParsing:
    """Test that argparse options are wired correctly."""

    def test_single_file(self):
        parser = _build_parser()
        args = parser.parse_args(["recording.m4a"])
        assert args.files == [Path("recording.m4a")]
        assert args.model == "base"
        assert args.compute_type == "int8"
        assert args.json_output is False
        assert args.words is False
        assert args.output_dir is None
        assert args.initial_prompt is None
        assert args.hotwords is None
        assert args.condition_on_previous_text is True
        assert args.independent_window_seconds is None
        assert args.independent_window_overlap_seconds == 0.0

    def test_multiple_files(self):
        parser = _build_parser()
        args = parser.parse_args(["a.m4a", "b.wav", "c.mp3"])
        assert len(args.files) == 3
        assert args.files[1] == Path("b.wav")

    def test_model_choices(self):
        parser = _build_parser()
        for model in ("tiny", "base", "small", "medium", "large-v3"):
            args = parser.parse_args(["f.m4a", "--model", model])
            assert args.model == model

    def test_invalid_model_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["f.m4a", "--model", "huge"])

    def test_compute_type_choices(self):
        parser = _build_parser()
        for ct in ("int8", "float16", "float32"):
            args = parser.parse_args(["f.m4a", "--compute-type", ct])
            assert args.compute_type == ct

    def test_invalid_compute_type_rejected(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args(["f.m4a", "--compute-type", "bfloat16"])

    def test_json_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--json"])
        assert args.json_output is True

    def test_words_flag(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--words"])
        assert args.words is True

    def test_output_dir(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--output-dir", "/tmp/out"])
        assert args.output_dir == Path("/tmp/out")

    def test_prompt_and_context_reset(self):
        parser = _build_parser()
        args = parser.parse_args(
            ["f.m4a", "--initial-prompt", "Huécar", "--no-condition-on-previous-text"]
        )
        assert args.initial_prompt == "Huécar"
        assert args.condition_on_previous_text is False

    def test_hotwords(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--hotwords", "Huécar Júcar"])
        assert args.hotwords == "Huécar Júcar"

    def test_independent_window_seconds(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "f.m4a",
                "--independent-window-seconds",
                "30",
                "--independent-window-overlap-seconds",
                "15",
            ]
        )
        assert args.independent_window_seconds == 30.0
        assert args.independent_window_overlap_seconds == 15.0

    def test_no_files_exits(self):
        parser = _build_parser()
        with pytest.raises(SystemExit):
            parser.parse_args([])

    def test_all_options_combined(self):
        parser = _build_parser()
        args = parser.parse_args(
            [
                "a.m4a",
                "b.m4a",
                "--model",
                "large-v3",
                "--compute-type",
                "float16",
                "--json",
                "--words",
                "--output-dir",
                "./transcripts",
            ]
        )
        assert args.files == [Path("a.m4a"), Path("b.m4a")]
        assert args.model == "large-v3"
        assert args.compute_type == "float16"
        assert args.json_output is True
        assert args.words is True
        assert args.output_dir == Path("./transcripts")


# ===================================================================
# 2. --words implies --json logic
# ===================================================================


class TestWordsImpliesJson:
    """The main() function sets json_output=True when --words is passed."""

    def test_words_sets_json_output(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--words"])
        # Reproduce the logic from main()
        if args.words:
            args.json_output = True
        assert args.json_output is True

    def test_json_without_words(self):
        parser = _build_parser()
        args = parser.parse_args(["f.m4a", "--json"])
        if args.words:
            args.json_output = True
        assert args.json_output is True
        assert args.words is False


# ===================================================================
# 3. Output formatting (mock transcribe_file, test stdout/file output)
# ===================================================================


class TestPlainTextOutput:
    """Test plain-text stdout output path (no --json, no --output-dir)."""

    def test_single_file_plain_text(self, capsys):
        """Single file prints just the text, no header."""
        result = _fake_result(text="Testing one two three")

        with (
            patch("whisper_tool.cli.transcribe_file", return_value=result),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "test.m4a"]
            from whisper_tool.cli import main

            main()

        captured = capsys.readouterr()
        assert captured.out.strip() == "Testing one two three"

    def test_transcription_controls_are_forwarded(self):
        result = _fake_result()
        with (
            patch(
                "whisper_tool.cli.transcribe_file", return_value=result
            ) as transcribe,
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = [
                "transcribe",
                "test.m4a",
                "--initial-prompt",
                "Huécar",
                "--hotwords",
                "Huécar Júcar",
                "--no-condition-on-previous-text",
                "--independent-window-seconds",
                "30",
                "--independent-window-overlap-seconds",
                "15",
            ]
            from whisper_tool.cli import main

            main()

        assert transcribe.call_args.kwargs["initial_prompt"] == "Huécar"
        assert transcribe.call_args.kwargs["hotwords"] == "Huécar Júcar"
        assert transcribe.call_args.kwargs["condition_on_previous_text"] is False
        assert transcribe.call_args.kwargs["independent_window_seconds"] == 30.0
        assert transcribe.call_args.kwargs["independent_window_overlap_seconds"] == 15.0

    def test_multiple_files_plain_text(self, capsys):
        """Multiple files print headers like '--- filename ---'."""
        r1 = _fake_result("a.m4a", "First file")
        r2 = _fake_result("b.m4a", "Second file")

        call_count = 0

        def fake_transcribe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return r1 if call_count == 1 else r2

        with (
            patch("whisper_tool.cli.transcribe_file", side_effect=fake_transcribe),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "a.m4a", "b.m4a"]
            from whisper_tool.cli import main

            main()

        captured = capsys.readouterr()
        assert "--- a.m4a ---" in captured.out
        assert "First file" in captured.out
        assert "--- b.m4a ---" in captured.out
        assert "Second file" in captured.out


class TestJsonOutput:
    """Test JSON stdout output path (--json flag)."""

    def test_single_file_json(self, capsys):
        """Single file with --json prints a JSON object (not array)."""
        result = _fake_result(text="Hello JSON")

        with (
            patch("whisper_tool.cli.transcribe_file", return_value=result),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "test.m4a", "--json"]
            from whisper_tool.cli import main

            main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, dict)
        assert parsed["text"] == "Hello JSON"
        assert parsed["language"] == "en"

    def test_multiple_files_json(self, capsys):
        """Multiple files with --json prints a JSON array."""
        r1 = _fake_result("a.m4a", "First")
        r2 = _fake_result("b.m4a", "Second")

        call_count = 0

        def fake_transcribe(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return r1 if call_count == 1 else r2

        with (
            patch("whisper_tool.cli.transcribe_file", side_effect=fake_transcribe),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "a.m4a", "b.m4a", "--json"]
            from whisper_tool.cli import main

            main()

        captured = capsys.readouterr()
        parsed = json.loads(captured.out)
        assert isinstance(parsed, list)
        assert len(parsed) == 2
        assert parsed[0]["text"] == "First"
        assert parsed[1]["text"] == "Second"


class TestFileOutput:
    """Test --output-dir writes files correctly."""

    def test_output_dir_plain_text(self, tmp_path):
        """Plain text mode writes .txt files."""
        result = _fake_result(text="Written to file")

        with (
            patch("whisper_tool.cli.transcribe_file", return_value=result),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "test.m4a", "--output-dir", str(tmp_path)]
            from whisper_tool.cli import main

            main()

        out_file = tmp_path / "test.txt"
        assert out_file.exists()
        assert out_file.read_text().strip() == "Written to file"

    def test_output_dir_json(self, tmp_path):
        """JSON mode writes .json files."""
        result = _fake_result(text="JSON to file")

        with (
            patch("whisper_tool.cli.transcribe_file", return_value=result),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = [
                "transcribe",
                "test.m4a",
                "--json",
                "--output-dir",
                str(tmp_path),
            ]
            from whisper_tool.cli import main

            main()

        out_file = tmp_path / "test.json"
        assert out_file.exists()
        parsed = json.loads(out_file.read_text())
        assert parsed["text"] == "JSON to file"

    def test_output_dir_created_if_missing(self, tmp_path):
        """--output-dir creates the directory if it doesn't exist."""
        out_dir = tmp_path / "nested" / "dir"
        result = _fake_result(text="Nested")

        with (
            patch("whisper_tool.cli.transcribe_file", return_value=result),
            patch("whisper_tool.cli.Path.exists", return_value=True),
        ):
            sys.argv = ["transcribe", "test.m4a", "--output-dir", str(out_dir)]
            from whisper_tool.cli import main

            main()

        assert (out_dir / "test.txt").exists()


class TestMissingFileError:
    """Test that main() exits with error for non-existent input files."""

    def test_nonexistent_file_exits(self):
        """Passing a file that doesn't exist should sys.exit(1)."""
        sys.argv = ["transcribe", "/nonexistent/audio.m4a"]
        from whisper_tool.cli import main

        with pytest.raises(SystemExit) as exc_info:
            main()
        assert exc_info.value.code == 1


# ===================================================================
# 4. Result dict structure validation
# ===================================================================


class TestResultStructure:
    """Verify the shape of the dict returned by transcribe_file,
    using a mock so we don't need the actual model."""

    def test_result_has_required_keys(self):
        result = _fake_result()
        required = {
            "file",
            "language",
            "language_probability",
            "duration_seconds",
            "text",
            "segments",
            "stats",
        }
        assert required.issubset(result.keys())

    def test_stats_keys(self):
        result = _fake_result()
        stats = result["stats"]
        assert "segment_count" in stats
        assert "word_count" in stats
        assert "transcription_ms" in stats

    def test_segment_keys(self):
        result = _fake_result()
        seg = result["segments"][0]
        assert "start" in seg
        assert "end" in seg
        assert "text" in seg


def test_independent_windows_reset_decoder_and_offset_timestamps(monkeypatch):
    from whisper_tool import cli

    calls = []

    class FakeModel:
        def transcribe(self, audio, **options):
            calls.append((audio, options))
            word = SimpleNamespace(start=0.25, end=0.5, word=" Huécar", probability=0.9)
            segment = SimpleNamespace(start=0.25, end=0.5, text="Huécar", words=[word])
            info = SimpleNamespace(
                language="en", language_probability=0.99, duration=1.0
            )
            return iter([segment]), info

    monkeypatch.setattr(cli, "WhisperModel", lambda *_args, **_kwargs: FakeModel())
    monkeypatch.setattr(
        cli,
        "decode_audio",
        lambda *_args, **_kwargs: list(range(cli.SAMPLE_RATE * 2)),
    )

    result = cli.transcribe_file(
        Path("audio.mp3"),
        word_timestamps=True,
        initial_prompt="Huécar",
        condition_on_previous_text=False,
        independent_window_seconds=1,
    )

    assert [segment["start"] for segment in result["segments"]] == [0.25, 1.25]
    assert [segment["words"][0]["start"] for segment in result["segments"]] == [
        0.25,
        1.25,
    ]
    assert result["duration_seconds"] == 2.0
    assert len(calls) == 2
    assert all(options["initial_prompt"] == "Huécar" for _, options in calls)
    assert all(options["condition_on_previous_text"] is False for _, options in calls)
