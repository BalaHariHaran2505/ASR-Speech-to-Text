"""Tests for asr.py. The Whisper model is mocked so tests run fast and offline."""
import struct
import wave
from types import SimpleNamespace

import pytest

import asr


def make_wav(path, seconds=1, rate=16000):
    with wave.open(str(path), "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(rate)
        wf.writeframes(struct.pack("<h", 0) * rate * seconds)
    return str(path)


class FakeModel:
    def transcribe(self, path, beam_size=5):
        segs = [SimpleNamespace(text=" hello"), SimpleNamespace(text=" world ")]
        return iter(segs), SimpleNamespace(language="en")


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        asr.validate_audio_path("nope.wav")


def test_unsupported_format_raises(tmp_path):
    f = tmp_path / "notes.txt"
    f.write_text("hi")
    with pytest.raises(ValueError):
        asr.validate_audio_path(str(f))


def test_valid_wav_passes(tmp_path):
    wav = make_wav(tmp_path / "a.wav")
    assert asr.validate_audio_path(wav) == wav


def test_invalid_model_size():
    with pytest.raises(ValueError):
        asr.load_model("gigantic")


def test_transcribe_joins_segments(tmp_path):
    wav = make_wav(tmp_path / "a.wav")
    text, lang = asr.transcribe_file(wav, model=FakeModel())
    assert text == "hello world"
    assert lang == "en"


def test_save_transcript(tmp_path):
    out = tmp_path / "out.txt"
    asr.save_transcript("hello world", str(out))
    assert out.read_text(encoding="utf-8").strip() == "hello world"


def test_main_missing_file_returns_error():
    assert asr.main(["missing.wav"]) == 1


def test_main_end_to_end(tmp_path, monkeypatch, capsys):
    wav = make_wav(tmp_path / "a.wav")
    out = tmp_path / "t.txt"
    monkeypatch.setattr(asr, "load_model", lambda size: FakeModel())
    assert asr.main([wav, "--output", str(out)]) == 0
    assert "hello world" in capsys.readouterr().out
    assert out.exists()
