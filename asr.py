"""ASR (Automatic Speech Recognition) tool - converts speech to text.

Uses faster-whisper (open-source Whisper model). Runs locally, no API key needed.

Usage:
    python asr.py audio.wav
    python asr.py audio.mp3 --model base --output transcript.txt
    python asr.py --mic --seconds 5
"""
import argparse
import os
import sys
import tempfile
import wave

SUPPORTED_FORMATS = (".wav", ".mp3", ".m4a", ".flac", ".ogg")
MODEL_SIZES = ("tiny", "base", "small")


def validate_audio_path(path):
    """Check the file exists and has a supported extension."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")
    ext = os.path.splitext(path)[1].lower()
    if ext not in SUPPORTED_FORMATS:
        raise ValueError(
            f"Unsupported format '{ext}'. Supported: {', '.join(SUPPORTED_FORMATS)}"
        )
    return path


def load_model(model_size="tiny"):
    """Load the Whisper model (downloaded once on first run, then cached)."""
    if model_size not in MODEL_SIZES:
        raise ValueError(f"Model must be one of {MODEL_SIZES}")
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device="cpu", compute_type="int8")


def transcribe_file(path, model_size="tiny", model=None):
    """Transcribe an audio file. Returns (full_text, detected_language)."""
    validate_audio_path(path)
    model = model or load_model(model_size)
    segments, info = model.transcribe(path, beam_size=5)
    text = " ".join(seg.text.strip() for seg in segments).strip()
    return text, info.language


def record_from_mic(seconds=5, sample_rate=16000):
    """Record from the microphone and return path to a temporary WAV file."""
    import sounddevice as sd

    print(f"Recording for {seconds} seconds... speak now.")
    audio = sd.rec(int(seconds * sample_rate), samplerate=sample_rate,
                   channels=1, dtype="int16")
    sd.wait()
    tmp = tempfile.NamedTemporaryFile(suffix=".wav", delete=False)
    with wave.open(tmp.name, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(audio.tobytes())
    return tmp.name


def save_transcript(text, output_path):
    """Write the transcript to a text file."""
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(text + "\n")
    return output_path


def main(argv=None):
    parser = argparse.ArgumentParser(description="Speech-to-Text (ASR) tool")
    parser.add_argument("audio", nargs="?", help="Path to audio file")
    parser.add_argument("--mic", action="store_true", help="Record from microphone")
    parser.add_argument("--seconds", type=int, default=5, help="Mic recording length")
    parser.add_argument("--model", default="tiny", choices=MODEL_SIZES,
                        help="Model size (bigger = more accurate, slower)")
    parser.add_argument("--output", help="Save transcript to this .txt file")
    args = parser.parse_args(argv)

    if not args.mic and not args.audio:
        parser.error("Provide an audio file or use --mic")

    try:
        path = record_from_mic(args.seconds) if args.mic else args.audio
        print("Transcribing...")
        text, lang = transcribe_file(path, args.model)
    except (FileNotFoundError, ValueError) as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    print(f"\nDetected language: {lang}")
    print(f"Transcript: {text}")
    if args.output:
        save_transcript(text, args.output)
        print(f"Saved to {args.output}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
