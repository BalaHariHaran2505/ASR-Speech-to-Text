# ASR – Speech-to-Text Tool

A simple command-line **Automatic Speech Recognition (ASR)** tool written in Python.
It converts speech in an audio file (or from your microphone) into text.
It runs **fully offline** after the first model download and needs **no API keys**.

## Features
- Transcribe audio files (`.wav`, `.mp3`, `.m4a`, `.flac`, `.ogg`)
- Record from the microphone and transcribe (`--mic`)
- Automatic language detection
- Save transcript to a `.txt` file
- Choose model size (`tiny`, `base`, `small`) to trade speed for accuracy
- Unit tests with `pytest`

## Tools & Technologies
| Tool | Purpose |
|------|---------|
| Python 3.9+ | Programming language |
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) | Speech recognition (open-source Whisper model, CTranslate2) |
| sounddevice | Microphone recording (optional) |
| argparse, wave | Command-line interface and WAV handling (standard library) |
| pytest | Testing |
| Git & GitHub | Version control and hosting |

## Project Structure
```
ASR-Speech-to-Text/
├── asr.py            # Main ASR tool (CLI + functions)
├── test_asr.py       # Unit tests
├── requirements.txt  # Dependencies
├── README.md
└── .gitignore
```

## Setup
```bash
git clone https://github.com/<your-username>/ASR-Speech-to-Text.git
cd ASR-Speech-to-Text

python -m venv venv
# Windows: venv\Scripts\activate
# Linux/Mac: source venv/bin/activate

pip install -r requirements.txt
```
> The first run downloads the Whisper model (~75 MB for `tiny`). After that it works offline.

## Usage
```bash
# Transcribe a file
python asr.py sample.wav

# Better accuracy + save output
python asr.py sample.mp3 --model base --output transcript.txt

# Record 5 seconds from the microphone
python asr.py --mic --seconds 5
```

Example output:
```
Transcribing...

Detected language: en
Transcript: Hello, this is a test of the speech recognition tool.
Saved to transcript.txt
```

## Run Tests
```bash
pytest -v
```
Tests mock the model, so they run instantly without downloading anything.

## How It Works
1. The audio path is validated (exists, supported format).
2. The Whisper model is loaded on CPU (int8 quantized for speed).
3. Audio is split into segments and decoded into text.
4. Segments are joined, printed, and optionally saved.

## Limitations
- Accuracy depends on audio quality and model size.
- CPU-only, so long files with larger models are slower.

## Author
Hari – B.Tech CSE, SRM Institute of Science and Technology, Trichy
