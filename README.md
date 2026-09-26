# LocalVoice

A hands-free, two-way voice interface for a local LLM. Talk to it, it
transcribes, sends the text to a local model server, and speaks the
reply back — no keyboard, no cloud services, and no LM Studio Bionic
(this uses LM Studio's core app purely as a model server).

Target machine: MacBook Pro, Apple Silicon, 64GB unified memory.

## Architecture

```
mic -> Silero VAD (turn detection) -> whisper.cpp (STT) ->
LM Studio chat completion -> Piper (TTS) -> speaker -> back to listening
```

See `CLAUDE.md` for the fuller project context and `PLAN.md` for the
phased build-out and current status.

## Prerequisites

```bash
brew install python@3.12 portaudio ffmpeg whisper-cpp
```

Also install:
- **LM Studio** (the core app, from lmstudio.ai) — this project talks to
  it purely as an OpenAI-compatible server, driven via its `lms` CLI.
  No GUI interaction is required once a model is downloaded.
- **Piper** — the v1 TTS voice. Grab a binary/build and a voice model
  such as `en_US-lessac-medium` (both a `.onnx` file and its matching
  `.onnx.json` config).

## Setup

```bash
git clone <this repo> ~/Projects/localvoice   # if you haven't already
cd ~/Projects/localvoice
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Edit `config.yaml`:
- `llm.model` — the model identifier LM Studio reports once loaded
  (see `lms ps` below).
- `tts.piper_binary` / `tts.piper_model` / `tts.piper_model_config` —
  paths to your Piper install and voice files, if not on PATH.

## Running it

**1. Start LM Studio headlessly:**

```bash
lms get <your-chosen-model>       # one-time download, GGUF quant
lms load <your-chosen-model> -y   # loads with Metal acceleration
lms server start                  # exposes http://localhost:1234/v1
lms ps                            # confirm it's loaded; copy the model id into config.yaml
```

**2. Start whisper.cpp's server**, holding a ggml model resident in
memory (small.en or base.en is a good speed/accuracy trade-off for a
live conversation):

```bash
whisper-server -m models/ggml-small.en.bin --host 127.0.0.1 --port 8081
```

**3. Run the voice loop:**

```bash
source .venv/bin/activate
python main.py
```

Speak naturally — a short pause is what tells it you're done talking
(no button to hold).

## Status

Phases 1–7 (barge-in), 8's wake word (off by default), and 9 (tool
use — web search/fetch via an isolated Brave profile, off by default)
are implemented. Not yet done: sentence-level streaming to cut
perceived latency, the Orpheus TTS upgrade, and shell/exec tool access.
See `PLAN.md`.
