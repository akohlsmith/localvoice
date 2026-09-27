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

If `tools.search.provider` is `"firecrawl"` (see `config.yaml`'s
`tools.search` comments for the other option, `"duckduckgo"`, which
needs no key), create a file named `dot-env` in the project root
(gitignored — never put a real API key in `config.yaml`, which is
tracked in git) with:

```
FIRECRAWL_API_KEY=<your key from firecrawl.dev>
```

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

Phases 1–9 are implemented: barge-in, wake word (off by default), and
tool use (web search/fetch via an isolated Brave profile — web_search
can use either DuckDuckGo or Firecrawl's search API; web_fetch is
always the local browser). whisper.cpp/Piper/LM Studio going down
mid-session now triggers a reconnect-and-wait loop instead of crashing
(see `reconnect` in `config.yaml`). A phone can also stand in for this
machine's own mic/speaker (`phone_ui` in `config.yaml`, off by default)
-- see "Phone UI" below. Not yet done: sentence-level streaming to cut
perceived latency, the Orpheus TTS upgrade, and shell/exec tool access.
See `PLAN.md`.

## Phone UI

Set `phone_ui.enabled: true` in `config.yaml` and run `python main.py`
as usual -- it prints a `https://<lan-ip>:8443/?token=...` URL to open
on a phone on the same network. A self-signed cert is generated on
first run (into `phone_ui.cert_dir`); the phone will show a "not
private" warning the first time -- this is unavoidable for a
self-signed cert and only needs accepting once. iOS requires HTTPS for
microphone access (Safari and Chrome on iOS are both WebKit, so this
applies to either), which is why this can't just be plain HTTP.

Tap **Enable** to start streaming the phone's mic (requests microphone
permission the first time); tap **Stop** to interrupt whatever the
assistant is currently saying, the manual equivalent of the local
barge-in.
