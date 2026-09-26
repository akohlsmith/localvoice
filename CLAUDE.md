# CLAUDE.md — Project Context for AI Agents

(Also applies if you're reading this as AGENTS.md or similar — same
content, different convention. Copy or symlink as needed for your tool.)

## What this is

A hands-free, two-way voice interface for a local LLM. You talk, it
transcribes, sends the text to a local model server, and speaks the
reply back — no keyboard, no cloud services, and no LM Studio Bionic
(Bionic is a separate vertically-integrated agent app; this project
uses LM Studio's core app purely as an OpenAI-compatible model server).

## Target environment

- MacBook Pro, Apple Silicon, 64GB unified memory.
- LM Studio running headless (`lms server start`), serving a GGUF chat
  model over `http://localhost:1234/v1`. GGUF is a deliberate
  consistency choice here, not a hard requirement — MLX is a valid,
  arguably more native-optimized alternative on this hardware if that
  ever gets revisited.
- A `whisper.cpp` `server` binary running separately, holding a ggml
  model resident in memory, reachable at `stt.whisper_server_url` in
  `config.yaml`.
- Piper installed and reachable via `tts.piper_binary` in `config.yaml`,
  with a voice model such as `en_US-lessac-medium`.

## Architecture

```
mic -> Silero VAD (turn detection) -> whisper.cpp (STT) ->
LM Studio chat completion -> Piper (TTS) -> speaker -> back to listening
```

## Where things live

- `config.yaml` — every endpoint, model name, and threshold. Nothing
  pipeline-related should be hardcoded elsewhere.
- `audio/` — mic capture, VAD/turn-detection, playback, wake-word gate,
  echo cancellation, noise reduction.
- `stt/` — whisper.cpp HTTP client.
- `tts/` — Piper CLI wrapper (v1 voice; Orpheus is the planned upgrade).
- `llm/` — OpenAI-compatible client against LM Studio, holds
  conversation history and runs the tool-call loop.
- `tools/` — tool schemas + dispatch (`registry.py`) and implementations
  (`browser.py`: `web_search`/`web_fetch` via an isolated Brave
  profile). Off by default in `config.yaml`.
- `scripts/` — one-off admin scripts, not part of the runtime loop
  (e.g. `prime_browser_profile.py`).
- `main.py` — the orchestration loop; deliberately the smallest file,
  since everything else is a call into an external tool or server.

## Conventions

- Commit in small, focused commits — one feature or fix per commit,
  not an omnibus commit bundling several. Commit as work lands rather
  than saving it all up for one commit at the end of a session; this
  also keeps each commit's own doc updates (PLAN.md/README.md status
  lines) paired with the change that motivated them, instead of a
  separate catch-up doc commit later.
- Keep whisper.cpp and Piper as external processes, not embedded
  libraries — keeps each stage independently swappable and debuggable.
- One responsibility per module; don't grow `main.py` into a place
  where pipeline logic lives — new capability should go in its own
  module under `audio/`, `stt/`, `tts/`, or `llm/`.
- Config changes go in `config.yaml`, not as new hardcoded constants.

## Current status — see PLAN.md for the authoritative, checkable list

Phases 1–6 are implemented: a working end-to-end loop with a simple
voice. Barge-in (phase 7), a wake word (phase 8, off by default), tool
use (phase 9: web_search/web_fetch via a real, isolated browser), and
noise reduction (phase 10, off by default, for highway-speed vehicle
use) are also done. Not yet done: sentence-level streaming latency
tuning, the Orpheus TTS upgrade, and shell/exec tool access
(deliberately deferred — no code for it exists). Read PLAN.md before
starting new work — it also lists known gaps (no conversation
persistence, no reconnect handling for whisper.cpp/Piper being down).
