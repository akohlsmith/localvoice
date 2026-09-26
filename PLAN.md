# Build Plan

Tracks the phased plan for this project and what's actually done, so
anyone (human or agent) picking this up later knows where things stand
without re-deriving the whole design.

- [x] **1. Stand up LM Studio as a headless server** — core app only, no
      Bionic, driven via `lms` CLI. GGUF chat model, not MLX (a
      deliberate consistency choice, not a hard technical requirement —
      MLX is a legitimate alternative on Apple Silicon if revisited).
- [x] **2. Scaffold the Python project** — this repo: `audio/`, `stt/`,
      `tts/`, `llm/`, `config.yaml`, `requirements.txt`.
- [x] **3. Get transcription working (whisper.cpp)** — `stt/whisper_client.py`
      posts audio to a whisper.cpp `server` process running separately.
- [x] **4. Get a voice working — Piper (v1)** — `tts/tts_client.py`
      wraps the Piper CLI. Simple, fast, no token-decode step.
- [x] **5. Wire up voice activity detection** — `audio/vad.py`, Silero
      VAD plus a turn-detection state machine (speech-then-silence ==
      end of turn). This is the actual "no keyboard" mechanism.
- [x] **6. Build the orchestration loop** — `main.py` ties capture ->
      VAD -> STT -> LLM -> TTS -> playback -> back to listening, with
      conversation history kept across turns.
- [ ] **7. Tune for conversational latency** — stream the LLM response
      (`ChatClient.ask_streaming` exists but isn't wired into `main.py`
      yet), split it into sentences and send each to Piper as it
      arrives rather than waiting for the full reply. Add barge-in:
      stop playback (`audio.playback.stop()`) if VAD detects speech
      while audio is playing.
- [ ] **8. Upgrade the voice, optionally add a wake word** — swap Piper
      for Orpheus-3B (GGUF, loaded as a second model in LM Studio
      alongside the chat model — 64GB holds both comfortably), which
      needs an added SNAC decode step to turn its output tokens into
      audio. Optionally gate listening behind a keyword (openWakeWord)
      so an always-on mic doesn't transcribe ambient conversation.

## Known gaps / things to watch

- No persistence of conversation history across restarts.
- No handling yet for whisper.cpp or Piper being unreachable/not
  started — the loop will just raise on the first request.
- Playback in `audio/playback.py` blocks (`sd.wait()`); barge-in
  (phase 7) will need this refactored to run on its own thread so VAD
  can interrupt it mid-sentence.
