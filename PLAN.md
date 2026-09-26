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
      arrives rather than waiting for the full reply.
  - [x] Barge-in: `main.speak_with_barge_in` plays the reply via
        `audio.playback.play_wav_bytes_async`, scores each incoming mic
        chunk with the VAD while it plays, and calls `playback.stop()`
        after 2 consecutive speech frames — cutting the assistant off
        instead of waiting for the full reply to finish.
  - [ ] Sentence-level streaming into Piper is still open.
- [ ] **8. Upgrade the voice, optionally add a wake word** — swap Piper
      for Orpheus-3B (GGUF, loaded as a second model in LM Studio
      alongside the chat model — 64GB holds both comfortably), which
      needs an added SNAC decode step to turn its output tokens into
      audio.
  - [x] Wake word: `audio/wake_word.py`'s `WakeWordGate` requires a
        configured stretch of silence (`wake_word.silence_before_ms`)
        before it will even consider a candidate utterance, then
        transcribes just that candidate with a decoding hint
        (`stt.transcribe(..., prompt=...)`) and checks it against
        `wake_word.phrase` ("gemma" by default) before switching
        `main.py` into normal command capture. No separate wake-word
        model/training (openWakeWord) needed — reuses TurnDetector +
        whisper.cpp. Off by default (`wake_word.enabled: false`).
        Verified end-to-end against real Silero VAD + real whisper.cpp:
        correct wake phrase after enough silence triggers; the same
        phrase spoken too soon after other speech is discarded before
        even reaching STT; an unrelated sentence reaches STT but
        correctly fails the phrase match.
- [x] **9. Tool use** — `llm/chat_client.py`'s `ask()` now runs a real
      tool-call loop (OpenAI-style `tools`/`tool_calls`, capped at
      `MAX_TOOL_ITERATIONS`), confirmed working against the loaded
      Gemma model. `tools/registry.py` holds the schema + dispatch;
      `tools/browser.py` implements `web_search`/`web_fetch` by driving
      a real, isolated Brave profile (`tools.browser.profile_dir` in
      `config.yaml`, separate from the everyday Brave profile/cookies —
      `scripts/prime_browser_profile.py` selectively copies specific
      site cookies in if a tool call needs a logged-in session). Off by
      default (`tools.browser.enabled: false`) — flip it on to
      use it. Note: true headless Chromium gets served a CAPTCHA by
      search.brave.com specifically; `tools/browser.py` works around
      this by running headed-but-off-screen instead (see its
      docstring). Shell/exec tool access was deliberately deferred —
      no code for it exists yet.

## Known gaps / things to watch

- No persistence of conversation history across restarts.
- No handling yet for whisper.cpp or Piper being unreachable/not
  started — the loop will just raise on the first request.
- Barge-in currently cuts off the whole reply, not just the remaining
  audio — the full reply text is already in `chat.history` even if the
  user only heard part of it spoken. Fine for now; revisit if it
  causes the model to reference lines the user never heard.
- No echo cancellation: with speakers and mic on the same machine,
  `speak_with_barge_in` was self-interrupting almost immediately (~64ms)
  because the mic hears the assistant's own voice and Silero correctly
  scores it as speech. `barge_in.prob_threshold`/`consecutive_frames`
  in `config.yaml` raise the bar a lot (0.9 confidence, ~512ms
  sustained) as a stopgap — real interruptions still work, but loud
  sustained echo could still eventually cross that bar. Real fixes:
  headphones (mic physically can't hear the speakers), or actual echo
  cancellation.
