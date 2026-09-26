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
        Bug found after enabling it live: `WakeWordGate` only guarantees
        silence preceded the *utterance as a whole* — if a longer
        sentence with the phrase buried in it has no pause inside it
        (e.g. "I was thinking Gemma might be a nice name"), the whole
        sentence is one utterance that correctly passes the silence
        gate, and a plain substring match then fired on "Gemma" mid-
        sentence, which itself was never preceded by any silence. Fixed
        by requiring the phrase at the *start* of the transcript (mod
        leading whitespace/punctuation) with a word boundary after it,
        not just present anywhere — ties the check to the phrase itself
        rather than to whatever utterance it happened to be part of.
        Re-verified against real audio: a wake phrase buried mid-sentence
        no longer matches; an isolated phrase and phrase-plus-query both
        still do.
        Shortcut back to wake-word listening (easy to wake it and then
        forget what you meant to ask): `main.py` tracks how long it's
        been pure silence via `turn_detector.in_speech` transitions
        (reset the instant real speech starts) since the wake word;
        past `wake_word.command_timeout_ms` (3000ms default) with
        nothing said, it gives up and goes back to waiting for the
        phrase. Separately, `is_cancel_phrase()` checks the captured
        command against `wake_word.cancel_phrases` ("cancel",
        "nevermind", etc.) as an *exact* match after normalizing — not
        substring, so "cancel my 3pm meeting" is still a real command —
        and aborts the same way without ever reaching the LLM. Verified
        the timeout state machine against real Silero VAD: exactly
        93 chunks (~3000ms) of pure silence trips it; 2s of silence
        followed by real speech correctly resets the counter and still
        captures the utterance normally.
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
- [x] **10. Noise reduction** — for highway-speed vehicle use.
      `audio/noise_reduce.py`'s `NoiseReducer` runs two stages on every
      mic chunk (applied in the main loop before `turn_detector`/
      `wake_gate`, and in `speak_with_barge_in` after echo cancellation,
      same order WebRTC's pipeline uses and for the same reason — AEC
      assumes a linear mic/reference relationship that a nonlinear NR
      stage would distort if run first): a high-pass biquad (RBJ Audio
      EQ Cookbook formula, ~100Hz default) cutting deep engine rumble
      unconditionally, then adaptive spectral subtraction (Boll 1979 +
      Berouti et al.'s oversubtraction/floor) that continuously
      estimates the noise spectrum from low-energy frames and subtracts
      it from every frame, including ones with concurrent speech — the
      part that actually matters at highway speed, since the noise
      never stops. No noise sample was available, so defaults are
      reasoned from documented vehicle cabin acoustics (highway cabin
      noise ~65-71 dB(A), peaking 100-500Hz — [Road Noise and DSP: Tune
      for the Highway](https://audiointensity.com/blogs/dsp/road-noise-compensation),
      [GRAS: Road Noise Frequency](https://www.grasacoustics.com/industries/automotive/road-noise-testing))
      rather than measured, and validated against a synthetic noise
      proxy shaped to that spectrum, not a real recording.
      Found and fixed a real tuning mistake before shipping it: the
      first noise/non-noise decision (bottom 30th percentile of recent
      frame energy, adapting fast) got contaminated by real speech
      energy — inter-syllable dips aren't true silence, so ~30% of
      frames got treated as "noise" regardless, and the noise estimate
      drifted upward through an utterance (+48% over 6s in testing),
      increasingly over-subtracting real speech. Measured against real
      whisper.cpp transcription (synthesized speech + the noise proxy):
      this made transcription *worse* than doing nothing. Fixed with a
      stricter percentile (10th), slower adaptation (0.05), and longer
      history (150 frames); re-tested at 10dB/5dB SNR with transcription
      matching clean audio exactly, only degrading at unrealistically
      harsh 0dB/-5dB SNR. Off by default (`noise_reduction.enabled:
      false`); the module docstring and config.yaml's comments carry
      the full reasoning for anyone retuning it against a real vehicle
      recording.

## Known gaps / things to watch

- No persistence of conversation history across restarts.
- No handling yet for whisper.cpp or Piper being unreachable/not
  started — the loop will just raise on the first request.
- Barge-in currently cuts off the whole reply, not just the remaining
  audio — the full reply text is already in `chat.history` even if the
  user only heard part of it spoken. Fine for now; revisit if it
  causes the model to reference lines the user never heard.
- Echo cancellation (`audio/echo_cancel.py`): with speakers and mic on
  the same machine, `speak_with_barge_in` was self-interrupting almost
  immediately because the mic hears the assistant's own voice, which
  Silero scores at ~1.0 confidence for seconds at a stretch — a
  `prob_threshold`/`consecutive_frames`-only stopgap couldn't
  distinguish that from a real interruption at all. A single-channel
  NLMS adaptive filter, using the known reference (playback) signal,
  now subtracts a predicted copy of the echo before each chunk is
  scored. Two things had to be learned the hard way rather than
  guessed: (1) an initial version anchored tightly to one measured
  delay (~410ms ± a narrow 150ms window) worked perfectly replayed
  against its own calibration recording, but failed live in a fresh
  session — the delay isn't perfectly stable run to run, so the window
  needed widening (50-650ms) with a fast adaptation rate (`mu: 1.9`,
  near NLMS's 2.0 stability limit) to converge reliably within it; (2)
  the filter needs ~1-2s of real audio to converge at all, so
  `barge_in.grace_period_ms` fully disables the trigger check (not
  just loosens it) for that long at the start of every reply, while
  the canceller still runs and adapts. Headphones remain a simpler,
  fully-robust alternative if echo cancellation ever proves unreliable
  on a different machine — recalibrating
  `echo_cancel.expected_delay_ms`/`filter_length_ms` there is the first
  thing to try (see config.yaml's comments).

  Still self-interrupted after the above, on a real wake-word ->
  arithmetic-question exchange. Root cause: `MicStream`'s queue keeps
  queuing chunks continuously, including through the STT + LLM call +
  TTS synthesis that all block for real time between capturing the
  user's utterance and calling `speak_with_barge_in` — nobody's
  consuming the queue during that gap, so it backlogs (confirmed: a
  3.5s gap queued 109 chunks). The first chunks pulled afterward were
  then a burst of stale pre-reply audio consumed almost instantly
  (already sitting in the queue, no real wait) — desyncing both the
  echo canceller's reference alignment (assumes one chunk per loop
  iteration ≈ real elapsed time) and the grace period (counted in
  chunks, so it elapsed in a fraction of a real second rather than
  `grace_period_ms`). Fixed with `MicStream.drain()`, called at the
  start of `speak_with_barge_in` right before playback starts, so
  monitoring always begins from a clean "now" regardless of how long
  the preceding turn took to think. Re-verified across 6 runs with
  backlogs from 1 to 156 chunks (0.05s-5s of accumulated delay) — all
  passed, including the exact backlog size that reproduced the bug.

  Still not zero, though: a real "explain the Monty Hall paradox" reply
  self-interrupted just after the first sentence. Measured over 15
  trials replaying that exact reply (same text, same punctuation and
  paragraph breaks) end to end: 1/15 (~6.7%) false-triggered, at 3.76s
  in -- just past grace_period_ms, not during it. This is a real
  residual failure rate with the current settings, not something to
  claim is fully solved. Options if it keeps happening, roughly in
  order of effort: raise grace_period_ms (pushes the risk window later,
  doesn't shrink it for a reply that runs long); raise
  barge_in.consecutive_frames/prob_threshold further (reduces the rate
  at the cost of slightly slower real-interruption response);
  headphones (eliminates it entirely, since the mic then can't hear the
  speakers at all).
