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
      site cookies in if a tool call needs a logged-in session). Note:
      true headless Chromium gets served a CAPTCHA by search.brave.com
      specifically; `tools/browser.py` works around this by running
      headed-but-off-screen instead (see its docstring). Shell/exec
      tool access was deliberately deferred — no code for it exists
      yet.

      Enabled by default now (`tools.browser.enabled: true`) after
      fixing a real bug: with it off, a user asking for something that
      needs a live lookup (e.g. "what's the current price of diesel in
      Irvine, California") would get an answer from the model's stale
      internal knowledge, or worse — pushed hard enough to "use the
      tool", the model doesn't just refuse or guess, it can hallucinate
      its own tool-call syntax as plain text content, e.g.
      `<|tool_call>call:brave_search:search{queries:["..."]}<tool_call|>`,
      confirmed live against the actual LM Studio server. Since
      `message.tool_calls` is empty in that case (no tools were ever
      registered with the API request), that raw text fell through
      `ChatClient.ask()`'s existing logic as a normal reply and got
      spoken by Piper verbatim — reported as "tool tags sent to the
      audio channel". Root cause was `tools.browser.enabled: false`
      meaning no `tools` array was ever sent to the model at all, so a
      user request that implied tool use had no structured mechanism to
      satisfy it. Fixed three ways: (1) enabled the tool for real, so
      genuine `tool_calls` fire (confirmed clean end-to-end: real
      `web_search` dispatch, real result fed back, clean final answer,
      no leaked tags); (2) `main.py` now appends an explicit "you have
      real tools, you must use them for current/real-time info, never
      guess or claim no access" instruction to the system prompt, but
      *only* when tools are actually registered — telling the model it
      has tools it doesn't have is exactly what caused the
      hallucination, so this must stay conditional, never a static
      addition to `config.yaml`'s own `system_prompt`; (3) added
      `llm/chat_client.py`'s `_looks_like_leaked_tool_call()` as
      defense-in-depth — if a reply ever still looks like leaked
      tool-call syntax (a model hiccup, a future config/prompt
      mismatch), it's discarded in favor of a short fallback reply
      instead of being spoken verbatim, same rationale as
      `tts/tts_client.py`'s markdown stripping.

      `web_search` moved off search.brave.com shortly after: with the
      tool enabled for real use, Brave Search started serving its own
      "verifying you're not a bot" challenge to this automated flow —
      confirmed live, and confirmed it's not just the old
      headless-vs-off-screen CAPTCHA above (a fresh, never-used profile
      still got challenged, and so did a version with
      `--disable-blink-features=AutomationControlled` hiding
      `navigator.webdriver`, meaning Brave checks more than that one
      signal). That's Brave declining to allow this kind of automated
      access to their results page — their sanctioned path is the
      official Search API (api.search.brave.com), not deeper
      fingerprint spoofing to defeat the check, which is where this
      stopped rather than continuing to engineer around it.
      `web_search` now queries DuckDuckGo's server-rendered
      `html.duckduckgo.com/html/` endpoint instead (extracting just its
      `#links` results container, to avoid burning `max_chars` on a
      region/date filter menu) — confirmed clean, no challenge, still
      driving the same real Brave browser app via the same isolated
      profile. `web_fetch` was never affected (it loads whatever URL
      it's given, unrelated to either search provider).

      `web_search` now has a second provider option:
      `tools/firecrawl.py`'s `FirecrawlSearch` calls Firecrawl's hosted
      search API (`api.firecrawl.dev/v1/search`) directly over HTTPS,
      no local browser involved for search itself, returning
      structured (sometimes pre-scraped, markdown-formatted) results. Selected via
      `tools.search.provider` in `config.yaml` (`"duckduckgo"`, via the
      browser as above, or `"firecrawl"`); `tools/registry.py`'s
      `build_dispatcher` takes an optional separate `search_tool` so
      `web_fetch` always stays on the local Brave browser regardless of
      which one handles `web_search` — deliberately not routed through
      Firecrawl's own `/v1/scrape` fetch endpoint, by explicit
      instruction. The API key is read from `FIRECRAWL_API_KEY` in a
      `dot-env` file (main.py's `load_dotenv()`, a tiny dependency-free
      loader) — named `dot-env` rather than the usual `.env` by
      explicit preference (dotfiles are hidden by default), and
      gitignored either way: a real key must never land in
      `config.yaml`, which is tracked in git. Verified end-to-end
      through the real dispatcher and through a full `ChatClient.ask()`
      round-trip — real tool call, real Firecrawl result, clean spoken
      final answer.
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
- [x] **11. Automatic context compaction** — `llm/chat_client.py`
      queries LM Studio's own extended REST API
      (`/api/v0/models`, not part of the OpenAI-compatible surface) at
      startup for the loaded model's actual context length, and reads
      `usage.total_tokens` off every chat completion response to track
      how full the conversation actually is — no separate tokenizer or
      guesswork needed, both come straight from the server. Once usage
      reaches `llm.context_compaction_threshold` (0.8 by default) of
      that limit, `main.py` speaks a brief notice
      ("One moment, I'm compacting our conversation...") and calls
      `ChatClient.compact()`, which asks the model to summarize the
      conversation so far (skipping raw tool-call/tool-result plumbing,
      which is noisy and not useful once compacted) and replaces the
      full turn-by-turn history with just the system prompt plus that
      summary. Verified end-to-end against the real server with an
      artificially tiny `context_limit`: usage tracked correctly turn
      by turn, compaction triggered at the right threshold, and a
      follow-up question after compaction ("what city do I live in
      again?") was answered correctly from the summary alone. If the
      server doesn't expose `/api/v0/models` (a different
      OpenAI-compatible backend, or an older LM Studio),
      `context_limit` stays `None` and compaction simply never
      triggers, rather than guessing a number that could compact too
      early or too late — `main.py` prints a note at startup when this
      happens.
- [x] **12. Phone UI** — lets a phone (Safari or Chrome on iOS — both
      WebKit, identical behavior) act as the mic+speaker instead of this
      machine's own, over a small page served by the Python process
      itself. Two big buttons: **Enable** (starts/stops streaming the
      phone's mic) and **Stop** (interrupts TTS playback, the manual
      equivalent of barge-in). Off by default (`phone_ui.enabled: false`).

      Deliberately built as a second implementation of the exact
      interface `audio/capture.py`'s `MicStream` already exposes
      (`start/stop/chunks/play/is_playing/stop_playback/drain`, plus one
      new `consume_stop_signal()` method that `MicStream` gets as a
      trivial always-`False` stub) — `audio/phone_stream.py`'s
      `PhoneMicStream` is that second implementation, so `main.py` only
      needed to pick which one to construct (`phone_ui.enabled`) and add
      one extra check in `speak_with_barge_in`'s loop; STT/LLM/TTS and
      wake word/noise reduction needed zero changes, since they were
      already audio-source-agnostic.

      `web/server.py`'s `PhoneUIServer` owns the transport: a `websockets`
      server (new dependency) running its own asyncio event loop on a
      dedicated background thread (the only other thread in the codebase
      besides `MicStream`'s play-lock), serving `web/static/*` (the page,
      `app.js`, an `AudioWorkletProcessor` for mic capture) and the
      WebSocket endpoint from one port. HTTPS is mandatory, not a
      preference: iOS only grants `getUserMedia` in a secure context, and
      there's no user-facing override for that on iOS the way desktop
      Chrome has one — so a self-signed cert is generated once via the
      system `openssl` binary (subprocess, no new pip dependency) and
      cached under `phone_ui.cert_dir`. The phone shows a "not private"
      warning once; there's no way around that for a self-signed cert.
      A shared-secret `?token=` query param (auto-generated if unset)
      gates the page, since this listens on the LAN.

      `PhoneMicStream` resamples incoming phone audio (native rate,
      declared via a `hello` message) to 16kHz using the *already
      existing* `audio/echo_cancel.py`'s `resample_linear` — no new
      resampling code, and it keeps `app.js` maximally simple (it never
      resamples). A background heartbeat thread keeps `chunks()` yielding
      silent chunks whenever the phone isn't actively streaming (idle,
      disconnected, disabled) — without it, `speak_with_barge_in`'s
      `for chunk in mic.chunks(): ...` loop would block forever the first
      time nothing was being streamed, since (unlike PortAudio's local
      callback) nothing guarantees a phone produces data on its own.
      `main.py` also forces the NLMS `EchoCanceller` off in phone mode
      regardless of `barge_in.echo_cancel.enabled` — its fixed delay/
      filter-length were tuned for this Mac's own CoreAudio path and are
      meaningless (possibly harmful to VAD scoring) over a phone's own
      audio stack plus network jitter; phone mode relies on the browser's
      own built-in AEC (`getUserMedia`'s `echoCancellation` constraint)
      instead, same as any WebRTC calling app would.

      Verified two ways. (1) `tests/test_phone_stream.py`: a real
      `websockets` client (a "fake phone") against the real server — real
      TLS handshake with the real self-signed cert, real token gating
      (wrong token gets a real 403), real resampling/re-chunking math,
      `play()`/`is_playing()`/`stop_playback()`/`consume_stop_signal()`
      state transitions including the Stop-button and
      `playback_finished`-ack paths, and the heartbeat keeping `chunks()`
      alive with silence when idle — 15/15 passing, no real hardware
      needed. (2) A full live run: `main.py` started for real against the
      already-running whisper.cpp/LM Studio/Piper, driven by the same
      kind of fake-phone client streaming a real Piper-synthesized
      utterance ("What is two plus two?") — real whisper.cpp
      transcription, a real LM Studio reply ("Two plus two is four."),
      real Piper synthesis, and a real WAV sent back over the WebSocket,
      all through the actual `main.py` loop with no code path bypassed.

      What's *not* verified: the actual on-device Safari/iOS experience
      (the mic permission prompt, the self-signed cert warning flow, real
      playback through a real phone speaker) — nothing in this
      environment can drive a real iPhone, so that needs a human to check
      once by hand.

## Known gaps / things to watch

- [x] **Reconnection for whisper.cpp/Piper/LM Studio going down** --
      previously any of the three external processes being unreachable
      (killed, crashed, or not started yet) raised an uncaught exception
      straight out of `mic.chunks()`'s loop and crashed the whole voice
      assistant. Fixed with a matching pair in each client
      (`stt/whisper_client.py`'s `WhisperUnavailableError`,
      `tts/tts_client.py`'s `PiperUnavailableError`,
      `llm/chat_client.py`'s `LlmUnavailableError`, each with an
      `is_available()` health check) and `main.py`'s
      `call_with_reconnect()`, which wraps every call into one of the
      three (transcribe, ask, synthesize, compact) so that on failure it
      prints a notice, polls `is_available()` at `reconnect.
      poll_interval_s` until the service responds again, then retries
      the same call rather than dropping the turn or crashing.
      `WhisperClient.transcribe()` also retries a single dropped/blipped
      request internally (`stt.max_retries`/`retry_backoff_s`) before
      treating the server as actually down, so a one-off network hiccup
      doesn't trigger the full reconnect flow. `ChatClient.ask()` rolls
      the conversation history back to exactly its pre-call state on
      failure (it appends the user turn before calling the API) --
      without this, a retried `ask()` after reconnecting would leave two
      consecutive user turns in history with no assistant reply between
      them, which several local chat templates (including Gemma's)
      require to strictly alternate. Verified with mocked failures
      (`tests/test_service_reconnect.py`, no real servers needed): retry
      behavior, the three `*UnavailableError` types, `is_available()`
      accuracy, `ChatClient.ask()`'s history rollback and clean retry,
      and `call_with_reconnect()`'s resume-once-available behavior — all
      passing. Also verified live against real processes
      (`tests/test_live_reconnect_smoke.py`, no mocking): a disposable
      whisper.cpp server (separate port, the real one left running)
      transcribing real Piper-synthesized audio, `SIGKILL`ed outright,
      respawned unattended by a background thread 3s later --
      `call_with_reconnect` genuinely blocked, noticed the respawn, and
      returned the correct transcription; same pattern for Piper
      (binary path swapped to bogus, then back, on a timer). LM Studio's
      reconnect path was verified live too, once a real LM Studio server
      came up on the machine this was tested on: rather than killing the
      user's actual running server (a GUI app with no scripted restart
      available), the test pointed a real `ChatClient` at a genuinely
      closed port (a real OS-level connection refusal) to get a real
      `LlmUnavailableError`/`is_available() == False`, then retargeted
      the same client at the real, already-running server on a delayed
      background thread — `call_with_reconnect` waited (~4.9s) and came
      back with an actual completion from the real loaded model
      (`google/gemma-4-26b-a4b-qat`, which replied exactly
      "RECONNECTED" as asked), with history left as a clean single
      user/assistant pair, not a dangling duplicate. All three services'
      reconnect paths are now verified against real processes, not just
      mocks (`tests/test_live_reconnect_smoke.py`).

      Aside, unrelated to this fix: the LM Studio server encountered
      during that live run was a process named `Bionic`, which
      CLAUDE.md explicitly says this project should not use (core
      LM Studio app only). It answered the OpenAI-compatible API
      correctly and served the exact model config.yaml expects, so the
      test itself is unaffected, but flagged to the user in case that
      wasn't the intended app to have running.
- Echo cancellation convergence is still a fixed timer
  (`barge_in.grace_period_ms`), not a measured/detected convergence
  state. For at least one reply text, the echo canceller hasn't fully
  converged by the time the timer elapses, and the leftover echo can
  reach VAD confidence similar to real speech -- a small residual
  self-trigger risk that threshold tuning alone can't fully eliminate
  without also suppressing real barge-in (see the barge-in section
  above for the measured trade-off). The real fix is making the
  canceller converge faster/more reliably, or detecting convergence
  directly (e.g. from filter weight stability or residual energy)
  instead of trusting a fixed delay for every possible reply.
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
  in -- just past grace_period_ms, not during it.

  Resolved by raising the bar rather than guessing at new numbers:
  `tests/tune_barge_in_thresholds.py` logs full per-chunk probability
  traces across repeated live replays without ever actually stopping
  playback (so a reply always completes regardless of what a threshold
  would have done), then sweeps candidate (prob_threshold,
  consecutive_frames) pairs against the logged data. Across 20 traces
  of the same reply, 2 false-triggered at the old 0.9/16 -- and both
  came within 1 chunk of still crossing even 0.93/16 (15/16
  consecutive), so 0.93 and 24 were chosen together for real headroom
  rather than either bare minimum. Confirmed 0/20 against the logged
  traces, then re-verified live with actual stopping enabled
  (`--confirm` mode, not just trace replay): 10/10 passed. Re-run that
  script (or `tests/validate_echo_canceller.py` /
  `tests/measure_echo_delay.py`) rather than guessing if this ever
  needs retuning on different hardware -- headphones remain the
  fallback if it keeps happening anyway (eliminates it entirely, since
  the mic then can't hear the
  speakers at all).

  Regression found right after shipping the 0.93/24 retune above: live
  testing found barge-in didn't fire *at all* any more, on any reply.
  Root cause was in `main.speak_with_barge_in` itself, not the
  thresholds: it skipped calling `turn_detector.vad.speech_prob()`
  entirely for every chunk inside `grace_period_chunks` (not just
  skipping the *action* on those chunks). Silero VAD is a streaming
  model that carries internal state between calls, and starving it for
  the whole ~2s grace period left it under-confident for a while after
  scoring resumed -- confirmed directly (`tests/debug_grace_period_hypothesis.py`):
  the same real interruption, scored the broken way, cleared 0.93 on
  0/79 chunks; scored with speech_prob() called every chunk (grace flag
  only gating the *action*, matching every diagnostic script's
  behavior, which is why the tuning above never caught this), it
  cleared 0.93 on 79/470 chunks, peak 0.9999. Fixed in `main.py`: always
  call `speech_prob()`, gate only the trigger action on the grace
  period. Confirmed live post-fix with a real person genuinely
  interrupting at different points in a reply
  (`tests/confirm_real_barge_in.py`): 4/4 real interruptions fired in
  the "middle"/"late" rounds (early attempts inside the still-active
  grace period correctly don't fire, by design).

  That fix also surfaced a second, real trade-off rather than a bug:
  on at least one reply text different from the Monty Hall one above,
  the echo canceller hadn't fully converged by the time
  grace_period_ms elapsed, and the leftover echo reached similar
  sustained VAD confidence to real speech -- 2/3 live self-triggers on
  that text at 0.93/24. Raising the bar further (tried 0.93/32, then
  0.97/32, using `tests/sweep_thresholds_fast.py` -- a faster variant
  of the trace sweep that reuses one long-lived model instance across
  trials instead of reloading it every time, which also turned out to
  matter: a combination that read 0/15 in that sweep still
  live-false-triggered 1/5 when re-confirmed with a freshly-created
  model per trial, closer to what `tune_barge_in_thresholds.py
  --confirm` and real conversational turns look like) did eliminate
  that self-trigger (6/6 clean at 0.97/32) but also suppressed genuine
  interruption on the same text (0/3 real attempts fired in the same
  live round) -- past a point, confident echo and confident real
  speech overlap enough on this text that no (prob_threshold,
  consecutive_frames) pair separates them cleanly. Deliberately kept at
  0.93/24 (real barge-in confirmed reliable, small known residual
  self-trigger risk on some reply audio) rather than trade away real
  functionality for a threshold that might not fire at all -- see
  config.yaml's comment for the full reasoning. The real fix for the
  residual risk is a more robust echo canceller (faster/more reliable
  convergence, or detecting convergence directly instead of trusting a
  fixed grace timer for every possible reply), not further threshold
  tweaking; noted below under known gaps.

  Separately, intermittent crackling was reported in TTS output.
  Measured rather than guessed: `tests/measure_output_underruns.py`
  ran the real barge-in monitoring loop (echo cancellation + VAD)
  concurrently with playback and found 0 PortAudio-reported output
  underflows and per-chunk processing comfortably within its 32ms
  budget (p50 ~7ms) even over a 2.75-minute reply -- ruling out a
  Python/GIL performance problem. `tests/compare_samplerate_crackle.py`
  then ruled out a plain samplerate mismatch (Piper's 16kHz vs. the
  output device's native 48kHz): identical either way. Isolating
  further by ear: plain playback alone (no concurrent mic capture) was
  clean; a concurrent but *idle* mic stream (draining chunks, no
  processing) was also clean; only concurrent real processing (echo
  cancellation + VAD, i.e. actual barge-in monitoring) reproduced the
  crackle -- even though that processing measured well within budget
  and neither stream ever reported an underrun. Root cause: mic capture
  and TTS playback were two independently opened, independently
  clocked PortAudio streams (`audio.capture.MicStream`'s InputStream
  and `audio.playback`'s separate `sd.play()` output stream); on this
  machine the built-in mic and speakers are exposed as separate
  CoreAudio devices even though physically one machine, so a brief
  scheduling hiccup in one stream isn't synchronized with the other,
  which can glitch the DAC without either stream's own `status` flag
  ever being set. Fixed by merging capture and playback into one
  full-duplex `sd.Stream` (`MicStream.play()`/`is_playing()`/
  `stop_playback()` in `audio/capture.py`, wired into
  `main.speak_with_barge_in`) so input and output are produced in the
  same callback, on the same clock, by construction -- not just given
  a bigger buffer or less work to dodge the same race. Re-confirmed
  clean on the same 2.75-minute reply through the new path, and
  re-confirmed barge-in itself still fires correctly with a precisely
  timed real interruption. `audio.playback`'s own `play_wav_bytes_async`/
  `is_playing`/`stop` remain for standalone playback with no
  concurrent mic capture (a one-off script, etc.) -- only the
  concurrent-capture-and-playback path needed to change.
