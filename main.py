"""Entry point: hands-free voice loop.

listen -> VAD detects end of utterance -> whisper.cpp transcribes ->
LM Studio generates a reply -> Piper speaks it -> back to listening.

Run `python main.py` after starting LM Studio's server and whisper.cpp's
server (see README.md).
"""

import os
import re
import sys
import time

import numpy as np
import yaml

from audio import playback
from audio.capture import MicStream
from audio.echo_cancel import EchoCanceller, resample_linear
from audio.noise_reduce import NoiseReducer
from audio.vad import SileroVAD, TurnDetector
from audio.wake_word import WakeWordGate
from llm.chat_client import ChatClient, LlmUnavailableError
from stt.whisper_client import WhisperClient, WhisperUnavailableError
from tts.tts_client import PiperTTS, PiperUnavailableError

CHUNK_SAMPLES = 512  # required window size for Silero VAD at 16kHz
DEFAULT_CANCEL_PHRASES = ["cancel", "nevermind", "never mind", "forget it"]
COMPACTION_NOTICE = "One moment, I'm compacting our conversation to keep things running smoothly."

# Any of these means "the external process/server this call depends on
# is unreachable" -- see wait_for_service/call_with_reconnect below.
SERVICE_UNAVAILABLE_ERRORS = (WhisperUnavailableError, PiperUnavailableError, LlmUnavailableError)


def wait_for_service(name: str, is_available, poll_interval_s: float = 3.0):
    """Blocks, polling is_available() every poll_interval_s, until it
    returns True. Only prints on state change (not every poll) so a long
    outage doesn't spam the console."""
    print(f"({name} is unreachable -- waiting for it to come back...)")
    while not is_available():
        time.sleep(poll_interval_s)
    print(f"({name} is back.)")


def call_with_reconnect(fn, service_name: str, is_available, poll_interval_s: float = 3.0):
    """Calls fn(); if it raises one of SERVICE_UNAVAILABLE_ERRORS, waits
    for that service to come back (via is_available) and retries fn()
    with the same arguments, indefinitely. This is what makes whisper.cpp
    or Piper being killed and restarted mid-session a "wait a moment"
    experience instead of a crash -- the turn resumes from wherever it
    was once the server responds again, rather than being dropped."""
    while True:
        try:
            return fn()
        except SERVICE_UNAVAILABLE_ERRORS as exc:
            print(f"({service_name} error: {exc})")
            wait_for_service(service_name, is_available, poll_interval_s)


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def load_dotenv(path="dot-env"):
    """Tiny, dependency-free loader for secrets that must never land in
    config.yaml (which is tracked in git) -- KEY=value lines, '#'
    comments and blank lines ignored. Named dot-env rather than the
    usual .env by explicit preference (dotfiles are hidden by default).
    Only sets a var if it isn't already in the environment, so an
    explicitly exported value always wins over the file."""
    if not os.path.exists(path):
        return
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            os.environ.setdefault(key.strip(), value.strip())


def is_cancel_phrase(text: str, cancel_phrases) -> bool:
    """True if text, once normalized, IS one of cancel_phrases exactly --
    not just contains one, so "cancel my 3pm meeting" is a real command,
    not an abort."""
    normalized = re.sub(r"[^a-z0-9\s]", "", text.lower()).strip()
    normalized = re.sub(r"\s+", " ", normalized)
    return normalized in cancel_phrases


def speak_with_barge_in(
    wav_bytes,
    mic,
    turn_detector,
    prob_threshold,
    consecutive_frames,
    echo_canceller=None,
    grace_period_chunks=0,
    noise_reducer=None,
) -> bool:
    """Plays a reply; stops if the mic picks up the user talking over it.

    With speakers and mic on the same machine, the mic also picks up the
    assistant's own voice — measured on this machine at a real, stable
    ~410ms delay, strong enough for Silero to score it at ~1.0 confidence
    for seconds at a stretch, indistinguishable from a real interruption
    by threshold alone. echo_canceller (audio/echo_cancel.py) subtracts a
    predicted copy of that echo, using the known reference (playback)
    signal, before each chunk is scored.

    The adaptive filter needs a couple of seconds of real audio to
    converge, though — measured across repeated live runs, false triggers
    only ever happened within roughly the first 1.5-2s of a reply, never
    after. grace_period_chunks fully disables the trigger check (not just
    loosens it) for that many chunks at the start of playback, while still
    running the canceller so it's converged by the time barge-in becomes
    live -- no interruption at all is possible in that window, which is
    fine (a few seconds, not the whole reply).

    Returns True if playback was cut short. The chunk that triggered the cut
    is fed into turn_detector so capture of the new utterance starts right
    away, without waiting for the next full loop iteration.
    """
    # Decoded and resampled to mic.sample_rate regardless of whether echo
    # cancellation is enabled: playback now goes out through mic's own
    # duplex stream (see audio/capture.py), which only has one samplerate
    # for both directions, rather than through a second, independently
    # clocked output stream -- that's what used to cause intermittent
    # crackling in the TTS output during barge-in monitoring, confirmed via
    # tests/measure_output_underruns.py and tests/compare_samplerate_crackle.py.
    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    ref_audio = resample_linear(raw_ref, ref_sr, mic.sample_rate)
    if echo_canceller is not None:
        echo_canceller.reset()

    # Between the last chunk of the user's utterance and here, STT + the LLM
    # call + TTS synthesis all block for real time while the mic keeps
    # queuing chunks nobody's consuming. Without draining that backlog, the
    # first chunks pulled below would be a burst of stale pre-reply audio
    # consumed almost instantly (queue already has the data, no real wait) --
    # which desyncs both the echo canceller's reference alignment (which
    # assumes one chunk per loop iteration tracks real elapsed time) and the
    # grace period (counted in chunks, so it'd elapse in a fraction of a
    # real second instead of grace_period_ms). Confirmed by reproducing it:
    # a 3.5s gap before this point queued 109 backlogged chunks and caused
    # an early false trigger at 1.9s elapsed, well inside the grace period.
    mic.drain()
    mic.play(ref_audio)
    consecutive_speech = 0
    chunk_index = 0
    ref_pos = 0
    for chunk in mic.chunks():
        if not mic.is_playing():
            return False

        if echo_canceller is not None:
            n = len(chunk)
            ref_slice = ref_audio[ref_pos : ref_pos + n]
            if len(ref_slice) < n:
                ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
            ref_pos += n
            echo_canceller.push_reference(ref_slice)
            scored_chunk = echo_canceller.cancel(chunk)
        else:
            scored_chunk = chunk

        # Noise reduction after echo cancellation, not before: AEC's adaptive
        # filter assumes a linear relationship between mic and reference
        # signal, and noise reduction's per-bin gain is a nonlinear operation
        # that would distort that assumption if applied first (the standard
        # order in e.g. WebRTC's audio pipeline is AEC, then noise
        # suppression, for the same reason).
        if noise_reducer is not None:
            scored_chunk = noise_reducer.process(scored_chunk)

        in_grace_period = chunk_index < grace_period_chunks
        chunk_index += 1

        # Silero VAD is a streaming model that carries internal state
        # between calls -- it must be fed every chunk to stay "warmed up",
        # even chunks we don't act on. Skipping speech_prob() entirely for
        # the whole grace period (instead of just not acting on it) left
        # the model uncalled for ~2s, and real barge-in attempts right
        # after the grace period then scored consistently near zero even
        # while genuinely talking (confirmed: 0/79 chunks reached the
        # 0.93 threshold with the call skipped during grace, vs. 79
        # reaching it, some as high as 0.9999, in the same recording with
        # the call kept live throughout). So: always score, only gate the
        # trigger action on the grace period.
        prob = turn_detector.vad.speech_prob(scored_chunk)
        if in_grace_period:
            continue

        if prob >= prob_threshold:
            consecutive_speech += 1
            if consecutive_speech >= consecutive_frames:
                mic.stop_playback()
                turn_detector.process_chunk(scored_chunk)
                return True
        else:
            consecutive_speech = 0
    return False


def main():
    load_dotenv()
    cfg = load_config()
    sample_rate = cfg["audio"]["sample_rate"]

    mic = MicStream(
        sample_rate=sample_rate,
        chunk_samples=CHUNK_SAMPLES,
        channels=cfg["audio"]["channels"],
        device=cfg["audio"]["input_device"],
        output_device=cfg["audio"]["output_device"],
    )

    vad = SileroVAD(sample_rate=sample_rate)
    turn_detector = TurnDetector(
        vad,
        sample_rate=sample_rate,
        chunk_samples=CHUNK_SAMPLES,
        prob_threshold=cfg["vad"]["speech_prob_threshold"],
        silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
        min_speech_ms=cfg["vad"]["min_speech_ms"],
    )

    barge_in_cfg = cfg.get("barge_in", {})
    barge_in_prob_threshold = barge_in_cfg.get("prob_threshold", 0.9)
    barge_in_consecutive_frames = barge_in_cfg.get("consecutive_frames", 16)
    barge_in_chunk_ms = 1000 * CHUNK_SAMPLES / sample_rate
    grace_period_chunks = int(barge_in_cfg.get("grace_period_ms", 2000) / barge_in_chunk_ms)

    echo_cfg = barge_in_cfg.get("echo_cancel", {})
    echo_canceller = None
    if echo_cfg.get("enabled", True):
        echo_canceller = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg.get("expected_delay_ms", 50.0),
            filter_length_ms=echo_cfg.get("filter_length_ms", 600.0),
            mu=echo_cfg.get("mu", 1.9),
        )

    noise_cfg = cfg.get("noise_reduction", {})
    noise_reducer = None
    if noise_cfg.get("enabled", False):
        noise_reducer = NoiseReducer(
            sample_rate=sample_rate,
            chunk_size=CHUNK_SAMPLES,
            highpass_hz=noise_cfg.get("highpass_hz", 100.0),
            oversubtraction=noise_cfg.get("oversubtraction", 1.2),
            spectral_floor=noise_cfg.get("spectral_floor", 0.1),
            noise_update_rate=noise_cfg.get("noise_update_rate", 0.05),
        )

    reconnect_poll_interval_s = cfg.get("reconnect", {}).get("poll_interval_s", 3.0)

    stt = WhisperClient(
        server_url=cfg["stt"]["whisper_server_url"],
        sample_rate=sample_rate,
        language=cfg["stt"]["language"],
        request_timeout_s=cfg["stt"].get("request_timeout_s", 30.0),
        max_retries=cfg["stt"].get("max_retries", 2),
        retry_backoff_s=cfg["stt"].get("retry_backoff_s", 1.0),
    )

    wake_cfg = cfg.get("wake_word", {})
    wake_enabled = wake_cfg.get("enabled", False)
    wake_phrase = wake_cfg.get("phrase", "")
    wake_prompt = f"The wake word is {wake_phrase.capitalize()}." if wake_phrase else None
    wake_gate = None
    if wake_enabled:
        wake_gate = WakeWordGate(
            vad,
            sample_rate=sample_rate,
            chunk_samples=CHUNK_SAMPLES,
            prob_threshold=cfg["vad"]["speech_prob_threshold"],
            silence_hangover_ms=wake_cfg.get("silence_hangover_ms", 500),
            min_speech_ms=wake_cfg.get("min_speech_ms", 150),
            silence_before_ms=wake_cfg.get("silence_before_ms", 1500),
        )
    listening_for_wake_word = wake_enabled
    command_timeout_chunks = int(wake_cfg.get("command_timeout_ms", 3000) / barge_in_chunk_ms)
    command_wait_chunks = 0
    cancel_phrases = {p.strip().lower() for p in wake_cfg.get("cancel_phrases", DEFAULT_CANCEL_PHRASES)}

    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
        synth_timeout_s=cfg["tts"].get("synth_timeout_s", 30.0),
    )

    browser = None
    tool_schemas = None
    tool_dispatch = None
    system_prompt = cfg["llm"].get("system_prompt", "")
    browser_cfg = cfg.get("tools", {}).get("browser", {})
    search_cfg = cfg.get("tools", {}).get("search", {})
    if browser_cfg.get("enabled"):
        from tools.browser import BraveTools
        from tools.registry import TOOL_SCHEMAS, build_dispatcher

        browser = BraveTools(
            profile_dir=browser_cfg["profile_dir"],
            headless=browser_cfg.get("headless", True),
            timeout_ms=browser_cfg.get("timeout_ms", 15000),
        )

        # web_fetch always uses `browser` above (the local Brave browser),
        # regardless of this setting -- only web_search has a choice of
        # provider. "firecrawl" needs FIRECRAWL_API_KEY (see dot-env,
        # loaded above); missing it degrades to an error string from the
        # tool call itself rather than failing startup, same as any other
        # tool-call failure the model can see and react to.
        search_tool = None
        search_provider = search_cfg.get("provider", "duckduckgo")
        if search_provider == "firecrawl":
            from tools.firecrawl import FirecrawlSearch

            firecrawl_cfg = search_cfg.get("firecrawl", {})
            search_tool = FirecrawlSearch(
                api_key=os.environ.get("FIRECRAWL_API_KEY"),
                result_limit=firecrawl_cfg.get("result_limit", 5),
            )

        tool_schemas = TOOL_SCHEMAS
        tool_dispatch = build_dispatcher(browser, search_tool)
        # Only true when tools are actually registered with the API below --
        # telling the model it has tools it doesn't have is what caused it to
        # hallucinate its own tool-call syntax as spoken text in the first
        # place (see PLAN.md's tool-use section). Also: without an explicit
        # "you must call this, don't guess" instruction, smaller/quantized
        # models often skip the tool and answer from stale internal
        # knowledge instead, even when the schema is right there.
        system_prompt += (
            "\n\nYou have real tools available: web_search and web_fetch, "
            "which load real, current web pages through a real browser. "
            "When a question depends on current, real-time, or "
            "location-specific information you cannot know for certain -- "
            "prices, weather, news, sports scores, stock prices, and "
            "similar -- you must call web_search (or web_fetch) to get a "
            "real answer. Never guess, estimate, or answer from memory for "
            "this kind of question, and never say you do not have access "
            "to search or browsing -- you do."
        )

    chat = ChatClient(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        system_prompt=system_prompt,
        temperature=cfg["llm"].get("temperature", 0.7),
        tools=tool_schemas,
        tool_dispatch=tool_dispatch,
        compaction_threshold=cfg["llm"].get("context_compaction_threshold", 0.8),
        request_timeout_s=cfg["llm"].get("request_timeout_s", 60.0),
    )
    if chat.context_limit is None:
        print("(context length not available from this server -- automatic compaction disabled)")

    print("Listening... (Ctrl+C to quit)" if not listening_for_wake_word else f'Waiting for "{wake_phrase}"...')
    mic.start()
    try:
        for chunk in mic.chunks():
            if noise_reducer is not None:
                chunk = noise_reducer.process(chunk)

            if listening_for_wake_word:
                candidate = wake_gate.process_chunk(chunk)
                if candidate is None:
                    continue
                heard = call_with_reconnect(
                    lambda: stt.transcribe(candidate, prompt=wake_prompt),
                    "whisper.cpp",
                    stt.is_available,
                    reconnect_poll_interval_s,
                )
                # Must be the FIRST word, not just present anywhere in the
                # transcript. WakeWordGate only guarantees silence preceded
                # the utterance as a whole -- if the whole utterance is a
                # longer sentence with no pause inside it (e.g. "I think
                # gemma is a nice name"), a plain substring match would
                # fire on "gemma" mid-sentence, which never had any silence
                # actually in front of it. Requiring the match at the
                # start (mod leading whitespace/punctuation) plus a word
                # boundary is what actually ties the check to the phrase
                # itself, not just to the utterance it happened to be part of.
                match = re.match(
                    r"^[\s,.!?—-]*(" + re.escape(wake_phrase) + r")\b", heard, re.IGNORECASE
                )
                if match is None:
                    continue
                match_index = match.start(1)
                print("Wake word heard — listening...")
                listening_for_wake_word = False
                command_wait_chunks = 0

                # If the user kept talking right after the wake word with no
                # pause, whisper transcribes it all as one utterance — the
                # part after the phrase is the start of their actual query,
                # not a separate thing to wait for. Losing this was the bug:
                # only the yes/no wake-word match used to survive, and
                # whatever came with it in the same breath was discarded.
                remainder = heard[match_index + len(wake_phrase) :].strip().strip(",.!?—-").strip()
                if not remainder:
                    continue
                text = remainder
            else:
                # Easy to wake it up and then forget what you meant to ask.
                # Track how long it's been pure silence (never even started
                # speaking) since the wake word -- once that exceeds
                # command_timeout_ms, give up and go back to waiting for the
                # phrase rather than leaving the mic "listening" forever.
                # Reset the moment real speech starts; from there the normal
                # end-of-utterance silence_hangover_ms takes over as usual.
                was_in_speech = turn_detector.in_speech
                utterance = turn_detector.process_chunk(chunk)
                if wake_enabled and not was_in_speech and not turn_detector.in_speech:
                    command_wait_chunks += 1
                    if command_wait_chunks >= command_timeout_chunks:
                        print("No command heard — back to waiting for the wake word.")
                        listening_for_wake_word = True
                        command_wait_chunks = 0
                        print(f'Waiting for "{wake_phrase}"...')
                    continue
                command_wait_chunks = 0
                if utterance is None:
                    continue

                text = call_with_reconnect(
                    lambda: stt.transcribe(utterance),
                    "whisper.cpp",
                    stt.is_available,
                    reconnect_poll_interval_s,
                )
                if not text:
                    continue
            print(f"You said: {text}")

            if is_cancel_phrase(text, cancel_phrases):
                print("Cancelled.")
                if wake_enabled:
                    listening_for_wake_word = True
                    print(f'Waiting for "{wake_phrase}"...')
                continue

            reply = call_with_reconnect(
                lambda: chat.ask(text), "LM Studio", chat.is_available, reconnect_poll_interval_s
            )
            print(f"Assistant: {reply}")

            wav_bytes = call_with_reconnect(
                lambda: tts.synthesize(reply), "Piper", tts.is_available, reconnect_poll_interval_s
            )
            if speak_with_barge_in(
                wav_bytes,
                mic,
                turn_detector,
                barge_in_prob_threshold,
                barge_in_consecutive_frames,
                echo_canceller,
                grace_period_chunks,
                noise_reducer,
            ):
                print("(interrupted)")

            if chat.needs_compaction():
                print(f"(context at {chat.last_total_tokens}/{chat.context_limit} tokens -- compacting)")
                notice_wav = call_with_reconnect(
                    lambda: tts.synthesize(COMPACTION_NOTICE),
                    "Piper",
                    tts.is_available,
                    reconnect_poll_interval_s,
                )
                speak_with_barge_in(
                    notice_wav,
                    mic,
                    turn_detector,
                    barge_in_prob_threshold,
                    barge_in_consecutive_frames,
                    echo_canceller,
                    grace_period_chunks,
                    noise_reducer,
                )
                call_with_reconnect(chat.compact, "LM Studio", chat.is_available, reconnect_poll_interval_s)
                print("(context compacted)")

            if wake_enabled:
                listening_for_wake_word = True
                print(f'Waiting for "{wake_phrase}"...')
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mic.stop()
        if browser is not None:
            browser.stop()


if __name__ == "__main__":
    sys.exit(main())
