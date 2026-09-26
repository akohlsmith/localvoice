"""Entry point: hands-free voice loop.

listen -> VAD detects end of utterance -> whisper.cpp transcribes ->
LM Studio generates a reply -> Piper speaks it -> back to listening.

Run `python main.py` after starting LM Studio's server and whisper.cpp's
server (see README.md).
"""

import re
import sys

import numpy as np
import yaml

from audio import playback
from audio.capture import MicStream
from audio.echo_cancel import EchoCanceller, resample_linear
from audio.vad import SileroVAD, TurnDetector
from audio.wake_word import WakeWordGate
from llm.chat_client import ChatClient
from stt.whisper_client import WhisperClient
from tts.tts_client import PiperTTS

CHUNK_SAMPLES = 512  # required window size for Silero VAD at 16kHz


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def speak_with_barge_in(
    wav_bytes,
    mic,
    turn_detector,
    prob_threshold,
    consecutive_frames,
    echo_canceller=None,
    grace_period_chunks=0,
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
    ref_audio = None
    ref_pos = 0
    if echo_canceller is not None:
        raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
        ref_audio = resample_linear(raw_ref, ref_sr, echo_canceller.sample_rate)
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
    playback.play_wav_bytes_async(wav_bytes)
    consecutive_speech = 0
    chunk_index = 0
    for chunk in mic.chunks():
        if not playback.is_playing():
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

        in_grace_period = chunk_index < grace_period_chunks
        chunk_index += 1
        if in_grace_period:
            continue

        prob = turn_detector.vad.speech_prob(scored_chunk)
        if prob >= prob_threshold:
            consecutive_speech += 1
            if consecutive_speech >= consecutive_frames:
                playback.stop()
                turn_detector.process_chunk(chunk)
                return True
        else:
            consecutive_speech = 0
    return False


def main():
    cfg = load_config()
    sample_rate = cfg["audio"]["sample_rate"]

    mic = MicStream(
        sample_rate=sample_rate,
        chunk_samples=CHUNK_SAMPLES,
        channels=cfg["audio"]["channels"],
        device=cfg["audio"]["input_device"],
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

    stt = WhisperClient(
        server_url=cfg["stt"]["whisper_server_url"],
        sample_rate=sample_rate,
        language=cfg["stt"]["language"],
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

    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )

    browser = None
    tool_schemas = None
    tool_dispatch = None
    browser_cfg = cfg.get("tools", {}).get("browser", {})
    if browser_cfg.get("enabled"):
        from tools.browser import BraveTools
        from tools.registry import TOOL_SCHEMAS, build_dispatcher

        browser = BraveTools(
            profile_dir=browser_cfg["profile_dir"],
            headless=browser_cfg.get("headless", True),
            timeout_ms=browser_cfg.get("timeout_ms", 15000),
        )
        tool_schemas = TOOL_SCHEMAS
        tool_dispatch = build_dispatcher(browser)

    chat = ChatClient(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        system_prompt=cfg["llm"].get("system_prompt", ""),
        temperature=cfg["llm"].get("temperature", 0.7),
        tools=tool_schemas,
        tool_dispatch=tool_dispatch,
    )

    print("Listening... (Ctrl+C to quit)" if not listening_for_wake_word else f'Waiting for "{wake_phrase}"...')
    mic.start()
    try:
        for chunk in mic.chunks():
            if listening_for_wake_word:
                candidate = wake_gate.process_chunk(chunk)
                if candidate is None:
                    continue
                heard = stt.transcribe(candidate, prompt=wake_prompt)
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
                utterance = turn_detector.process_chunk(chunk)
                if utterance is None:
                    continue

                text = stt.transcribe(utterance)
                if not text:
                    continue
            print(f"You said: {text}")

            reply = chat.ask(text)
            print(f"Assistant: {reply}")

            wav_bytes = tts.synthesize(reply)
            if speak_with_barge_in(
                wav_bytes,
                mic,
                turn_detector,
                barge_in_prob_threshold,
                barge_in_consecutive_frames,
                echo_canceller,
                grace_period_chunks,
            ):
                print("(interrupted)")

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
