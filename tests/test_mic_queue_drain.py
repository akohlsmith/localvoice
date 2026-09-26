"""Regression test for the mic-queue-backlog bug in speak_with_barge_in,
fixed by MicStream.drain().

Problem this solved: between capturing the user's utterance and calling
speak_with_barge_in, STT + the LLM call + TTS synthesis all block for
real time. MicStream's callback keeps queuing chunks throughout --
nobody's consuming them during that gap, so the queue backlogs. The
first chunks pulled afterward were then a burst of stale, already-past
audio consumed almost instantly (already sitting in the queue, no real
wait), which desynced both the echo canceller's reference alignment
(assumes one chunk per loop iteration tracks real elapsed time) and the
grace period (counted in chunks, so it elapsed in a fraction of a real
second instead of grace_period_ms). This was reported live (a real
wake-word -> arithmetic-question exchange self-interrupted), then
reproduced directly: simulating a 3.5s "thinking time" gap before
speak_with_barge_in queued exactly 109 backlogged chunks and caused a
false trigger at 1.93s elapsed -- well inside what should have been the
grace period.

Fixed with MicStream.drain(), called at the very start of
speak_with_barge_in, right before playback starts -- so monitoring
always begins from a clean "now" regardless of how long the preceding
turn took to think.

This script re-creates that exact scenario (and a spread of other
backlog sizes) against the real, current speak_with_barge_in, and
reports PASS/FAIL against the simple expectation: a reply with no real
interruption should never self-trigger, no matter how long the gap
before it was called.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/test_mic_queue_drain.py
"""

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio.capture import MicStream  # noqa: E402
from audio.echo_cancel import EchoCanceller  # noqa: E402
from audio.vad import SileroVAD, TurnDetector  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = main.CHUNK_SAMPLES

# The exact gap that reproduced the bug historically, plus a spread of
# others to check the fix isn't sensitive to the specific delay.
THINKING_TIME_DELAYS_S = [0.05, 0.5, 1.5, 2.0, 3.5, 5.0]


def run_trial(wav_bytes, thinking_time_s, cfg):
    barge_in_chunk_ms = 1000 * CHUNK / SAMPLE_RATE
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / barge_in_chunk_ms)

    mic = MicStream(sample_rate=SAMPLE_RATE, chunk_samples=CHUNK, channels=1, device=None)
    vad = SileroVAD(sample_rate=SAMPLE_RATE)
    turn_detector = TurnDetector(
        vad,
        sample_rate=SAMPLE_RATE,
        chunk_samples=CHUNK,
        prob_threshold=cfg["vad"]["speech_prob_threshold"],
        silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
        min_speech_ms=cfg["vad"]["min_speech_ms"],
    )
    echo_cfg = cfg["barge_in"]["echo_cancel"]
    ec = EchoCanceller(
        sample_rate=SAMPLE_RATE,
        expected_delay_ms=echo_cfg["expected_delay_ms"],
        filter_length_ms=echo_cfg["filter_length_ms"],
        mu=echo_cfg["mu"],
    )

    mic.start()
    # Simulates the STT + LLM + TTS-synthesis "thinking time" gap during
    # which nobody drains the mic queue -- the exact precondition for the
    # bug, before speak_with_barge_in's internal drain() call runs.
    time.sleep(thinking_time_s)
    backlog_size = mic._q.qsize()

    t0 = time.time()
    interrupted = main.speak_with_barge_in(
        wav_bytes,
        mic,
        turn_detector,
        cfg["barge_in"]["prob_threshold"],
        cfg["barge_in"]["consecutive_frames"],
        ec,
        grace_period_chunks,
        None,
    )
    elapsed = time.time() - t0
    mic.stop()
    return backlog_size, interrupted, elapsed


def main_cli():
    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize("The sum of all numbers from 1 to 100 is 5,050.")

    results = []
    for delay in THINKING_TIME_DELAYS_S:
        backlog_size, interrupted, elapsed = run_trial(wav_bytes, delay, cfg)
        status = "FAIL (false-triggered)" if interrupted else "PASS"
        print(f"thinking_time={delay:>4.2f}s backlog={backlog_size:>3d} chunks -> {status} "
              f"(elapsed={elapsed:.2f}s)")
        results.append(not interrupted)

    print()
    print(f"{sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main_cli()
