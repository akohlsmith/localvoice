"""Confirms whether main.speak_with_barge_in -- the actual production
function, not a trace/measurement stand-in -- really stops playback
when a real person genuinely interrupts it.

Problem this solves: tests/validate_true_barge_in.py measures whether
real speech *would* clear barge_in.prob_threshold/consecutive_frames
by logging VAD probabilities without ever calling playback.stop() --
it's an analysis tool, not a demonstration of the real stopping
behavior, and running it can look exactly like a live barge-in failure
even when nothing is wrong (this confused a report of "barge-in isn't
working" that turned out to be this distinction, not a real bug).

This script instead calls main.speak_with_barge_in() directly --
the exact function main.py's loop uses -- against a long reply, and
you try to interrupt it for real, at different points across several
trials (early, middle, late), the way you'd actually use it. Needs a
real person reacting live: run it in a terminal you're watching.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/confirm_real_barge_in.py
    venv/bin/python tests/confirm_real_barge_in.py --trials 5
"""

import argparse
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

# Long enough (~25-30s) that "early/middle/late" are meaningfully
# different points, and repeated interrupt attempts are plausible
# within a single reply.
TEST_TEXT = (
    "Let's talk through this slowly, one step at a time, so there is "
    "plenty of room to interrupt whenever you want to. First, imagine "
    "a long, rambling explanation that just keeps going, sentence "
    "after sentence, the way a real answer sometimes does when there "
    "is a lot of ground to cover. Second, notice that nothing "
    "important is being said here at all, it is just filler text, "
    "existing only to give you enough time to try cutting in near the "
    "beginning, then again somewhere in the middle, and then again "
    "near the very end, across several separate attempts. Third, and "
    "finally, if this sentence is still playing when you try to "
    "interrupt, that means there was still time left, which is "
    "exactly the point of making the text this long in the first "
    "place."
)


def confirm(n_trials, when_label):
    cfg = main.load_config()
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]

    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(TEST_TEXT)

    results = []
    for trial in range(n_trials):
        mic = MicStream(sample_rate=sample_rate, chunk_samples=chunk, channels=1, device=None)
        vad = SileroVAD(sample_rate=sample_rate)
        turn_detector = TurnDetector(
            vad,
            sample_rate=sample_rate,
            chunk_samples=chunk,
            prob_threshold=cfg["vad"]["speech_prob_threshold"],
            silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
            min_speech_ms=cfg["vad"]["min_speech_ms"],
        )
        ec = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg["expected_delay_ms"],
            filter_length_ms=echo_cfg["filter_length_ms"],
            mu=echo_cfg["mu"],
        )
        mic.start()
        time.sleep(0.3)
        print(f"trial {trial + 1}/{n_trials}: playing reply -- interrupt it {when_label}", flush=True)
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
        status = "STOPPED (barge-in fired)" if interrupted else "played to completion (no barge-in)"
        print(f"  -> {status} after {elapsed:.2f}s", flush=True)
        results.append(interrupted)
        time.sleep(1.0)

    return results


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=3)
    args = parser.parse_args()

    print("=== round 1: interrupt EARLY (just after playback clearly starts) ===")
    early = confirm(args.trials, "EARLY")

    print()
    print("=== round 2: interrupt in the MIDDLE ===")
    middle = confirm(args.trials, "in the MIDDLE")

    print()
    print("=== round 3: interrupt LATE (near the end) ===")
    late = confirm(args.trials, "LATE")

    print()
    print("=== summary ===")
    print(f"early:  {sum(early)}/{len(early)} stopped")
    print(f"middle: {sum(middle)}/{len(middle)} stopped")
    print(f"late:   {sum(late)}/{len(late)} stopped")


if __name__ == "__main__":
    main_cli()
