"""One-off instrumentation: calls the REAL main.speak_with_barge_in
unmodified, but wraps turn_detector.vad.speech_prob to log every
chunk's probability as it's actually computed inside that function --
not a re-implementation, the literal production code path -- so a
live "no barge-in" report can be diagnosed against real per-chunk
data instead of guessing. Prints a compact summary (longest run over
several thresholds, and where in the timeline it happened) rather than
every chunk, since a long reply is 1000+ chunks.

Not meant to stay useful long-term the way the other tests/ scripts
are -- delete once the underlying bug (if any) is found and fixed, or
keep it if it's still handy for future barge-in regressions.
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
TEST_TEXT = (
    "Let's talk through this slowly, one step at a time, so there is "
    "plenty of room to interrupt whenever you want to. First, imagine "
    "a long, rambling explanation that just keeps going, sentence "
    "after sentence, the way a real answer sometimes does when there "
    "is a lot of ground to cover."
)


def main_cli():
    cfg = main.load_config()
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]
    base_prob = cfg["barge_in"]["prob_threshold"]
    base_frames = cfg["barge_in"]["consecutive_frames"]

    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(TEST_TEXT)

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

    trace = []
    call_count = [0]
    speak_at_chunk = grace_period_chunks + int(1000 * 1.0 / chunk_ms)
    prompted = [False]
    real_speech_prob = vad.speech_prob

    def logging_speech_prob(chunk_arr):
        prob = real_speech_prob(chunk_arr)
        trace.append(float(prob))
        if not prompted[0] and call_count[0] >= speak_at_chunk:
            print("  >>> SPEAK NOW (talk continuously for a few seconds) <<<", flush=True)
            prompted[0] = True
        call_count[0] += 1
        return prob

    vad.speech_prob = logging_speech_prob

    mic.start()
    time.sleep(0.3)
    print(f"grace_period_chunks={grace_period_chunks}  prob_threshold={base_prob}  consecutive_frames={base_frames}")
    print(f"playing reply -- speak in ~{grace_period_chunks * chunk_ms / 1000 + 1.0:.1f}s...", flush=True)
    t0 = time.time()
    interrupted = main.speak_with_barge_in(
        wav_bytes, mic, turn_detector, base_prob, base_frames, ec, grace_period_chunks, None,
    )
    elapsed = time.time() - t0
    mic.stop()

    print(f"result: {'STOPPED' if interrupted else 'played to completion'} after {elapsed:.2f}s")
    print(f"total chunks scored: {len(trace)}")

    for prob_threshold in [0.8, 0.85, 0.9, 0.93, 0.95]:
        best = 0
        cur = 0
        best_at = 0
        for i, p in enumerate(trace):
            if i < grace_period_chunks:
                cur = 0
                continue
            if p >= prob_threshold:
                cur += 1
                if cur > best:
                    best = cur
                    best_at = i
            else:
                cur = 0
        print(f"prob>={prob_threshold}: longest run={best} frames (~{best*32}ms), ending at chunk {best_at} "
              f"(~{best_at*32/1000:.1f}s in)")

    print()
    print("max prob overall:", max(trace) if trace else None)
    print("chunks with prob>=0.93:", sum(1 for p in trace if p >= 0.93))
    print("chunks with prob>=0.5:", sum(1 for p in trace if p >= 0.5))


if __name__ == "__main__":
    main_cli()
