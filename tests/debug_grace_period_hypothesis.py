"""One-off: tests the hypothesis that main.speak_with_barge_in's grace
period bug is skipping calls to turn_detector.vad.speech_prob()
entirely for chunks inside grace_period_chunks (main.py line ~123-126:
the `continue` happens BEFORE the speech_prob() call, not after).

Silero VAD is a streaming model with internal recurrent state carried
between calls. Every diagnostic/tuning script used so far
(tune_barge_in_thresholds.py, validate_true_barge_in.py) calls
speech_prob() on EVERY chunk unconditionally and only *uses* the grace
flag afterward to decide whether to act -- so unlike production, they
keep the model "warmed up" through the grace period. Hypothesis: real
production code's near-zero probabilities right after grace period
ends (confirmed via debug_real_barge_in_trace.py: max prob 0.81-0.95
but almost no chunks even clearing 0.5) is caused by the model getting
its very first calls only *after* a ~2s gap of never being invoked at
all, rather than by anything about the real interruption itself.

This script duplicates speak_with_barge_in's loop but calls
speech_prob() on every chunk (matching the diagnostic scripts, not
production) while still only *acting* on the trigger after grace
period -- if this alone restores strong, sustained high-confidence
detection where the real production function got near-silence, that
confirms the bug and the fix (call speech_prob() every chunk; use the
grace flag only to gate the action, not the call).
"""

import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio import playback  # noqa: E402
from audio.capture import MicStream  # noqa: E402
from audio.echo_cancel import EchoCanceller, resample_linear  # noqa: E402
from audio.vad import SileroVAD  # noqa: E402
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
    ec = EchoCanceller(
        sample_rate=sample_rate,
        expected_delay_ms=echo_cfg["expected_delay_ms"],
        filter_length_ms=echo_cfg["filter_length_ms"],
        mu=echo_cfg["mu"],
    )

    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    ref_audio = resample_linear(raw_ref, ref_sr, sample_rate)
    ec.reset()

    mic.start()
    time.sleep(0.3)
    mic.drain()
    playback.play_wav_bytes_async(wav_bytes)

    speak_at_chunk = grace_period_chunks + int(1000 * 1.0 / chunk_ms)
    print(f"grace_period_chunks={grace_period_chunks}  prob_threshold={base_prob}  consecutive_frames={base_frames}")
    print(f"playing reply -- speak in ~{speak_at_chunk * chunk_ms / 1000:.1f}s...", flush=True)

    trace = []
    ref_pos = 0
    chunk_index = 0
    consecutive_speech = 0
    prompted = False
    triggered_at = None
    t0 = time.time()
    for mic_chunk in mic.chunks():
        if not playback.is_playing():
            break
        n = len(mic_chunk)
        ref_slice = ref_audio[ref_pos: ref_pos + n]
        if len(ref_slice) < n:
            ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
        ref_pos += n
        ec.push_reference(ref_slice)
        scored_chunk = ec.cancel(mic_chunk)

        # KEY DIFFERENCE from main.speak_with_barge_in: call speech_prob()
        # on every chunk (matching the diagnostic scripts), and only use
        # in_grace_period to gate the *action*, not the call itself.
        prob = vad.speech_prob(scored_chunk)
        trace.append(float(prob))

        if not prompted and chunk_index >= speak_at_chunk:
            print("  >>> SPEAK NOW (talk continuously for a few seconds) <<<", flush=True)
            prompted = True

        in_grace_period = chunk_index < grace_period_chunks
        chunk_index += 1

        if in_grace_period:
            continue

        if prob >= base_prob:
            consecutive_speech += 1
            if consecutive_speech >= base_frames and triggered_at is None:
                triggered_at = chunk_index
                print(f"  *** WOULD TRIGGER at chunk {chunk_index} (~{chunk_index * chunk_ms / 1000:.1f}s) ***",
                      flush=True)
        else:
            consecutive_speech = 0

    playback.stop()
    mic.stop()
    elapsed = time.time() - t0
    print(f"done after {elapsed:.2f}s, {len(trace)} chunks scored")
    print(f"triggered: {triggered_at is not None}")
    print("max prob overall:", max(trace) if trace else None)
    print("chunks with prob>=0.93:", sum(1 for p in trace if p >= 0.93))
    print("chunks with prob>=0.5:", sum(1 for p in trace if p >= 0.5))


if __name__ == "__main__":
    main_cli()
