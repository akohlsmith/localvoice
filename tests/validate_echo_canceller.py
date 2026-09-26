"""Validates audio/echo_cancel.py's EchoCanceller against real captured
echo, using the metric that actually matters: how long a sustained
high-confidence VAD run survives after cancellation, not just an RMS
number.

Problem this solves: speak_with_barge_in only false-triggers when the
mic's VAD-scored signal sustains a high probability for
barge_in.consecutive_frames in a row. A canceller that reduces overall
RMS but leaves sustained high-confidence runs intact hasn't actually
fixed anything. This script measures the real thing:

  1. Captures a live ref/mic pair via MicStream (see measure_echo_delay.py
     for why MicStream specifically, not sd.rec()).
  2. Scores the raw captured audio with Silero VAD, chunk by chunk, and
     finds the longest run of consecutive chunks >= 0.9 confidence.
  3. Runs the SAME captured mic audio through EchoCanceller (using the
     project's real config.yaml settings, streaming chunk-by-chunk
     exactly as speak_with_barge_in does -- not a batch/offline shortcut)
     and finds the longest run of consecutive residual chunks >= 0.9.
  4. Reports both, plus per-chunk processing time (must stay well under
     32ms/chunk -- the real-time budget at 16kHz/512-sample chunks -- or
     the canceller itself would cause the backlog bug measure_echo_delay
     and tests/test_mic_queue_drain.py are about).

Historical result on this machine (a ~7s clip): raw mic sustained
>=0.9 confidence for 3264ms straight (an unconditional false trigger at
any reasonable consecutive_frames setting); after cancellation, that
dropped to 352ms, with per-chunk processing at ~1.6ms (20x margin under
budget). This is also how the EchoCanceller's indexing bugs were caught
during development -- a first, subtly wrong window-alignment
implementation showed literally no improvement here (longest run
barely changed), which is what revealed the bug rather than a code
review catching it.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/validate_echo_canceller.py
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio import playback  # noqa: E402
from audio.capture import MicStream  # noqa: E402
from audio.echo_cancel import EchoCanceller  # noqa: E402
from audio.vad import SileroVAD  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = main.CHUNK_SAMPLES


def longest_run_ge(probs, threshold):
    best = 0
    run = 0
    for p in probs:
        if p >= threshold:
            run += 1
            best = max(best, run)
        else:
            run = 0
    return best


def capture_ref_and_mic(wav_bytes, expected_duration_s):
    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    if ref_sr != SAMPLE_RATE:
        sys.exit(f"Voice's native rate ({ref_sr}Hz) != {SAMPLE_RATE}Hz; resample before use.")

    mic = MicStream(sample_rate=SAMPLE_RATE, chunk_samples=CHUNK, channels=1, device=None)
    mic.start()
    time.sleep(0.3)
    mic.drain()

    playback.play_wav_bytes_async(wav_bytes)
    collected = []
    n_chunks_needed = int((expected_duration_s + 1.0) * SAMPLE_RATE / CHUNK)
    gen = mic.chunks()
    for _ in range(n_chunks_needed):
        collected.append(next(gen))
    mic.stop()
    mic_signal = np.concatenate(collected)
    return raw_ref, mic_signal


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--text", default=(
        "This is a somewhat longer sentence, spoken so we can measure "
        "how well the echo canceller holds up against real echo picked "
        "up by the real microphone while it plays through the real speakers."
    ))
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(args.text)
    raw_ref, _ = playback.decode_wav_bytes(wav_bytes)
    duration = len(raw_ref) / SAMPLE_RATE
    print(f"Synthesized {duration:.2f}s clip. Capturing live ref/mic pair...")

    ref, mic_signal = capture_ref_and_mic(wav_bytes, duration)
    n = min(len(ref), len(mic_signal))
    ref, mic_signal = ref[:n], mic_signal[:n]
    n_chunks = n // CHUNK

    echo_cfg = cfg["barge_in"]["echo_cancel"]
    ec = EchoCanceller(
        sample_rate=SAMPLE_RATE,
        expected_delay_ms=echo_cfg["expected_delay_ms"],
        filter_length_ms=echo_cfg["filter_length_ms"],
        mu=echo_cfg["mu"],
    )

    residual = np.zeros(n_chunks * CHUNK, dtype=np.float32)
    t0 = time.perf_counter()
    for i in range(n_chunks):
        s, e = i * CHUNK, (i + 1) * CHUNK
        ec.push_reference(ref[s:e])
        residual[s:e] = ec.cancel(mic_signal[s:e])
    elapsed = time.perf_counter() - t0
    per_chunk_ms = elapsed / n_chunks * 1000
    budget_ms = 1000 * CHUNK / SAMPLE_RATE

    vad = SileroVAD(sample_rate=SAMPLE_RATE)

    def analyze(sig, label):
        probs = [vad.speech_prob(sig[i * CHUNK : (i + 1) * CHUNK]) for i in range(n_chunks)]
        run = longest_run_ge(probs, 0.9)
        print(f"{label}: max={max(probs):.3f} mean={sum(probs)/len(probs):.3f} "
              f"longest_run_ge_0.9={run} ({run * 32}ms)")

    print()
    analyze(mic_signal[: n_chunks * CHUNK], "raw mic (with echo)")
    analyze(residual, "residual (after echo cancellation)")
    print()
    print(f"processing time: {per_chunk_ms:.2f} ms/chunk (real-time budget: {budget_ms:.0f}ms)")
    if per_chunk_ms > budget_ms:
        print("WARNING: over budget -- this would itself cause the mic queue backlog")
        print("that tests/test_mic_queue_drain.py is about.")


if __name__ == "__main__":
    main_cli()
