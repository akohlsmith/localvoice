"""Measures the real speaker-to-mic echo delay and coupling strength on
this machine, using the actual MicStream/playback classes -- not a
generic recording tool -- since that distinction turned out to matter.

Problem this solves: audio/echo_cancel.py needs to know roughly how far
back (in time) to look in the reference signal to find the echo. The
first attempt at measuring this used sounddevice's sd.rec() convenience
function to record while sd.play() played a clip, and found almost no
coupling at all (peak amplitude ~0.003, nowhere near enough to explain
speak_with_barge_in self-interrupting every time in practice). Redoing
the same measurement with the project's own MicStream class (a raw,
low-latency callback-driven InputStream, the way main.py actually
captures audio) found the opposite: peak amplitude ~0.2, and Silero VAD
scoring the raw captured audio at ~1.0 confidence within ~500ms and
holding there for seconds. The two capture methods are not
interchangeable for this purpose -- likely because a "convenience"
higher-latency stream interacts differently with the mic's own
AGC/noise-suppression firmware than a raw low-latency one does, though
that's a hypothesis, not confirmed. The lesson: always measure with the
real capture path, not a quick substitute.

This script does the MicStream-based measurement: plays a clip through
the speakers while simultaneously recording from the mic via MicStream,
then cross-correlates the two signals to report:
  - peak amplitude / RMS of what the mic actually picked up
  - the estimated delay (in ms) via cross-correlation
  - the normalized correlation coefficient at that delay (how confident
    the "delay" number actually reflects real acoustic coupling, vs
    coincidental noise correlation)
  - delay stability across the first half vs second half of the clip

Historical result on this machine: ~410ms delay, stable within a single
clip (first/second half within ~8ms of each other) -- but NOT stable
run to run, discovered only when audio/echo_cancel.py's first version
(anchored tightly to this one measurement, ±150ms window) worked
perfectly replayed against its own calibration recording but failed
live in a fresh session. That's why echo_cancel.py uses a wide
(50-650ms) window rather than trusting a single measurement like this
one precisely -- but re-running this script is still the right first
step when tuning for a different machine, to know the right ballpark.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/measure_echo_delay.py
    venv/bin/python tests/measure_echo_delay.py --duration 8
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
from audio.vad import SileroVAD  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

SAMPLE_RATE = 16000
CHUNK = main.CHUNK_SAMPLES


def record_with_micstream(wav_bytes, expected_duration_s):
    """Plays wav_bytes while recording from the mic via MicStream (the
    project's real capture path), returning the captured samples."""
    mic = MicStream(sample_rate=SAMPLE_RATE, chunk_samples=CHUNK, channels=1, device=None)
    mic.start()
    time.sleep(0.3)
    mic.drain()

    playback.play_wav_bytes_async(wav_bytes)
    collected = []
    n_chunks_needed = int((expected_duration_s + 1.5) * SAMPLE_RATE / CHUNK)
    gen = mic.chunks()
    for _ in range(n_chunks_needed):
        collected.append(next(gen))
    mic.stop()
    return np.concatenate(collected)


def cross_correlate_delay(ref, mic, sample_rate, max_lag_s=0.6):
    n = min(len(ref), len(mic))
    ref = ref[:n].astype(np.float64)
    mic = mic[:n].astype(np.float64)
    max_lag = int(max_lag_s * sample_rate)

    corr = np.correlate(mic, ref, mode="full")
    lags = np.arange(-n + 1, n)
    mask = (lags >= 0) & (lags <= max_lag)
    sub_corr = corr[mask]
    sub_lags = lags[mask]
    best_idx = np.argmax(np.abs(sub_corr))
    best_lag = sub_lags[best_idx]

    shifted_ref = np.roll(ref, best_lag)
    shifted_ref[:best_lag] = 0
    valid = slice(best_lag, n)
    r = np.corrcoef(mic[valid], shifted_ref[valid])[0, 1]

    peak_over_std = abs(sub_corr[best_idx]) / sub_corr.std()
    return best_lag, best_lag / sample_rate * 1000, r, peak_over_std


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--duration", type=float, default=6.0, help="Approx seconds of speech to synthesize")
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    # Padded out with filler so the clip runs roughly --duration seconds
    # regardless of voice speaking rate.
    text = (
        "Testing testing one two three, measuring the echo delay between "
        "the speaker and the microphone on this machine. " * max(1, int(args.duration / 6))
    )
    wav_bytes = tts.synthesize(text)
    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    ref = raw_ref if ref_sr == SAMPLE_RATE else None
    if ref is None:
        sys.exit(f"Voice's native rate ({ref_sr}Hz) != {SAMPLE_RATE}Hz; resample ref before correlating.")
    actual_duration = len(ref) / SAMPLE_RATE

    print(f"Synthesized {actual_duration:.2f}s clip. Recording via MicStream while it plays...")
    mic_signal = record_with_micstream(wav_bytes, actual_duration)
    print(f"Captured {len(mic_signal) / SAMPLE_RATE:.2f}s, peak amplitude={np.abs(mic_signal).max():.4f}, "
          f"RMS={np.sqrt(np.mean(mic_signal.astype(np.float64) ** 2)):.4f}")

    lag_samples, lag_ms, corr_coef, confidence = cross_correlate_delay(ref, mic_signal, SAMPLE_RATE)
    print(f"Estimated delay: {lag_samples} samples = {lag_ms:.1f}ms")
    print(f"Normalized correlation coefficient at that delay: {corr_coef:.3f}")
    print(f"Peak/std confidence ratio: {confidence:.2f} (higher = more confidently real, not noise)")

    n = min(len(ref), len(mic_signal))
    half = n // 2
    for label, sl in [("first half", slice(0, half)), ("second half", slice(half, n))]:
        lag_s, lag_s_ms, _, _ = cross_correlate_delay(ref[sl], mic_signal[sl], SAMPLE_RATE)
        print(f"  {label} delay: {lag_s} samples = {lag_s_ms:.1f}ms")

    print()
    print("Also worth sanity-checking: does Silero VAD score the raw captured")
    print("audio as speech at all? (If not, something about capture is off --")
    print("compare against the ~1.0 confidence historically measured here.)")
    vad = SileroVAD(sample_rate=SAMPLE_RATE)
    n_chunks = len(mic_signal) // CHUNK
    probs = [vad.speech_prob(mic_signal[i * CHUNK : (i + 1) * CHUNK]) for i in range(n_chunks)]
    print(f"max VAD probability on raw captured audio: {max(probs):.3f}")


if __name__ == "__main__":
    main_cli()
