"""Checks whether the reported TTS crackling is a real-time performance
problem: does the barge-in monitoring loop (echo cancellation + VAD,
running in the same Python process/thread as playback) fall behind
its per-chunk budget often enough to starve the audio output stream?

Problem this diagnoses: speak_with_barge_in's per-chunk work (NLMS
echo cancellation -- a Python loop over each of the 512 samples with a
~9600-sample dot product per sample -- plus a Silero VAD inference)
all happens on the main thread while sounddevice's input and output
streams each run their own PortAudio-driven callback thread. Under
CPython, those callbacks still need the GIL to run their Python-level
code; if the main thread holds it too long per chunk (budget: 512
samples / 16000 Hz = 32ms), the output stream's callback can be
delayed past its own deadline, which PortAudio reports as an
"output underflow" -- audibly, a click/crackle.

Unlike sd.play() (used by audio/playback.py, which hides underrun
status from the caller), this script opens its own sd.OutputStream
with an explicit callback so it can inspect the `status` argument
sounddevice/PortAudio provides on every callback and count real
underflows directly, while running the exact same echo-cancel + VAD
workload speak_with_barge_in does, concurrently, on the main thread --
so a real underrun shows up as a number, not a guess from listening.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/measure_output_underruns.py
    venv/bin/python tests/measure_output_underruns.py --trials 3
"""

import argparse
import os
import sys
import time

import numpy as np
import sounddevice as sd

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


class ManualPlayer:
    """Like playback.play_wav_bytes_async, but with our own callback so
    we can see PortAudio's underflow status instead of it being hidden
    inside sd.play()."""

    def __init__(self, audio: np.ndarray, sample_rate: int):
        self.audio = audio
        self.sample_rate = sample_rate
        self.pos = 0
        self.underflow_count = 0
        self.callback_count = 0
        self.max_callback_gap = 0.0
        self._last_callback_time = None
        self._stream = sd.OutputStream(
            samplerate=sample_rate,
            channels=1,
            dtype="float32",
            callback=self._callback,
            finished_callback=self._finished,
        )
        self._done = False

    def _callback(self, outdata, frames, time_info, status):
        now = time.time()
        if self._last_callback_time is not None:
            gap = now - self._last_callback_time
            self.max_callback_gap = max(self.max_callback_gap, gap)
        self._last_callback_time = now
        self.callback_count += 1

        if status:
            self.underflow_count += 1

        remaining = len(self.audio) - self.pos
        n = min(frames, remaining)
        if n > 0:
            outdata[:n, 0] = self.audio[self.pos: self.pos + n]
            self.pos += n
        if n < frames:
            outdata[n:, 0] = 0.0
        if remaining <= frames:
            raise sd.CallbackStop()

    def _finished(self):
        self._done = True

    def start(self):
        self._stream.start()

    def is_playing(self) -> bool:
        return not self._done

    def stop(self):
        self._stream.stop()
        self._stream.close()


def run_trial(wav_bytes, cfg, trial_num):
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]

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

    player = ManualPlayer(ref_audio, sample_rate)

    mic.start()
    time.sleep(0.3)
    mic.drain()
    player.start()

    print(f"trial {trial_num}: playing + monitoring (echo cancel + VAD) concurrently...", flush=True)

    loop_times_ms = []
    ref_pos = 0
    chunk_index = 0
    for mic_chunk in mic.chunks():
        if not player.is_playing():
            break
        t0 = time.perf_counter()

        n = len(mic_chunk)
        ref_slice = ref_audio[ref_pos: ref_pos + n]
        if len(ref_slice) < n:
            ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
        ref_pos += n
        ec.push_reference(ref_slice)
        scored_chunk = ec.cancel(mic_chunk)

        in_grace = chunk_index < grace_period_chunks
        chunk_index += 1
        prob = vad.speech_prob(scored_chunk)  # always scored, per the recent fix
        _ = prob

        loop_times_ms.append((time.perf_counter() - t0) * 1000)

    mic.stop()
    player.stop()

    budget_ms = chunk_ms
    over_budget = [t for t in loop_times_ms if t > budget_ms]
    print(f"  chunks processed: {len(loop_times_ms)}")
    print(f"  loop body time (ms): max={max(loop_times_ms):.2f} "
          f"p99={np.percentile(loop_times_ms, 99):.2f} "
          f"p50={np.percentile(loop_times_ms, 50):.2f}  (budget={budget_ms:.1f}ms/chunk)")
    print(f"  chunks over budget: {len(over_budget)}/{len(loop_times_ms)}")
    print(f"  output stream: callbacks={player.callback_count}  "
          f"underflow_flags={player.underflow_count}  "
          f"max_gap_between_callbacks={player.max_callback_gap * 1000:.1f}ms")
    return player.underflow_count, over_budget


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=3)
    parser.add_argument("--text", default=TEST_TEXT)
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(args.text)

    total_underflows = 0
    for trial in range(args.trials):
        underflows, _ = run_trial(wav_bytes, cfg, trial)
        total_underflows += underflows
        time.sleep(1.0)

    print()
    print(f"total underflow flags across {args.trials} trials: {total_underflows}")


if __name__ == "__main__":
    main_cli()
