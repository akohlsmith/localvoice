"""Compares playback at Piper's native rate (16kHz, what audio/playback.py
does today) against the same audio resampled to the output device's own
native rate, to check whether a sample-rate mismatch is the source of
reported "random crackling" in TTS output.

Problem this diagnoses: Piper's voice models here synthesize at 16kHz,
but sd.play() (audio/playback.py) opens the output stream directly at
that rate regardless of what the actual output device's native rate
is. On this machine the default output device (MacBook Pro Speakers)
reports default_samplerate=48000. Asking CoreAudio/PortAudio to run a
device at a non-native rate means either the OS reconfigures the
hardware's clock for every utterance, or a resampler runs in the
audio path -- both are common, well-known sources of intermittent
clicks/pops on macOS, and would look exactly like "random crackling"
that isn't tied to CPU load (tests/measure_output_underruns.py already
ruled out a Python/GIL performance problem: 0 underflow flags, ~7ms
p50 loop time against a 32ms budget).

This plays the same reply twice back to back: once at Piper's native
16kHz (today's behavior), once resampled to the device's own native
rate first (today's fix candidate). Listen to both and compare --
underflow_flags from sounddevice are also reported directly for each,
though a real device-level rate-switch glitch may not show up as a
PortAudio-reported underflow at all (it can happen below that layer),
so your ears are the real test here, not just the counters.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/compare_samplerate_crackle.py
"""

import os
import sys
import time

import numpy as np
import sounddevice as sd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio import playback  # noqa: E402
from audio.echo_cancel import resample_linear  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

TEST_TEXT = (
    "Let's talk through this slowly, one step at a time, so there is "
    "plenty of room to interrupt whenever you want to. First, imagine "
    "a long, rambling explanation that just keeps going, sentence "
    "after sentence, the way a real answer sometimes does when there "
    "is a lot of ground to cover."
)


def play_and_count_underflows(audio: np.ndarray, sample_rate: int, label: str):
    underflow_count = [0]
    callback_count = [0]
    pos = [0]

    def callback(outdata, frames, time_info, status):
        if status:
            underflow_count[0] += 1
        callback_count[0] += 1
        remaining = len(audio) - pos[0]
        n = min(frames, remaining)
        if n > 0:
            outdata[:n, 0] = audio[pos[0]: pos[0] + n]
            pos[0] += n
        if n < frames:
            outdata[n:, 0] = 0.0
        if remaining <= frames:
            raise sd.CallbackStop()

    print(f"--- playing at {sample_rate} Hz ({label}) -- listen for crackling ---", flush=True)
    stream = sd.OutputStream(samplerate=sample_rate, channels=1, dtype="float32", callback=callback)
    stream.start()
    while stream.active:
        time.sleep(0.05)
    stream.stop()
    stream.close()
    print(f"  underflow_flags={underflow_count[0]}  callbacks={callback_count[0]}")


def main_cli():
    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(TEST_TEXT)
    audio, native_rate = playback.decode_wav_bytes(wav_bytes)
    print(f"Piper synthesized at {native_rate} Hz, {len(audio)/native_rate:.2f}s")

    device_info = sd.query_devices(kind="output")
    device_rate = int(device_info["default_samplerate"])
    print(f"Output device '{device_info['name']}' native rate: {device_rate} Hz")
    print()

    play_and_count_underflows(audio, native_rate, "Piper's native rate, today's behavior")
    time.sleep(1.0)

    if device_rate != native_rate:
        resampled = resample_linear(audio, native_rate, device_rate)
        play_and_count_underflows(resampled, device_rate, "resampled to device's native rate")
    else:
        print("(device rate already matches Piper's rate -- nothing to compare)")


if __name__ == "__main__":
    main_cli()
