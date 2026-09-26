"""Play back WAV audio through the default output device.

Kept deliberately simple for v1 (blocks until playback finishes). Barge-in
(stopping playback if the user starts talking again) is a phase-7 item —
sd.stop() is the hook to call from a VAD check running on another thread
once that's added.
"""

import io
import wave

import numpy as np
import sounddevice as sd

_DTYPE_BY_SAMPWIDTH = {1: np.uint8, 2: np.int16, 4: np.int32}


def play_wav_bytes(wav_bytes: bytes):
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = _DTYPE_BY_SAMPWIDTH.get(sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    audio = np.frombuffer(frames, dtype=dtype)
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels)

    sd.play(audio, samplerate=sample_rate)
    sd.wait()


def play_wav_file(path: str):
    with open(path, "rb") as f:
        play_wav_bytes(f.read())


def stop():
    """Interrupt any playback currently in progress."""
    sd.stop()
