"""Play back WAV audio through the default output device.

decode_wav_bytes is used both by play_wav_bytes_async below and by
main.py's speak_with_barge_in, which needs the same audio as a
normalized float32 array both as the echo canceller's reference signal
and as what it hands to MicStream.play() (audio/capture.py) for actual
playback.

play_wav_bytes_async/is_playing/stop (this module's own separate
output stream) are for standalone playback with no concurrent mic
capture -- a one-off script, a notification sound, etc. Playback that
happens *while* the mic is also being monitored (barge-in) goes
through MicStream.play() instead, sharing one full-duplex stream with
capture rather than opening a second, independently clocked one:
running mic capture and this module's playback as two separate streams
at the same time measurably caused intermittent crackling in the TTS
output (see PLAN.md's barge-in section and tests/measure_output_underruns.py
/ tests/compare_samplerate_crackle.py), even though neither stream ever
reported a buffer underrun -- a single duplex stream removes the
two-clock-domain path by construction.
"""

import io
import wave

import numpy as np
import sounddevice as sd

_DTYPE_BY_SAMPWIDTH = {1: np.uint8, 2: np.int16, 4: np.int32}
_MAX_BY_SAMPWIDTH = {1: 128.0, 2: 32768.0, 4: 2147483648.0}


def decode_wav_bytes(wav_bytes: bytes):
    """Returns (audio, sample_rate): audio is float32, normalized to
    [-1, 1], mono (channels averaged down if the WAV has more than one)."""
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sample_rate = wf.getframerate()
        n_channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        frames = wf.readframes(wf.getnframes())

    dtype = _DTYPE_BY_SAMPWIDTH.get(sampwidth)
    if dtype is None:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    audio = np.frombuffer(frames, dtype=dtype).astype(np.float32) / _MAX_BY_SAMPWIDTH[sampwidth]
    if n_channels > 1:
        audio = audio.reshape(-1, n_channels).mean(axis=1)

    return audio, sample_rate


def play_wav_bytes_async(wav_bytes: bytes):
    """Starts playback without blocking. Use is_playing()/stop() to control it."""
    audio, sample_rate = decode_wav_bytes(wav_bytes)
    sd.play(audio, samplerate=sample_rate)


def is_playing() -> bool:
    stream = sd.get_stream()
    return stream is not None and stream.active


def play_wav_bytes(wav_bytes: bytes):
    """Blocking playback, for callers that don't need barge-in (e.g. one-off tests)."""
    play_wav_bytes_async(wav_bytes)
    sd.wait()


def play_wav_file(path: str):
    with open(path, "rb") as f:
        play_wav_bytes(f.read())


def stop():
    """Interrupt any playback currently in progress."""
    sd.stop()
