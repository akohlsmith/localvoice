"""Client for a running whisper.cpp `server` instance.

This assumes the server binary from whisper.cpp is already running
separately with a model loaded (see README.md) — this client just POSTs
audio to it and reads back the transcript. Keeping whisper.cpp as an
external process means the model stays resident in memory across
utterances instead of reloading weights every time you speak.
"""

import io
import wave

import numpy as np
import requests


def _to_wav_bytes(audio: np.ndarray, sample_rate: int = 16000) -> bytes:
    """Convert float32 [-1, 1] mono audio to 16-bit PCM WAV bytes."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class WhisperClient:
    def __init__(self, server_url: str, sample_rate: int = 16000, language: str = "en"):
        self.server_url = server_url.rstrip("/")
        self.sample_rate = sample_rate
        self.language = language

    def transcribe(self, audio: np.ndarray) -> str:
        wav_bytes = _to_wav_bytes(audio, self.sample_rate)
        files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
        data = {"response_format": "json", "language": self.language}
        resp = requests.post(f"{self.server_url}/inference", files=files, data=data, timeout=30)
        resp.raise_for_status()
        return resp.json().get("text", "").strip()
