"""Client for a running whisper.cpp `server` instance.

This assumes the server binary from whisper.cpp is already running
separately with a model loaded (see README.md) — this client just POSTs
audio to it and reads back the transcript. Keeping whisper.cpp as an
external process means the model stays resident in memory across
utterances instead of reloading weights every time you speak.
"""

import io
import time
import wave

import numpy as np
import requests


class WhisperUnavailableError(Exception):
    """Raised when whisper.cpp's server can't be reached, times out, or
    errors out after all retries -- callers (main.py) treat this as
    "the server is down", not just "this one request failed"."""


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
    def __init__(
        self,
        server_url: str,
        sample_rate: int = 16000,
        language: str = "en",
        request_timeout_s: float = 30.0,
        max_retries: int = 2,
        retry_backoff_s: float = 1.0,
    ):
        self.server_url = server_url.rstrip("/")
        self.sample_rate = sample_rate
        self.language = language
        self.request_timeout_s = request_timeout_s
        self.max_retries = max_retries
        self.retry_backoff_s = retry_backoff_s

    def transcribe(self, audio: np.ndarray, prompt: str = None) -> str:
        """prompt biases decoding toward expected words (e.g. a wake phrase) —
        whisper.cpp's /inference honors it per-request. Matters a lot for
        short, out-of-context utterances: without a hint, a clean "Gemma."
        got misheard as "D'Emma"; with one, it came back correct.

        Retries transient failures (connection refused/reset, timeout) up
        to max_retries times with a doubling backoff before giving up — a
        single dropped request shouldn't trigger main.py's full
        reconnect-and-wait flow, only a server that's genuinely down.
        Raises WhisperUnavailableError once retries are exhausted, so
        main.py can tell "server unreachable" apart from other failures.
        """
        wav_bytes = _to_wav_bytes(audio, self.sample_rate)
        files = {"file": ("utterance.wav", wav_bytes, "audio/wav")}
        data = {"response_format": "json", "language": self.language}
        if prompt:
            data["prompt"] = prompt

        last_exc = None
        for attempt in range(self.max_retries + 1):
            try:
                resp = requests.post(
                    f"{self.server_url}/inference",
                    files=files,
                    data=data,
                    timeout=self.request_timeout_s,
                )
                resp.raise_for_status()
                return resp.json().get("text", "").strip()
            except requests.RequestException as exc:
                last_exc = exc
                if attempt < self.max_retries:
                    time.sleep(self.retry_backoff_s * (2**attempt))
        raise WhisperUnavailableError(
            f"whisper.cpp server at {self.server_url} is unreachable: {last_exc}"
        ) from last_exc

    def is_available(self, timeout_s: float = 2.0) -> bool:
        """Cheap reachability check used by main.py while waiting for the
        server to come back — any response at all (even a 404) means the
        process is up and listening; only a connection-level failure
        (refused, reset, timed out) means it isn't."""
        try:
            requests.get(f"{self.server_url}/", timeout=timeout_s)
            return True
        except requests.RequestException:
            return False
