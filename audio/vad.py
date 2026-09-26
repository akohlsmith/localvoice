"""Voice activity detection and turn-taking, built on Silero VAD.

Two pieces:
  - SileroVAD: scores a single audio chunk for speech probability.
  - TurnDetector: consumes a stream of chunks and decides when a full
    utterance has been spoken (speech started, then enough trailing
    silence to call it done). This is the thing that removes the
    keyboard/push-to-talk from the loop.
"""

import numpy as np
import torch


class SileroVAD:
    def __init__(self, sample_rate=16000):
        self.sample_rate = sample_rate
        self.model, _utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad",
            model="silero_vad",
            force_reload=False,
            onnx=False,
        )
        self.model.eval()

    def speech_prob(self, chunk: np.ndarray) -> float:
        """chunk: float32 mono PCM in [-1, 1], length must be a Silero-supported
        window (512 samples at 16kHz)."""
        tensor = torch.from_numpy(chunk)
        with torch.no_grad():
            prob = self.model(tensor, self.sample_rate).item()
        return prob


class TurnDetector:
    """Buffers incoming chunks and returns a complete utterance once the
    speaker has paused long enough to count as done talking."""

    def __init__(
        self,
        vad: SileroVAD,
        sample_rate=16000,
        chunk_samples=512,
        prob_threshold=0.5,
        silence_hangover_ms=700,
        min_speech_ms=250,
    ):
        self.vad = vad
        self.prob_threshold = prob_threshold

        chunk_ms = 1000 * chunk_samples / sample_rate
        self.silence_hangover_frames = max(1, int(silence_hangover_ms / chunk_ms))
        self.min_speech_frames = max(1, int(min_speech_ms / chunk_ms))

        self._reset()

    def _reset(self):
        self._buffer = []
        self._in_speech = False
        self._silence_run = 0
        self._speech_frames = 0

    def process_chunk(self, chunk: np.ndarray):
        """Feed one chunk. Returns a concatenated np.ndarray utterance when a
        turn just ended, otherwise None."""
        prob = self.vad.speech_prob(chunk)
        is_speech = prob >= self.prob_threshold

        if is_speech:
            self._buffer.append(chunk)
            self._in_speech = True
            self._speech_frames += 1
            self._silence_run = 0
            return None

        if not self._in_speech:
            # Silence before any speech has started — nothing to do.
            return None

        # We were in speech and just saw a silent chunk — keep a little
        # trailing silence for natural cadence, then check the hangover.
        self._buffer.append(chunk)
        self._silence_run += 1

        if self._silence_run < self.silence_hangover_frames:
            return None

        utterance = None
        if self._speech_frames >= self.min_speech_frames:
            utterance = np.concatenate(self._buffer)
        self._reset()
        return utterance
