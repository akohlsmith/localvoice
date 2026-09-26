"""Single-channel acoustic echo cancellation via per-sample NLMS.

Used only while barge-in monitors the mic during TTS playback. Measured
on this machine: with speakers and mic on the same device, the mic picks
up the assistant's own voice at a real, stable delay (~410ms — mostly
audio-driver buffering, not acoustic travel time) at a strength Silero
VAD scores at ~1.0 confidence, sustained for seconds — indistinguishable
from a real interruption by threshold alone (see PLAN.md's known gaps).

This holds a rolling buffer of the known reference (playback) signal and
adapts a short FIR filter to predict the echo from it, subtracting the
prediction to leave (mostly) just the near-end signal. Validated offline
against a real captured recording: raw mic sustained >=0.9 VAD confidence
for 3.26s straight; the residual only sustained it for 96ms.

Not a general-purpose AEC: the delay is configured, not auto-detected
(it was measured stable across a whole clip on this machine — see
config.yaml's comment on why it might need recalibrating elsewhere), no
non-linear echo modeling, no double-talk detection. Good enough for this
one fixed, direct echo path.
"""

import numpy as np
from numpy.lib.stride_tricks import sliding_window_view


def resample_linear(audio: np.ndarray, orig_sr: int, target_sr: int) -> np.ndarray:
    """Piper voices don't all share one native sample rate; this keeps the
    reference aligned to the mic's rate (16kHz) regardless of which voice
    is configured."""
    if orig_sr == target_sr:
        return audio.astype(np.float32)
    duration = len(audio) / orig_sr
    n_target = max(1, int(round(duration * target_sr)))
    orig_times = np.arange(len(audio)) / orig_sr
    target_times = np.arange(n_target) / target_sr
    return np.interp(target_times, orig_times, audio).astype(np.float32)


class EchoCanceller:
    def __init__(
        self,
        sample_rate: int,
        expected_delay_ms: float,
        filter_length_ms: float,
        mu: float = 0.5,
        eps: float = 1e-4,
    ):
        self.sample_rate = sample_rate
        self.delay = max(0, int(expected_delay_ms / 1000 * sample_rate))
        self.filter_length = max(1, int(filter_length_ms / 1000 * sample_rate))
        self.mu = mu
        self.eps = eps
        self.weights = np.zeros(self.filter_length, dtype=np.float64)
        # Long enough to hold, for the newest chunk, a full filter_length
        # window ending `delay` samples before "now" -- plus one chunk of
        # slack since successive chunk-sized windows shift within it.
        self._max_chunk = sample_rate  # generous upper bound, resized on first push if needed
        self._history_len = self.delay + self.filter_length + self._max_chunk
        self.ref_history = np.zeros(self._history_len, dtype=np.float64)

    def reset(self):
        self.weights[:] = 0.0
        self.ref_history[:] = 0.0

    def push_reference(self, ref_chunk: np.ndarray):
        """Call once per mic chunk, with the reference audio queued for
        the speaker at (approximately) the same real-time position —
        i.e. advance through the played-back array in lockstep with the
        incoming mic chunks, not with wall-clock time directly."""
        n = len(ref_chunk)
        if n >= self._history_len:
            self.ref_history = ref_chunk[-self._history_len :].astype(np.float64)
        else:
            self.ref_history = np.concatenate([self.ref_history[n:], ref_chunk.astype(np.float64)])

    def cancel(self, mic_chunk: np.ndarray) -> np.ndarray:
        """Call once per mic chunk, immediately after push_reference() for
        that same chunk. Returns the echo-cancelled residual."""
        n = len(mic_chunk)
        fl = self.filter_length
        history = self.ref_history
        # For output sample j (j=n-1 is "now"), the window must end at
        # history's position `len(history) - n + j - delay` -- i.e. delay
        # samples before that sample's own position in history. That
        # spans history[end-(n+fl-1) : end] with end = len(history)-delay;
        # sliding_window_view over exactly that span yields the n windows
        # directly, one per output sample, in order.
        end = len(history) - self.delay
        start = end - (n + fl - 1)
        if start < 0:
            segment = np.concatenate([np.zeros(-start), history[0:end]])
        else:
            segment = history[start:end]
        windows = sliding_window_view(segment, fl)

        weights = self.weights
        residual = np.empty(n, dtype=np.float64)
        for i in range(n):
            window = windows[i]
            y_hat = window @ weights
            e = mic_chunk[i] - y_hat
            residual[i] = e
            norm = window @ window + self.eps
            weights = weights + (self.mu * e / norm) * window
        self.weights = weights
        return residual.astype(np.float32)
