"""Lightweight adaptive noise reduction for highway-speed vehicle noise.

No noise sample was available to calibrate against, so this is designed
from documented vehicle cabin acoustics rather than measured: highway
cabin noise runs ~65-71 dB(A), rising ~3dB per +10mph, and peaks in the
100-500Hz band -- overlapping bass and the lower vocal midrange, with
frequencies below ~300Hz mostly structure-borne and above mostly
airborne (what a mic actually picks up). Spectral subtraction is the
standard technique for in-car ASR noise robustness in the literature,
validated specifically for *stationary* cruising noise -- a reasonable
match for steady highway speed, though it degrades on non-stationary
noise (potholes, passing trucks) and can add "musical noise" artifacts
if oversubtraction is set too high. Retune via config.yaml once tested
against the real vehicle; the defaults here are reasoned, not measured.

Two stages, both continuously adaptive:

1. HighPassFilter: a biquad (Robert Bristow-Johnson's Audio EQ Cookbook
   formula) cutting below ~100Hz -- deep engine/structural rumble that's
   almost entirely non-speech, removed unconditionally and cheaply
   regardless of whether the driver is talking.

2. SpectralNoiseReducer: classic spectral subtraction (Boll 1979) with
   Berouti et al.'s oversubtraction + spectral floor to limit musical
   noise, continuously estimating the noise magnitude spectrum from
   low-energy frames (no separate noise-only calibration period needed)
   and subtracting it from every frame -- including frames that also
   contain speech, which is what actually matters at highway speed,
   since the noise is present throughout, not just between utterances.
   Uses 50%-overlap Hann-windowed overlap-add (COLA-exact at this
   overlap), adding exactly one chunk (32ms) of latency.

   First attempt at the noise/non-noise decision used the bottom 30th
   percentile of recent frame energy, updating fast (rate 0.1). Tested
   against real whisper.cpp transcription on synthesized speech mixed
   with a synthetic highway-noise proxy (shaped to the 100-500Hz peak
   above): the noise estimate drifted upward through the utterance
   (mean magnitude rose ~48% over 6s) because a 30th-percentile
   threshold guarantees ~30% of frames get treated as "noise" no
   matter what -- including quiet speech-adjacent frames (inter-
   syllable dips), not just true silence. That contamination
   increasingly over-subtracted real speech, and transcription was
   *worse* than doing nothing. A stricter percentile (10th), slower
   update (0.05), and longer history (150 frames, ~4.8s) fixed it --
   re-tested at 10dB/5dB SNR with transcription matching clean audio
   exactly, only degrading (like any system would) at unrealistically
   harsh 0dB/-5dB SNR.
"""

import numpy as np


class HighPassFilter:
    """RBJ Audio EQ Cookbook biquad high-pass. Filter state persists across
    chunks, unlike a naive independent per-chunk filter, so there's no
    discontinuity at chunk boundaries."""

    def __init__(self, sample_rate: int, cutoff_hz: float, q: float = 0.707):
        w0 = 2 * np.pi * cutoff_hz / sample_rate
        alpha = np.sin(w0) / (2 * q)
        cos_w0 = np.cos(w0)
        a0 = 1 + alpha
        self.b0 = ((1 + cos_w0) / 2) / a0
        self.b1 = (-(1 + cos_w0)) / a0
        self.b2 = ((1 + cos_w0) / 2) / a0
        self.a1 = (-2 * cos_w0) / a0
        self.a2 = (1 - alpha) / a0
        self._x1 = 0.0
        self._x2 = 0.0
        self._y1 = 0.0
        self._y2 = 0.0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        out = np.empty_like(chunk)
        b0, b1, b2, a1, a2 = self.b0, self.b1, self.b2, self.a1, self.a2
        x1, x2, y1, y2 = self._x1, self._x2, self._y1, self._y2
        for i in range(len(chunk)):
            x0 = float(chunk[i])
            y0 = b0 * x0 + b1 * x1 + b2 * x2 - a1 * y1 - a2 * y2
            out[i] = y0
            x2, x1 = x1, x0
            y2, y1 = y1, y0
        self._x1, self._x2, self._y1, self._y2 = x1, x2, y1, y2
        return out


class SpectralNoiseReducer:
    def __init__(
        self,
        chunk_size: int,
        oversubtraction: float = 1.2,
        spectral_floor: float = 0.1,
        noise_update_rate: float = 0.05,
        noise_energy_history: int = 150,
        noise_percentile: float = 10.0,
    ):
        self.chunk_size = chunk_size
        self.frame_size = chunk_size * 2
        self.window = np.hanning(self.frame_size)
        self.oversubtraction = oversubtraction
        self.spectral_floor = spectral_floor
        self.noise_update_rate = noise_update_rate
        self.noise_percentile = noise_percentile
        self._prev_chunk = np.zeros(chunk_size, dtype=np.float64)
        self._prev_tail = np.zeros(chunk_size, dtype=np.float64)
        self._noise_mag = None
        self._energy_history = np.zeros(noise_energy_history)
        self._energy_index = 0
        self._energy_filled = 0

    def process(self, chunk: np.ndarray) -> np.ndarray:
        frame = np.concatenate([self._prev_chunk, chunk.astype(np.float64)])
        self._prev_chunk = chunk.astype(np.float64)

        windowed = frame * self.window
        spectrum = np.fft.rfft(windowed)
        mag = np.abs(spectrum)

        energy = float(np.mean(mag**2))
        history_len = min(self._energy_filled, len(self._energy_history))
        if self._noise_mag is None:
            self._noise_mag = mag.copy()
        elif history_len >= 10:
            threshold = np.percentile(self._energy_history[:history_len], self.noise_percentile)
            if energy <= threshold:
                self._noise_mag = (
                    1 - self.noise_update_rate
                ) * self._noise_mag + self.noise_update_rate * mag
        self._energy_history[self._energy_index % len(self._energy_history)] = energy
        self._energy_index += 1
        self._energy_filled += 1

        gain = 1.0 - self.oversubtraction * (self._noise_mag / (mag + 1e-8))
        gain = np.clip(gain, self.spectral_floor, 1.0)
        clean_spectrum = spectrum * gain
        clean_frame = np.fft.irfft(clean_spectrum, n=self.frame_size)

        first_half = clean_frame[: self.chunk_size] + self._prev_tail
        self._prev_tail = clean_frame[self.chunk_size :]
        return first_half.astype(np.float32)


class NoiseReducer:
    """Combines both stages: high-pass first (cheap, unconditional), then
    spectral subtraction (adaptive, handles noise concurrent with speech)."""

    def __init__(
        self,
        sample_rate: int,
        chunk_size: int,
        highpass_hz: float = 100.0,
        oversubtraction: float = 1.2,
        spectral_floor: float = 0.1,
        noise_update_rate: float = 0.05,
        noise_energy_history: int = 150,
        noise_percentile: float = 10.0,
    ):
        self.highpass = HighPassFilter(sample_rate, highpass_hz) if highpass_hz > 0 else None
        self.spectral = SpectralNoiseReducer(
            chunk_size,
            oversubtraction=oversubtraction,
            spectral_floor=spectral_floor,
            noise_update_rate=noise_update_rate,
            noise_energy_history=noise_energy_history,
            noise_percentile=noise_percentile,
        )

    def process(self, chunk: np.ndarray) -> np.ndarray:
        if self.highpass is not None:
            chunk = self.highpass.process(chunk)
        return self.spectral.process(chunk)
