"""Continuous microphone capture, yielding fixed-size float32 chunks.

Silero VAD requires specific chunk sizes (512 samples at 16kHz), so the
mic stream is opened with that exact blocksize — every chunk handed to
the VAD is already the right shape, no buffering logic needed upstream.
"""

import queue

import sounddevice as sd


class MicStream:
    def __init__(self, sample_rate=16000, chunk_samples=512, channels=1, device=None):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.channels = channels
        self.device = device
        self._q = queue.Queue()
        self._stream = None

    def _callback(self, indata, frames, time_info, status):
        if status:
            print(f"[MicStream] status: {status}")
        # indata shape is (frames, channels) float32 in [-1, 1]
        mono = indata[:, 0].copy()
        self._q.put(mono)

    def start(self):
        self._stream = sd.InputStream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=self.device,
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def chunks(self):
        """Generator yielding chunk_samples-length float32 numpy arrays.

        Blocks until the next chunk is available. Call start() first.
        """
        while True:
            yield self._q.get()
