"""Continuous microphone capture, yielding fixed-size float32 chunks.

Silero VAD requires specific chunk sizes (512 samples at 16kHz), so the
mic stream is opened with that exact blocksize — every chunk handed to
the VAD is already the right shape, no buffering logic needed upstream.

Also plays back reference/reply audio through this same stream (see
play()/is_playing()/stop_playback()) rather than through a second,
independent output stream. Capture and TTS playback used to be two
separate PortAudio streams (this one, plus audio.playback's sd.play());
running both concurrently during barge-in monitoring caused
intermittent crackling in the TTS output, even though neither stream
ever reported a buffer underrun and per-chunk processing measured well
within budget (tests/measure_output_underruns.py). Root cause: two
independently opened streams don't share one hardware clock, and on
this machine the mic and speakers are exposed as separate CoreAudio
devices even though physically part of the same machine -- a brief
scheduling hiccup in one stream's callback isn't synchronized with the
other's, which can glitch the DAC without either side's `status` flag
ever being set.
One full-duplex stream removes that two-clock-domain path by
construction: input and output are produced in the same callback
invocation, on the same clock, every time.
"""

import queue
import threading

import numpy as np
import sounddevice as sd


class MicStream:
    def __init__(self, sample_rate=16000, chunk_samples=512, channels=1, device=None, output_device=None):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.channels = channels
        self.device = device
        self.output_device = output_device
        self._q = queue.Queue()
        self._stream = None

        self._play_lock = threading.Lock()
        self._play_audio = None
        self._play_pos = 0
        self._play_active = False

    def _callback(self, indata, outdata, frames, time_info, status):
        if status:
            print(f"[MicStream] status: {status}")
        # indata shape is (frames, channels) float32 in [-1, 1]
        mono = indata[:, 0].copy()
        self._q.put(mono)

        with self._play_lock:
            if self._play_active and self._play_audio is not None:
                remaining = len(self._play_audio) - self._play_pos
                n = min(frames, remaining)
                if n > 0:
                    outdata[:n, 0] = self._play_audio[self._play_pos : self._play_pos + n]
                    self._play_pos += n
                if n < frames:
                    outdata[n:, 0] = 0.0
                if remaining <= frames:
                    self._play_active = False
            else:
                outdata[:, 0] = 0.0

    def start(self):
        self._stream = sd.Stream(
            samplerate=self.sample_rate,
            channels=self.channels,
            dtype="float32",
            blocksize=self.chunk_samples,
            device=(self.device, self.output_device),
            callback=self._callback,
        )
        self._stream.start()

    def stop(self):
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()
            self._stream = None

    def play(self, audio: np.ndarray):
        """Starts playing `audio` (float32 mono, at self.sample_rate)
        through this stream's output side, without blocking. Call
        start() first."""
        with self._play_lock:
            self._play_audio = np.ascontiguousarray(audio, dtype=np.float32)
            self._play_pos = 0
            self._play_active = True

    def is_playing(self) -> bool:
        with self._play_lock:
            return self._play_active

    def stop_playback(self):
        """Interrupts playback started via play(), if any is in progress."""
        with self._play_lock:
            self._play_active = False
            self._play_audio = None
            self._play_pos = 0

    def chunks(self):
        """Generator yielding chunk_samples-length float32 numpy arrays.

        Blocks until the next chunk is available. Call start() first.
        """
        while True:
            yield self._q.get()

    def drain(self):
        """Discards any chunks queued but not yet consumed.

        The callback keeps queuing chunks continuously, including while
        nobody's pulling from chunks() (e.g. during a blocking STT/LLM/TTS
        call). Without this, the next chunks() consumed after such a gap
        would be a burst of stale, already-past audio rather than
        real-time input -- which breaks anything that assumes "one chunk
        pulled per loop iteration" tracks real elapsed time, such as
        speak_with_barge_in's echo-canceller reference alignment and grace
        period.
        """
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return
