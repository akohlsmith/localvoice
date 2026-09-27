"""PhoneMicStream: the same interface audio/capture.py's MicStream
exposes (start/stop/chunks/play/is_playing/stop_playback/drain), backed
by a phone connected over web/server.py's WebSocket endpoint instead of
sounddevice. main.py only ever calls methods on whichever one it
constructed (see config.yaml's phone_ui.enabled) -- nothing else in the
pipeline (VAD, wake word, noise reduction, STT/LLM/TTS) knows or cares
which source it's getting chunks from.

Also adds one method beyond MicStream's surface: consume_stop_signal(),
for the phone's manual Stop button. MicStream gets a trivial always-False
stub of the same method so speak_with_barge_in can check it uniformly
regardless of which stream type is in use.

Threading model: web/server.py runs a dedicated asyncio event loop on
its own background thread; this class's _on_connection/_handle_* methods
run on that thread (invoked by the WebSocket handler), while start/stop/
chunks/play/is_playing/stop_playback/consume_stop_signal are called from
main.py's synchronous thread. self._state_lock protects everything
shared between the two; the two directions cross via a thread-safe
queue.Queue (phone -> main.py) and PhoneUIServer.run_coroutine() (main.py
-> phone, since sending on a WebSocket connection is itself a coroutine
that must run on its own event loop).
"""

import io
import json
import queue
import threading
import time
import wave

import numpy as np

from audio.echo_cancel import resample_linear
from web.server import PhoneUIServer, generate_token

CHUNK_SAMPLES = 512  # must match main.CHUNK_SAMPLES -- the fixed Silero VAD window at 16kHz


def _wav_bytes_from_float32(audio: np.ndarray, sample_rate: int) -> bytes:
    """Wraps float32 PCM into a WAV container the phone's
    decodeAudioData() can play directly. Mirrors stt/whisper_client.py's
    own _to_wav_bytes -- each module owns its tiny WAV encode/decode
    rather than sharing a private cross-module helper, matching how
    audio/playback.py and stt/whisper_client.py already each have their
    own."""
    clipped = np.clip(audio, -1.0, 1.0)
    pcm16 = (clipped * 32767).astype(np.int16)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm16.tobytes())
    return buf.getvalue()


class PhoneMicStream:
    def __init__(
        self,
        host: str,
        port: int,
        cert_dir: str,
        auth_token: str = None,
        sample_rate: int = 16000,
        chunk_samples: int = CHUNK_SAMPLES,
    ):
        self.sample_rate = sample_rate
        self.chunk_samples = chunk_samples
        self.port = port
        self.auth_token = auth_token or generate_token()
        self._server = PhoneUIServer(host, port, cert_dir, self.auth_token, self._on_connection)

        self._q = queue.Queue()
        self._state_lock = threading.Lock()
        self._connection = None
        self._native_sample_rate = None
        self._resample_carry = np.zeros(0, dtype=np.float32)
        self._playing = False
        self._stop_requested = False
        self._last_push_time = time.monotonic()

        self._heartbeat_stop = threading.Event()
        self._heartbeat_thread = None

    def start(self):
        self._server.start()
        self._heartbeat_stop.clear()
        self._heartbeat_thread = threading.Thread(target=self._heartbeat_loop, daemon=True)
        self._heartbeat_thread.start()

    def stop(self):
        self._heartbeat_stop.set()
        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=2)
        self._server.stop()

    def chunks(self):
        """Blocks until the next chunk is available -- either real
        phone audio or a heartbeat silence chunk (see _heartbeat_loop)."""
        while True:
            yield self._q.get()

    def drain(self):
        while True:
            try:
                self._q.get_nowait()
            except queue.Empty:
                return

    def play(self, audio: np.ndarray):
        """audio arrives as 16kHz float32 PCM, same as MicStream.play()
        receives (main.speak_with_barge_in decodes+resamples Piper's WAV
        to mic.sample_rate before calling this, unconditionally -- see
        PLAN.md's phone_ui entry for why reusing that path, rather than
        sending Piper's native-rate WAV, is a deliberate non-change to
        the existing local-playback pipeline, not a new quality
        trade-off introduced here)."""
        wav_bytes = _wav_bytes_from_float32(audio, self.sample_rate)
        with self._state_lock:
            connection = self._connection
            self._playing = connection is not None

        if connection is None:
            # Nothing to play to -- don't leave is_playing() stuck True
            # waiting for an ack from a phone that was never there.
            return
        self._server.run_coroutine(self._send_audio(connection, wav_bytes))

    def is_playing(self) -> bool:
        with self._state_lock:
            return self._playing

    def stop_playback(self):
        with self._state_lock:
            self._playing = False

    def consume_stop_signal(self) -> bool:
        """True at most once per phone Stop-button tap -- consuming it
        clears the flag, so speak_with_barge_in's loop (which polls this
        every chunk) reacts to each press exactly once."""
        with self._state_lock:
            requested = self._stop_requested
            self._stop_requested = False
        return requested

    async def _send_audio(self, connection, wav_bytes: bytes):
        try:
            await connection.send(wav_bytes)
        except Exception as exc:  # connection dropped mid-send, etc.
            print(f"[PhoneMicStream] failed to send audio to phone: {exc}")
            with self._state_lock:
                self._playing = False

    async def _on_connection(self, connection):
        """Runs on web/server.py's event-loop thread, once per accepted
        WebSocket connection. Single concurrent phone by design -- a new
        connection simply replaces whatever was there before."""
        with self._state_lock:
            self._connection = connection
            self._native_sample_rate = None
            self._resample_carry = np.zeros(0, dtype=np.float32)
        print("[PhoneMicStream] phone connected")
        try:
            async for message in connection:
                self._handle_message(message)
        finally:
            with self._state_lock:
                if self._connection is connection:
                    self._connection = None
                    self._playing = False
                    self._stop_requested = False
            print("[PhoneMicStream] phone disconnected")

    def _handle_message(self, message):
        if isinstance(message, (bytes, bytearray)):
            self._handle_audio_frame(message)
            return

        try:
            data = json.loads(message)
        except ValueError:
            return
        msg_type = data.get("type")

        if msg_type == "hello":
            with self._state_lock:
                self._native_sample_rate = data.get("sample_rate")
            print(f"[PhoneMicStream] phone native sample rate: {self._native_sample_rate}")
        elif msg_type == "enable":
            print("[PhoneMicStream] phone mic enabled")
        elif msg_type == "disable":
            print("[PhoneMicStream] phone mic disabled")
        elif msg_type == "stop":
            with self._state_lock:
                self._stop_requested = True
                self._playing = False
        elif msg_type == "playback_finished":
            with self._state_lock:
                self._playing = False

    def _handle_audio_frame(self, frame: bytes):
        with self._state_lock:
            native_rate = self._native_sample_rate
        if native_rate is None:
            return  # ignore audio that arrives before the hello handshake

        incoming = np.frombuffer(frame, dtype="<f4")
        if len(incoming) == 0:
            return
        resampled = resample_linear(incoming, native_rate, self.sample_rate)

        with self._state_lock:
            buffer = np.concatenate([self._resample_carry, resampled])
            n_complete = len(buffer) // self.chunk_samples
            for i in range(n_complete):
                chunk = buffer[i * self.chunk_samples : (i + 1) * self.chunk_samples]
                self._q.put(chunk)
            self._resample_carry = buffer[n_complete * self.chunk_samples :]
            self._last_push_time = time.monotonic()

    def _heartbeat_loop(self):
        """Keeps main.py's `for chunk in mic.chunks(): ...` loops alive
        with silent chunks whenever the phone isn't actively streaming
        real audio (disabled, not yet connected, or between utterances).
        Unlike PortAudio's local callback, nothing here guarantees this
        source produces data on its own -- without this, the loop (and
        speak_with_barge_in's barge-in monitoring loop) would block
        forever the first time nothing is being streamed."""
        chunk_period_s = self.chunk_samples / self.sample_rate
        silence = np.zeros(self.chunk_samples, dtype=np.float32)
        while not self._heartbeat_stop.is_set():
            time.sleep(chunk_period_s)
            with self._state_lock:
                idle_for = time.monotonic() - self._last_push_time
            if idle_for >= chunk_period_s:
                self._q.put(silence)
                with self._state_lock:
                    self._last_push_time = time.monotonic()
