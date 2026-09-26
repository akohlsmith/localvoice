"""Wake-word gate: hands back a completed utterance only when it was
preceded by a configured stretch of silence, so main.py can transcribe
just that candidate and check it against the wake phrase before switching
into full command-capture mode.

Wraps its own TurnDetector for speech/silence boundaries rather than
reusing a separate wake-word model — no training pipeline needed. Music
or background noise that happens to cross the VAD's speech threshold
still has to be transcribed and match the phrase exactly (see main.py),
which is a second, much stricter filter beyond VAD alone.
"""

from audio.vad import TurnDetector


class WakeWordGate:
    def __init__(
        self,
        vad,
        sample_rate: int,
        chunk_samples: int,
        prob_threshold: float,
        silence_hangover_ms: int,
        min_speech_ms: int,
        silence_before_ms: int,
    ):
        self._turn_detector = TurnDetector(
            vad,
            sample_rate=sample_rate,
            chunk_samples=chunk_samples,
            prob_threshold=prob_threshold,
            silence_hangover_ms=silence_hangover_ms,
            min_speech_ms=min_speech_ms,
        )
        chunk_ms = 1000 * chunk_samples / sample_rate
        self._silence_before_frames = max(1, int(silence_before_ms / chunk_ms))
        self._silence_run = 0
        self._preceded_by_silence = False

    def process_chunk(self, chunk):
        """Feed one chunk. Returns a completed utterance (numpy array) only
        if it started after enough silence — otherwise None, whether
        nothing finished yet or it was discarded for starting too soon
        after other speech (so it never gets treated as a wake-word
        attempt in the first place)."""
        was_in_speech = self._turn_detector.in_speech
        utterance = self._turn_detector.process_chunk(chunk)

        if not was_in_speech:
            if self._turn_detector.in_speech:
                # Speech just started — lock in whether it earned an attempt.
                self._preceded_by_silence = self._silence_run >= self._silence_before_frames
            else:
                self._silence_run += 1

        if utterance is None:
            return None

        qualifies = self._preceded_by_silence
        self._silence_run = 0
        self._preceded_by_silence = False
        return utterance if qualifies else None
