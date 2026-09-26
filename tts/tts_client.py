"""Wraps the Piper TTS CLI binary.

Piper is the v1 voice: a single local call that returns a WAV directly,
with none of the token-decode plumbing an LLM-based TTS model (like
Orpheus) needs. Swap this module out once the rest of the loop is
proven and you want a more expressive voice.
"""

import os
import subprocess
import tempfile


class PiperTTS:
    def __init__(
        self,
        binary: str = "piper",
        model_path: str = None,
        config_path: str = None,
        speaker: int = None,
    ):
        self.binary = binary
        self.model_path = model_path
        self.config_path = config_path
        self.speaker = speaker

    def synthesize(self, text: str) -> bytes:
        """Returns WAV bytes for the given text."""
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as tmp:
            out_path = tmp.name

        try:
            cmd = [self.binary, "--model", self.model_path, "--output_file", out_path]
            if self.config_path:
                cmd += ["--config", self.config_path]
            if self.speaker is not None:
                cmd += ["--speaker", str(self.speaker)]

            subprocess.run(cmd, input=text.encode("utf-8"), capture_output=True, check=True)

            with open(out_path, "rb") as f:
                return f.read()
        finally:
            if os.path.exists(out_path):
                os.remove(out_path)
