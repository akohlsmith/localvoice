"""Wraps the Piper TTS CLI binary.

Piper is the v1 voice: a single local call that returns a WAV directly,
with none of the token-decode plumbing an LLM-based TTS model (like
Orpheus) needs. Swap this module out once the rest of the loop is
proven and you want a more expressive voice.
"""

import os
import re
import subprocess
import tempfile

# Piper has no SSML/emphasis support at all (confirmed: `piper --help` has
# only global params -- length-scale, noise-scale, volume -- no per-phrase
# markup). Worse, it doesn't just ignore markdown asterisks, it reads them
# aloud as the literal word "asterisk" (verified: "The *other* door" ->
# "The asterisk other asterisk door"). The system_prompt in config.yaml
# tells the model not to use markdown, but smaller/quantized models often
# leak formatting habits anyway -- this is the defense-in-depth cleanup
# for whatever slips through, not the primary fix.
_MARKDOWN_LINK = re.compile(r"\[([^\]]+)\]\([^)]+\)")
_MARKDOWN_HEADING = re.compile(r"^#{1,6}\s*", flags=re.MULTILINE)
_MARKDOWN_BULLET = re.compile(r"^[ \t]*[-*+]\s+", flags=re.MULTILINE)
_MARKDOWN_BOLD_STAR = re.compile(r"\*\*(.+?)\*\*")
_MARKDOWN_BOLD_UNDERSCORE = re.compile(r"__(.+?)__")
_MARKDOWN_ITALIC_STAR = re.compile(r"\*(.+?)\*")
_MARKDOWN_ITALIC_UNDERSCORE = re.compile(r"(?<!\w)_(.+?)_(?!\w)")
_MARKDOWN_CODE = re.compile(r"`([^`]+)`")


def strip_markdown(text: str) -> str:
    text = _MARKDOWN_LINK.sub(r"\1", text)
    text = _MARKDOWN_HEADING.sub("", text)
    text = _MARKDOWN_BULLET.sub("", text)
    text = _MARKDOWN_BOLD_STAR.sub(r"\1", text)
    text = _MARKDOWN_BOLD_UNDERSCORE.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_STAR.sub(r"\1", text)
    text = _MARKDOWN_ITALIC_UNDERSCORE.sub(r"\1", text)
    text = _MARKDOWN_CODE.sub(r"\1", text)
    return text


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
        text = strip_markdown(text)
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
