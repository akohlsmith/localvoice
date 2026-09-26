"""Entry point: hands-free voice loop.

listen -> VAD detects end of utterance -> whisper.cpp transcribes ->
LM Studio generates a reply -> Piper speaks it -> back to listening.

Run `python main.py` after starting LM Studio's server and whisper.cpp's
server (see README.md).
"""

import sys

import yaml

from audio.capture import MicStream
from audio.playback import play_wav_bytes
from audio.vad import SileroVAD, TurnDetector
from llm.chat_client import ChatClient
from stt.whisper_client import WhisperClient
from tts.tts_client import PiperTTS

CHUNK_SAMPLES = 512  # required window size for Silero VAD at 16kHz


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def main():
    cfg = load_config()
    sample_rate = cfg["audio"]["sample_rate"]

    mic = MicStream(
        sample_rate=sample_rate,
        chunk_samples=CHUNK_SAMPLES,
        channels=cfg["audio"]["channels"],
        device=cfg["audio"]["input_device"],
    )

    vad = SileroVAD(sample_rate=sample_rate)
    turn_detector = TurnDetector(
        vad,
        sample_rate=sample_rate,
        chunk_samples=CHUNK_SAMPLES,
        prob_threshold=cfg["vad"]["speech_prob_threshold"],
        silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
        min_speech_ms=cfg["vad"]["min_speech_ms"],
    )

    stt = WhisperClient(
        server_url=cfg["stt"]["whisper_server_url"],
        sample_rate=sample_rate,
        language=cfg["stt"]["language"],
    )

    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
    )

    chat = ChatClient(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        system_prompt=cfg["llm"].get("system_prompt", ""),
        temperature=cfg["llm"].get("temperature", 0.7),
    )

    print("Listening... (Ctrl+C to quit)")
    mic.start()
    try:
        for chunk in mic.chunks():
            utterance = turn_detector.process_chunk(chunk)
            if utterance is None:
                continue

            text = stt.transcribe(utterance)
            if not text:
                continue
            print(f"You said: {text}")

            reply = chat.ask(text)
            print(f"Assistant: {reply}")

            wav_bytes = tts.synthesize(reply)
            play_wav_bytes(wav_bytes)
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mic.stop()


if __name__ == "__main__":
    sys.exit(main())
