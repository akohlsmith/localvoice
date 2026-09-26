"""Entry point: hands-free voice loop.

listen -> VAD detects end of utterance -> whisper.cpp transcribes ->
LM Studio generates a reply -> Piper speaks it -> back to listening.

Run `python main.py` after starting LM Studio's server and whisper.cpp's
server (see README.md).
"""

import sys

import yaml

from audio import playback
from audio.capture import MicStream
from audio.vad import SileroVAD, TurnDetector
from llm.chat_client import ChatClient
from stt.whisper_client import WhisperClient
from tts.tts_client import PiperTTS

CHUNK_SAMPLES = 512  # required window size for Silero VAD at 16kHz
BARGE_IN_CONSECUTIVE_FRAMES = 2  # ~64ms of speech before treating it as a real interruption, not a blip


def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def speak_with_barge_in(wav_bytes, mic, turn_detector) -> bool:
    """Plays a reply; stops immediately if the mic picks up the user talking.

    Returns True if playback was cut short. The chunk that triggered the cut
    is fed into turn_detector so capture of the new utterance starts right
    away, without waiting for the next full loop iteration.
    """
    playback.play_wav_bytes_async(wav_bytes)
    consecutive_speech = 0
    for chunk in mic.chunks():
        if not playback.is_playing():
            return False
        prob = turn_detector.vad.speech_prob(chunk)
        if prob >= turn_detector.prob_threshold:
            consecutive_speech += 1
            if consecutive_speech >= BARGE_IN_CONSECUTIVE_FRAMES:
                playback.stop()
                turn_detector.process_chunk(chunk)
                return True
        else:
            consecutive_speech = 0
    return False


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
        speaker=cfg["tts"].get("speaker"),
    )

    browser = None
    tool_schemas = None
    tool_dispatch = None
    browser_cfg = cfg.get("tools", {}).get("browser", {})
    if browser_cfg.get("enabled"):
        from tools.browser import BraveTools
        from tools.registry import TOOL_SCHEMAS, build_dispatcher

        browser = BraveTools(
            profile_dir=browser_cfg["profile_dir"],
            headless=browser_cfg.get("headless", True),
            timeout_ms=browser_cfg.get("timeout_ms", 15000),
        )
        tool_schemas = TOOL_SCHEMAS
        tool_dispatch = build_dispatcher(browser)

    chat = ChatClient(
        base_url=cfg["llm"]["base_url"],
        model=cfg["llm"]["model"],
        system_prompt=cfg["llm"].get("system_prompt", ""),
        temperature=cfg["llm"].get("temperature", 0.7),
        tools=tool_schemas,
        tool_dispatch=tool_dispatch,
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
            if speak_with_barge_in(wav_bytes, mic, turn_detector):
                print("(interrupted)")
    except KeyboardInterrupt:
        print("\nStopping.")
    finally:
        mic.stop()
        if browser is not None:
            browser.stop()


if __name__ == "__main__":
    sys.exit(main())
