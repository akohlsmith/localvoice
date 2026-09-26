"""Live smoke test for the reconnection logic added in
stt/whisper_client.py, tts/tts_client.py, and main.py's
call_with_reconnect(). Unlike tests/test_service_reconnect.py (which
mocks every failure), this one uses real processes: it launches a real
whisper.cpp `server` on a scratch port, kills it outright, and confirms
main.call_with_reconnect() actually notices, waits, and resumes once a
replacement server comes up on its own (via a delayed background
respawn) -- the exact "killed and restarted mid-conversation" scenario
PLAN.md flagged as unverified. It does the analogous thing for Piper
(binary temporarily unresolvable, then resolvable again).

Needs real hardware/binaries (whisper-server, piper, a whisper model,
and -- for the LM Studio case -- a running LM Studio server with a
model loaded) and touches real ports/processes, so -- like the rest of
this directory -- it isn't meant for CI. It never touches the project's
own whisper.cpp server (config.yaml's stt.whisper_server_url / whatever
is already running for normal use, e.g. port 8081) or the project's own
LM Studio server; it spins up and tears down its own throwaway whisper
instance on WHISPER_TEST_PORT below, and for LM Studio it never kills
the real app (a GUI app with no scripted restart available here) --
instead it points a ChatClient at a genuinely closed port first (a real
refused connection, not mocked), then retargets it at the real,
already-running LM Studio server on a delayed background thread, which
exercises the identical is_available()/LlmUnavailableError/
call_with_reconnect path a real crash-and-restart would. If LM Studio
isn't reachable at startup, that one test is skipped (with a note) and
the rest still run.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/test_live_reconnect_smoke.py
"""

import os
import shutil
import subprocess
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests

import main  # noqa: E402
from audio.echo_cancel import resample_linear  # noqa: E402
from audio.playback import decode_wav_bytes  # noqa: E402
from llm.chat_client import ChatClient  # noqa: E402
from stt.whisper_client import WhisperClient  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

WHISPER_TEST_PORT = 8099
WHISPER_BINARY = shutil.which("whisper-server") or "/opt/homebrew/bin/whisper-server"
WHISPER_MODEL = "models/ggml-small.en.bin"

results = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name}: {status}")
    results.append(bool(condition))


def start_whisper_server():
    return subprocess.Popen(
        [WHISPER_BINARY, "-m", WHISPER_MODEL, "--host", "127.0.0.1", "--port", str(WHISPER_TEST_PORT)],
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )


def wait_until(condition, timeout_s, poll_s=0.2):
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if condition():
            return True
        time.sleep(poll_s)
    return False


def test_whisper_real_crash_and_restart():
    if not (os.path.exists(WHISPER_BINARY) and os.path.exists(WHISPER_MODEL)):
        print(f"SKIP whisper live test: {WHISPER_BINARY} or {WHISPER_MODEL} not found")
        return

    print(f"Starting a scratch whisper-server on port {WHISPER_TEST_PORT} (leaving the real one alone)...")
    proc = start_whisper_server()
    stt = WhisperClient(f"http://127.0.0.1:{WHISPER_TEST_PORT}", max_retries=1, retry_backoff_s=0.5)
    try:
        up = wait_until(stt.is_available, timeout_s=20)
        check("scratch whisper-server came up", up)
        if not up:
            return

        # Real audio through the real server, via Piper -- not a canned
        # WAV file, so this exercises the exact same encode/decode path
        # main.py uses.
        cfg = main.load_config()
        tts = PiperTTS(
            binary=cfg["tts"]["piper_binary"],
            model_path=cfg["tts"]["piper_model"],
            config_path=cfg["tts"].get("piper_model_config"),
        )
        wav_bytes = tts.synthesize("Testing whisper reconnection, one two three.")
        raw_audio, sr = decode_wav_bytes(wav_bytes)
        audio = resample_linear(raw_audio, sr, stt.sample_rate)

        baseline = stt.transcribe(audio)
        print(f"  baseline transcript: {baseline!r}")
        check("baseline transcription against the live server succeeds", "reconnect" in baseline.lower())

        print("Killing the scratch whisper-server (SIGKILL) to simulate a crash...")
        proc.kill()
        proc.wait(timeout=5)
        down = wait_until(lambda: not stt.is_available(), timeout_s=5)
        check("is_available() reflects the real server being dead", down)

        # Respawn it in the background after a delay, unattended -- this
        # is the "user notices and restarts it" step, done on a timer so
        # call_with_reconnect below has to actually wait and then
        # discover it on its own, not be handed a pre-warmed server.
        respawned = {}

        def delayed_respawn():
            time.sleep(3)
            print("  (background) respawning whisper-server...")
            respawned["proc"] = start_whisper_server()

        threading.Thread(target=delayed_respawn, daemon=True).start()

        t0 = time.time()
        recovered_text = main.call_with_reconnect(
            lambda: stt.transcribe(audio), "whisper.cpp", stt.is_available, poll_interval_s=0.5
        )
        elapsed = time.time() - t0
        print(f"  recovered transcript: {recovered_text!r} (waited {elapsed:.1f}s)")
        check("call_with_reconnect actually waited for the respawn (>2s)", elapsed > 2.0)
        check(
            "call_with_reconnect returned a correct transcription after the real restart",
            "reconnect" in recovered_text.lower(),
        )

        if "proc" in respawned:
            respawned["proc"].kill()
            respawned["proc"].wait(timeout=5)
    finally:
        if proc.poll() is None:
            proc.kill()
            proc.wait(timeout=5)


def test_piper_real_binary_recovery():
    if not shutil.which("piper") and not os.path.exists("venv/bin/piper"):
        print("SKIP Piper live test: no piper binary found")
        return

    cfg = main.load_config()
    real_binary = cfg["tts"]["piper_binary"]
    tts = PiperTTS(
        binary="piper-does-not-exist-binary",
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
    )
    check("is_available() is False for a genuinely missing binary", not tts.is_available())

    # Flip the binary path back to the real one on a timer, then run the
    # same recovery loop main.py uses -- proves synthesize() actually
    # succeeds once Piper is reachable again, not just that the
    # exception type is right.
    def delayed_fix():
        time.sleep(2)
        print("  (background) pointing PiperTTS back at the real binary...")
        tts.binary = real_binary

    threading.Thread(target=delayed_fix, daemon=True).start()

    t0 = time.time()
    wav_bytes = main.call_with_reconnect(
        lambda: tts.synthesize("Piper is back."), "Piper", tts.is_available, poll_interval_s=0.3
    )
    elapsed = time.time() - t0
    print(f"  recovered synth: {len(wav_bytes)} bytes of WAV (waited {elapsed:.1f}s)")
    check("call_with_reconnect waited for Piper to become resolvable again (>1s)", elapsed > 1.0)
    check("synthesize() returns real WAV bytes once Piper is back", wav_bytes[:4] == b"RIFF")


def test_llm_real_connection_recovery():
    cfg = main.load_config()
    real_base_url = cfg["llm"]["base_url"]
    model = cfg["llm"]["model"]

    try:
        requests.get(real_base_url.rstrip("/") + "/models", timeout=3).raise_for_status()
    except requests.RequestException as exc:
        print(f"SKIP LM Studio live test: real server at {real_base_url} not reachable ({exc})")
        return

    # Genuinely closed port -- a real refused connection, not a mocked
    # exception -- standing in for "LM Studio just crashed". We never
    # kill the user's actual running LM Studio/Bionic process here: it's
    # a GUI app with no scripted restart available in this environment,
    # so instead we retarget the same ChatClient at the real server on a
    # delayed background thread, exercising the identical
    # is_available()/LlmUnavailableError/call_with_reconnect path a real
    # crash-and-restart would.
    chat = ChatClient(base_url="http://127.0.0.1:19999/v1", model=model, context_limit=8192, request_timeout_s=5)
    history_before = list(chat.history)

    check("is_available() is False against a genuinely closed port", not chat.is_available())

    try:
        chat.ask("hello")
        raised = False
    except Exception as exc:
        raised = type(exc).__name__ == "LlmUnavailableError"
        print(f"  ask() against the closed port raised: {exc}")
    check("ask() against a real closed port raises LlmUnavailableError", raised)
    check("ask()'s failed attempt left history exactly as it was before", chat.history == history_before)

    def delayed_retarget():
        time.sleep(3)
        print(f"  (background) retargeting ChatClient at the real LM Studio server ({real_base_url})...")
        chat.client.base_url = real_base_url

    threading.Thread(target=delayed_retarget, daemon=True).start()

    t0 = time.time()
    reply = main.call_with_reconnect(
        lambda: chat.ask("Reply with only the single word: RECONNECTED"),
        "LM Studio",
        chat.is_available,
        poll_interval_s=0.5,
    )
    elapsed = time.time() - t0
    print(f"  recovered reply from the real model: {reply!r} (waited {elapsed:.1f}s)")
    check("call_with_reconnect waited for the real server to become reachable (>2s)", elapsed > 2.0)
    check("ask() returned a real reply from the live LM Studio model after reconnecting", bool(reply))
    check(
        "history has exactly one clean user/assistant pair after the recovered call",
        [m["role"] for m in chat.history if m["role"] != "system"] == ["user", "assistant"],
    )


test_whisper_real_crash_and_restart()
test_piper_real_binary_recovery()
test_llm_real_connection_recovery()

print()
print(f"{sum(results)}/{len(results)} passed")
if not all(results):
    sys.exit(1)
