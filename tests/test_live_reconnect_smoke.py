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

Needs real hardware/binaries (whisper-server, piper, a whisper model)
and touches real ports/processes, so -- like the rest of this
directory -- it isn't meant for CI. It never touches the project's own
whisper.cpp server (config.yaml's stt.whisper_server_url / whatever is
already running for normal use, e.g. port 8081); it spins up and tears
down its own instance on WHISPER_TEST_PORT below.

Requires LM Studio is NOT exercised here (no local LM Studio server was
running in the environment this was written in, and it isn't a
persistent process that can crash the same way mid-request the way
whisper.cpp/Piper can -- ChatClient's reconnect path is already covered
by mocks in test_service_reconnect.py). If you have LM Studio running,
you can verify the same way by hand: pause/kill `lms server` mid-reply
and confirm main.py's LM Studio branch waits and resumes instead of
crashing.

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

import main  # noqa: E402
from audio.echo_cancel import resample_linear  # noqa: E402
from audio.playback import decode_wav_bytes  # noqa: E402
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


test_whisper_real_crash_and_restart()
test_piper_real_binary_recovery()

print()
print(f"{sum(results)}/{len(results)} passed")
if not all(results):
    sys.exit(1)
