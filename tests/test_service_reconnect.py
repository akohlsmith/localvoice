"""Regression test for the "whisper.cpp/Piper/LM Studio goes down mid-
session" gap flagged in PLAN.md's Known Gaps: until this fix, any of the
three external processes being unreachable (killed, crashed, or just not
started yet) raised an uncaught exception straight out of the mic loop
and killed the whole voice assistant, per the exact commit that closed
this gap.

Unlike the other scripts in this directory, this one needs no real mic,
speakers, or running servers -- it mocks requests/subprocess/the OpenAI
client to simulate each service being down and then coming back, so it
can run anywhere (including CI). It checks:

  1. WhisperClient.transcribe() retries a transient failure and only
     raises WhisperUnavailableError once retries are exhausted; and that
     WhisperClient.is_available() tracks reachability correctly.
  2. PiperTTS.synthesize() turns a missing binary, a hang, and a
     nonzero exit into PiperUnavailableError instead of leaking the raw
     subprocess exception; and that PiperTTS.is_available() agrees.
  3. ChatClient.ask() rolls the conversation history back to exactly
     its pre-call state when LM Studio is unreachable, so a retried
     ask() with the same text doesn't leave two consecutive user turns
     in history (which broke tool-call-loop models expecting strict
     user/assistant alternation).
  4. main.call_with_reconnect() retries the wrapped call once the
     service's is_available() flips back to True, rather than raising.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/test_service_reconnect.py
"""

import os
import subprocess
import sys
from unittest import mock

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import requests  # noqa: E402

import main  # noqa: E402
from llm.chat_client import ChatClient, LlmUnavailableError  # noqa: E402
from stt.whisper_client import WhisperClient, WhisperUnavailableError  # noqa: E402
from tts.tts_client import PiperTTS, PiperUnavailableError  # noqa: E402

results = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name}: {status}")
    results.append(condition)


def test_whisper_retry_then_unavailable():
    client = WhisperClient("http://localhost:8081", max_retries=2, retry_backoff_s=0)
    audio = np.zeros(1600, dtype=np.float32)

    # Two transient failures followed by success -- should retry through
    # them and return normally, not raise.
    call_count = {"n": 0}

    def flaky_post(*args, **kwargs):
        call_count["n"] += 1
        if call_count["n"] <= 2:
            raise requests.ConnectionError("refused")
        resp = mock.Mock()
        resp.raise_for_status.return_value = None
        resp.json.return_value = {"text": "hello"}
        return resp

    with mock.patch("stt.whisper_client.requests.post", side_effect=flaky_post):
        text = client.transcribe(audio)
    check("WhisperClient.transcribe retries transient failures then succeeds", text == "hello")

    # Permanently down -- should raise WhisperUnavailableError, not a
    # raw requests exception, after exactly max_retries+1 attempts.
    with mock.patch(
        "stt.whisper_client.requests.post", side_effect=requests.ConnectionError("refused")
    ) as mocked:
        try:
            client.transcribe(audio)
            raised = False
        except WhisperUnavailableError:
            raised = True
        except Exception:
            raised = False
    check("WhisperClient.transcribe raises WhisperUnavailableError once retries exhaust", raised)
    check("WhisperClient.transcribe attempted exactly max_retries+1 times", mocked.call_count == 3)

    with mock.patch("stt.whisper_client.requests.get", side_effect=requests.ConnectionError("refused")):
        down = client.is_available()
    with mock.patch("stt.whisper_client.requests.get", return_value=mock.Mock()):
        up = client.is_available()
    check("WhisperClient.is_available reflects down/up state", down is False and up is True)


def test_piper_failures():
    tts = PiperTTS(binary="piper", model_path="voices/fake.onnx")

    with mock.patch("tts.tts_client.subprocess.run", side_effect=FileNotFoundError()):
        try:
            tts.synthesize("hello")
            raised_missing = False
        except PiperUnavailableError:
            raised_missing = True
    check("PiperTTS.synthesize raises PiperUnavailableError for a missing binary", raised_missing)

    with mock.patch(
        "tts.tts_client.subprocess.run",
        side_effect=subprocess.TimeoutExpired(cmd="piper", timeout=30),
    ):
        try:
            tts.synthesize("hello")
            raised_timeout = False
        except PiperUnavailableError:
            raised_timeout = True
    check("PiperTTS.synthesize raises PiperUnavailableError on a hang/timeout", raised_timeout)

    with mock.patch(
        "tts.tts_client.subprocess.run",
        side_effect=subprocess.CalledProcessError(1, "piper", stderr=b"bad model"),
    ):
        try:
            tts.synthesize("hello")
            raised_error = False
        except PiperUnavailableError as exc:
            raised_error = "bad model" in str(exc)
    check("PiperTTS.synthesize raises PiperUnavailableError on nonzero exit, keeping stderr", raised_error)

    with mock.patch("tts.tts_client.subprocess.run", side_effect=FileNotFoundError()):
        down = tts.is_available()
    with mock.patch("tts.tts_client.subprocess.run", return_value=None):
        up = tts.is_available()
    check("PiperTTS.is_available reflects down/up state", down is False and up is True)


def test_chat_client_rolls_back_history_on_disconnect():
    import httpx2
    from openai import APIConnectionError

    chat = ChatClient(base_url="http://localhost:1", model="fake-model", context_limit=8192)
    history_before = list(chat.history)

    req = httpx2.Request("POST", "http://localhost:1/v1/chat/completions")
    with mock.patch.object(
        chat.client.chat.completions,
        "create",
        side_effect=APIConnectionError(message="refused", request=req),
    ):
        try:
            chat.ask("what city do I live in?")
            raised = False
        except LlmUnavailableError:
            raised = True
    check("ChatClient.ask raises LlmUnavailableError when LM Studio is unreachable", raised)
    check("ChatClient.ask leaves history exactly as it was before the failed call", chat.history == history_before)

    # Confirm a subsequent successful ask() isn't left with a dangling
    # duplicate user turn from the failed attempt.
    fake_message = mock.Mock(tool_calls=None, content="Springfield.")
    fake_response = mock.Mock(choices=[mock.Mock(message=fake_message)], usage=None)
    with mock.patch.object(chat.client.chat.completions, "create", return_value=fake_response):
        reply = chat.ask("what city do I live in?")
    check("ChatClient.ask succeeds normally on retry after reconnect", reply == "Springfield.")
    check(
        "History has exactly one user/assistant pair after the retried call",
        [m["role"] for m in chat.history] == ["user", "assistant"],
    )


def test_call_with_reconnect_resumes_after_service_returns():
    state = {"down_polls": 2, "attempts": 0}

    def flaky_call():
        state["attempts"] += 1
        if state["attempts"] == 1:
            raise WhisperUnavailableError("down")
        return "ok"

    def is_available():
        state["down_polls"] -= 1
        return state["down_polls"] <= 0

    with mock.patch("main.time.sleep"):
        result = main.call_with_reconnect(flaky_call, "whisper.cpp", is_available, poll_interval_s=0)
    check("call_with_reconnect retries once the service reports available again", result == "ok")
    check("call_with_reconnect polled is_available until it returned True", state["down_polls"] == 0)


test_whisper_retry_then_unavailable()
test_piper_failures()
test_chat_client_rolls_back_history_on_disconnect()
test_call_with_reconnect_resumes_after_service_returns()

print()
print(f"{sum(results)}/{len(results)} passed")
if not all(results):
    sys.exit(1)
