"""Regression test for audio/phone_stream.py's PhoneMicStream and
web/server.py's PhoneUIServer -- the server-side half of the phone_ui
feature (an HTTPS/WebSocket listener that lets a phone's mic/speaker
stand in for the Mac's own, with an Enable and a Stop button on the
phone's page).

Unlike a real iPhone, this needs no physical device or browser: it
drives a real `websockets` client (a "fake phone") against the real
server -- real TLS handshake with the real self-signed cert, real token
gating, real WebSocket messages -- so it exercises the exact same code
path a genuine phone would, just without the actual Safari/getUserMedia
side, which nothing in this environment can drive. See PLAN.md's
phone_ui entry for what still needs a human to verify on a real device.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/test_phone_stream.py
"""

import asyncio
import json
import os
import ssl
import sys
import tempfile
import wave
import io

import numpy as np
from websockets.asyncio.client import connect as ws_connect
from websockets.exceptions import InvalidStatus

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from audio.phone_stream import PhoneMicStream  # noqa: E402
from web.server import ensure_self_signed_cert  # noqa: E402

TEST_PORT = 8788
TOKEN = "test-token-123"

results = []


def check(name, condition):
    status = "PASS" if condition else "FAIL"
    print(f"{name}: {status}")
    results.append(bool(condition))


def client_ssl_context():
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_CLIENT)
    ctx.check_hostname = False
    ctx.verify_mode = ssl.CERT_NONE
    return ctx


async def get_next_chunk(mic, timeout_s=5):
    gen = mic.chunks()
    return await asyncio.wait_for(asyncio.to_thread(next, gen), timeout=timeout_s)


async def test_cert_generation_and_reuse(cert_dir):
    cert1, key1 = ensure_self_signed_cert(cert_dir)
    mtime1 = os.path.getmtime(cert1)
    cert2, key2 = ensure_self_signed_cert(cert_dir)
    mtime2 = os.path.getmtime(cert2)
    check("ensure_self_signed_cert creates cert+key files", os.path.isfile(cert1) and os.path.isfile(key1))
    check("ensure_self_signed_cert reuses an existing cert rather than regenerating", cert1 == cert2 and mtime1 == mtime2)


async def test_wrong_token_rejected():
    try:
        async with ws_connect(
            f"wss://127.0.0.1:{TEST_PORT}/ws?token=wrong", ssl=client_ssl_context(), open_timeout=5
        ):
            rejected = False
    except InvalidStatus as exc:
        rejected = exc.response.status_code == 403
    check("wrong token is rejected with 403", rejected)


async def test_audio_resampling_and_chunking(mic):
    async with ws_connect(
        f"wss://127.0.0.1:{TEST_PORT}/ws?token={TOKEN}", ssl=client_ssl_context(), open_timeout=5
    ) as ws:
        native_rate = 48000
        await ws.send(json.dumps({"type": "hello", "sample_rate": native_rate}))
        await ws.send(json.dumps({"type": "enable"}))

        # Enough native-rate samples to produce at least one full
        # 512-sample window at 16kHz after resampling (512/16000 s of
        # audio -> that many samples at 48000Hz, plus margin).
        n_native = int(512 / mic.sample_rate * native_rate) + 2000
        synthetic = (0.1 * np.sin(np.linspace(0, 40 * np.pi, n_native))).astype(np.float32)
        # Send in a few frames, like the real client's batching does.
        for i in range(0, len(synthetic), 2000):
            await ws.send(synthetic[i : i + 2000].tobytes())

        chunk = await get_next_chunk(mic)
        check("PhoneMicStream.chunks() yields the right-sized 16kHz window", len(chunk) == mic.chunk_samples)
        check("yielded chunk is float32", chunk.dtype == np.float32)

        await ws.send(json.dumps({"type": "disable"}))


async def test_play_and_is_playing(mic):
    async with ws_connect(
        f"wss://127.0.0.1:{TEST_PORT}/ws?token={TOKEN}", ssl=client_ssl_context(), open_timeout=5
    ) as ws:
        await ws.send(json.dumps({"type": "hello", "sample_rate": 16000}))

        audio = (0.2 * np.sin(np.linspace(0, 20 * np.pi, 1600))).astype(np.float32)
        mic.play(audio)
        check("is_playing() is True right after play()", mic.is_playing())

        raw_wav = await asyncio.wait_for(ws.recv(), timeout=5)
        check("phone receives real WAV bytes (RIFF header)", raw_wav[:4] == b"RIFF")
        with wave.open(io.BytesIO(raw_wav), "rb") as wf:
            check("received WAV matches mic.sample_rate", wf.getframerate() == mic.sample_rate)
            check("received WAV has the right sample count", wf.getnframes() == len(audio))

        # Simulate the phone's AudioBufferSourceNode finishing naturally.
        await ws.send(json.dumps({"type": "playback_finished"}))
        await asyncio.sleep(0.2)
        check("is_playing() goes False after a playback_finished ack", not mic.is_playing())

        # Now the manual Stop button, mid-(simulated)-playback.
        mic.play(audio)
        await asyncio.wait_for(ws.recv(), timeout=5)  # drain the WAV frame
        await ws.send(json.dumps({"type": "stop"}))
        await asyncio.sleep(0.2)
        check("is_playing() goes False after a stop message", not mic.is_playing())
        check("consume_stop_signal() is True exactly once after Stop", mic.consume_stop_signal() and not mic.consume_stop_signal())


async def test_play_with_no_phone_connected(mic):
    # No client connected at this point in the script (each previous
    # test's `async with` has closed its connection by now).
    await asyncio.sleep(0.2)
    audio = np.zeros(160, dtype=np.float32)
    mic.play(audio)
    check("play() with nobody connected does not leave is_playing() stuck True", not mic.is_playing())


async def test_heartbeat_keeps_chunks_flowing(mic):
    # With no phone streaming real audio, chunks() must still not hang
    # forever -- this is what stops speak_with_barge_in's loop from
    # blocking indefinitely when the phone is idle/disconnected. Drain
    # first: earlier tests' synthetic audio may have queued more than
    # one 512-sample window, and only one was consumed there -- without
    # draining, this would see that real leftover chunk instead of a
    # fresh heartbeat silence one.
    mic.drain()
    chunk = await get_next_chunk(mic, timeout_s=3)
    check("chunks() keeps yielding (heartbeat silence) with no real audio flowing", chunk is not None)
    check("heartbeat chunks are silent", np.all(chunk == 0))


async def main():
    with tempfile.TemporaryDirectory() as cert_dir:
        await test_cert_generation_and_reuse(cert_dir)

        mic = PhoneMicStream(host="127.0.0.1", port=TEST_PORT, cert_dir=cert_dir, auth_token=TOKEN)
        mic.start()
        try:
            await test_wrong_token_rejected()
            await test_audio_resampling_and_chunking(mic)
            await test_play_and_is_playing(mic)
            await test_play_with_no_phone_connected(mic)
            await test_heartbeat_keeps_chunks_flowing(mic)
        finally:
            mic.stop()

    print()
    print(f"{sum(results)}/{len(results)} passed")
    if not all(results):
        sys.exit(1)


asyncio.run(main())
