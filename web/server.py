"""HTTPS + WebSocket transport for the phone UI (audio/phone_stream.py).

iOS only grants microphone access (getUserMedia) in a secure context --
https:// or http://localhost -- and there is no user-facing override
for that on iOS the way desktop Chrome has one (chrome://flags). A
phone on the LAN hitting the Mac's IP address is neither, so this must
speak HTTPS. Since there's no real domain to get a CA-signed cert for,
this generates a self-signed one on first run (via the system `openssl`
binary -- no new pip dependency, same "shell out to an external tool"
pattern stt/whisper_client.py and tts/tts_client.py already use) and
caches it under phone_ui.cert_dir. The phone's browser will show a
"this connection is not private" warning the first time -- there is no
way around that for a self-signed cert, and it only needs accepting
once per device.

Everything (the static page, app.js, the worklet, and the WebSocket
endpoint) is served from one port via one `websockets` server:
process_request() intercepts plain HTTP GETs and serves web/static/*
directly, and returns None (letting the handshake proceed) only for the
WebSocket upgrade at /ws -- so no second HTTP server/port is needed.

Runs its own asyncio event loop on a dedicated background thread --
this is the only thread in the codebase apart from MicStream's existing
play-lock. main.py's own loop stays fully synchronous; PhoneMicStream
bridges the two with asyncio.run_coroutine_threadsafe().
"""

import asyncio
import mimetypes
import os
import secrets
import ssl
import subprocess
import threading
import urllib.parse

from websockets.asyncio.server import serve
from websockets.http11 import Headers, Response

STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")


def ensure_self_signed_cert(cert_dir: str) -> tuple[str, str]:
    """Returns (cert_path, key_path), generating a self-signed cert/key
    into cert_dir on first call if they don't already exist there.
    Reused across restarts so the phone doesn't have to re-accept the
    "not private" warning every single time."""
    cert_dir = os.path.expanduser(cert_dir)
    os.makedirs(cert_dir, exist_ok=True)
    cert_path = os.path.join(cert_dir, "cert.pem")
    key_path = os.path.join(cert_dir, "key.pem")
    if os.path.exists(cert_path) and os.path.exists(key_path):
        return cert_path, key_path

    try:
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", key_path, "-out", cert_path,
                "-days", "3650", "-nodes", "-subj", "/CN=localvoice-phone-ui",
            ],
            check=True,
            capture_output=True,
        )
    except FileNotFoundError as exc:
        raise RuntimeError(
            "openssl not found -- required once to generate a self-signed cert for phone_ui. "
            "Install it (e.g. `brew install openssl`) and restart."
        ) from exc
    except subprocess.CalledProcessError as exc:
        stderr = exc.stderr.decode("utf-8", errors="replace") if exc.stderr else ""
        raise RuntimeError(f"openssl failed to generate a self-signed cert: {stderr}") from exc
    return cert_path, key_path


def generate_token() -> str:
    return secrets.token_urlsafe(16)


class PhoneUIServer:
    """Owns the HTTPS/WebSocket transport. `on_connection(connection)` is
    an async callable invoked once per accepted WebSocket connection --
    audio/phone_stream.py's PhoneMicStream supplies this and owns all
    the actual audio/control logic; this class only knows about TLS,
    routing, and thread/loop lifecycle."""

    def __init__(self, host: str, port: int, cert_dir: str, auth_token: str, on_connection):
        self.host = host
        self.port = port
        self.cert_dir = cert_dir
        self.auth_token = auth_token
        self._on_connection = on_connection

        self._loop = None
        self._thread = None
        self._ws_server = None
        self._ready = threading.Event()
        self._ready_error = None

    def start(self):
        """Starts the background thread and blocks until the server is
        actually listening (or raises if it failed to start), mirroring
        MicStream.start()'s "ready once this returns" contract."""
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()
        self._ready.wait()
        if self._ready_error is not None:
            raise self._ready_error

    def stop(self):
        if self._loop is None:
            return
        fut = asyncio.run_coroutine_threadsafe(self._shutdown(), self._loop)
        # Deliberately not stopping the loop from inside _shutdown() itself:
        # run_coroutine_threadsafe delivers this Future's result via a
        # done-callback that needs one more loop iteration to run. Calling
        # loop.stop() as _shutdown()'s last line races that callback --
        # the work genuinely finishes and the loop genuinely exits, but
        # fut.result() below can time out anyway because the callback that
        # would have set its result never got that extra iteration.
        # Confirmed directly: stopping the loop here instead, strictly
        # after fut.result() returns, is reliable.
        fut.result(timeout=5)
        self._loop.call_soon_threadsafe(self._loop.stop)
        self._thread.join(timeout=5)
        self._loop = None

    def run_coroutine(self, coro):
        """Schedules coro on the server's event loop from any other
        thread (i.e. main.py's synchronous loop). Returns a
        concurrent.futures.Future; callers may ignore it (fire-and-forget)
        or call .result() to block for the outcome."""
        return asyncio.run_coroutine_threadsafe(coro, self._loop)

    def _run(self):
        self._loop = asyncio.new_event_loop()
        asyncio.set_event_loop(self._loop)
        try:
            self._loop.run_until_complete(self._serve())
        except Exception as exc:  # noqa: BLE001 -- surfaced to start() via _ready_error
            self._ready_error = exc
            self._ready.set()
            return
        self._loop.run_forever()

    async def _serve(self):
        cert_path, key_path = ensure_self_signed_cert(self.cert_dir)
        ssl_context = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        ssl_context.load_cert_chain(cert_path, key_path)
        self._ws_server = await serve(
            self._ws_handler,
            self.host,
            self.port,
            ssl=ssl_context,
            process_request=self._process_request,
        )
        self._ready.set()

    async def _shutdown(self):
        if self._ws_server is not None:
            self._ws_server.close()
            await self._ws_server.wait_closed()

    async def _ws_handler(self, connection):
        await self._on_connection(connection)

    def _process_request(self, connection, request):
        parsed = urllib.parse.urlsplit(request.path)
        if self.auth_token:
            token = urllib.parse.parse_qs(parsed.query).get("token", [None])[0]
            if token != self.auth_token:
                body = b"Forbidden -- missing or incorrect ?token="
                return Response(403, "Forbidden", Headers([("Content-Type", "text/plain")]), body)

        if parsed.path == "/ws":
            return None  # let the WebSocket handshake proceed to _ws_handler

        return self._serve_static(parsed.path)

    def _serve_static(self, path: str):
        rel_path = "index.html" if path in ("", "/") else path.lstrip("/")
        full_path = os.path.normpath(os.path.join(STATIC_DIR, rel_path))
        if not full_path.startswith(STATIC_DIR) or not os.path.isfile(full_path):
            return Response(404, "Not Found", Headers(), b"not found")

        with open(full_path, "rb") as f:
            body = f.read()
        content_type = mimetypes.guess_type(full_path)[0] or "application/octet-stream"
        headers = Headers([("Content-Type", content_type), ("Content-Length", str(len(body)))])
        return Response(200, "OK", headers, body)
