// Phone UI client. Talks to web/server.py's WebSocket endpoint at
// /ws?token=... (same token this page itself was loaded with) and
// audio/phone_stream.py's PhoneMicStream on the other end.
//
// Protocol (see audio/phone_stream.py for the server side):
//   phone -> server: {"type":"hello","sample_rate":N} once on connect,
//     {"type":"enable"|"disable"} (informational), {"type":"stop"},
//     {"type":"playback_finished"}; binary frames = raw little-endian
//     Float32 mono PCM at the declared native sample rate.
//   server -> phone: binary frames = a WAV file (Piper's TTS output,
//     decodeAudioData() can play it directly).
(() => {
  const statusEl = document.getElementById("status");
  const enableBtn = document.getElementById("enableBtn");
  const stopBtn = document.getElementById("stopBtn");

  const BATCH_MS = 20; // how much mic audio to accumulate before sending one WS frame

  let ws = null;
  let audioCtx = null;
  let workletNode = null;
  let micStream = null;
  let enabled = false;

  let sendBuffer = [];
  let sendBufferLength = 0;
  let batchSamples = 0;

  let currentSource = null;
  let stoppedManually = false;

  function setStatus(text) {
    statusEl.textContent = text;
  }

  function wsUrl() {
    return `wss://${window.location.host}/ws${window.location.search}`;
  }

  function ensureSocket() {
    if (ws && (ws.readyState === WebSocket.OPEN || ws.readyState === WebSocket.CONNECTING)) {
      return Promise.resolve(ws);
    }
    return new Promise((resolve, reject) => {
      const socket = new WebSocket(wsUrl());
      socket.binaryType = "arraybuffer";
      socket.onopen = () => {
        ws = socket;
        resolve(socket);
      };
      socket.onmessage = (event) => handleServerMessage(event.data);
      socket.onerror = () => setStatus("Connection error.");
      socket.onclose = () => {
        setStatus("Disconnected.");
        setEnabled(false);
        ws = null;
      };
      setTimeout(() => {
        if (socket.readyState !== WebSocket.OPEN) reject(new Error("connect timeout"));
      }, 8000);
    });
  }

  function sendJson(obj) {
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(JSON.stringify(obj));
  }

  async function handleServerMessage(data) {
    if (typeof data === "string") return; // no server->phone control messages used in v1

    stoppedManually = false;
    setStatus("Speaking...");
    let audioBuffer;
    try {
      audioBuffer = await audioCtx.decodeAudioData(data);
    } catch (err) {
      setStatus(`Playback error: ${err.message}`);
      return;
    }
    const source = audioCtx.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(audioCtx.destination);
    currentSource = source;
    source.onended = () => {
      if (currentSource === source) currentSource = null;
      if (!stoppedManually) {
        sendJson({ type: "playback_finished" });
        setStatus(enabled ? "Listening..." : "Ready -- tap Enable");
      }
    };
    source.start();
  }

  function flushSendBuffer() {
    if (sendBufferLength === 0) return;
    const merged = new Float32Array(sendBufferLength);
    let offset = 0;
    for (const chunk of sendBuffer) {
      merged.set(chunk, offset);
      offset += chunk.length;
    }
    sendBuffer = [];
    sendBufferLength = 0;
    if (ws && ws.readyState === WebSocket.OPEN) ws.send(merged.buffer);
  }

  function onWorkletData(samples) {
    sendBuffer.push(samples);
    sendBufferLength += samples.length;
    if (sendBufferLength >= batchSamples) flushSendBuffer();
  }

  async function enableMic() {
    await ensureSocket();
    micStream = await navigator.mediaDevices.getUserMedia({
      audio: { echoCancellation: true, noiseSuppression: true, autoGainControl: true },
    });

    audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    batchSamples = Math.round(audioCtx.sampleRate * (BATCH_MS / 1000));
    await audioCtx.audioWorklet.addModule("/pcm-worklet.js");

    sendJson({ type: "hello", sample_rate: audioCtx.sampleRate });

    const source = audioCtx.createMediaStreamSource(micStream);
    workletNode = new AudioWorkletNode(audioCtx, "pcm-capture-processor");
    workletNode.port.onmessage = (event) => onWorkletData(event.data);
    source.connect(workletNode);
    // workletNode is deliberately never connected to audioCtx.destination
    // -- we only want its captured samples, never mic passthrough to
    // the speaker (that would just be feedback).

    sendJson({ type: "enable" });
    setEnabled(true);
    setStatus("Listening...");
  }

  function disableMic() {
    sendJson({ type: "disable" });
    if (micStream) {
      micStream.getTracks().forEach((track) => track.stop());
      micStream = null;
    }
    if (workletNode) {
      workletNode.port.onmessage = null;
      workletNode.disconnect();
      workletNode = null;
    }
    sendBuffer = [];
    sendBufferLength = 0;
    setEnabled(false);
    setStatus("Ready -- tap Enable");
  }

  function setEnabled(value) {
    enabled = value;
    enableBtn.textContent = value ? "Listening... (tap to disable)" : "Enable";
    enableBtn.classList.toggle("active", value);
  }

  enableBtn.addEventListener("click", async () => {
    try {
      if (enabled) {
        disableMic();
      } else {
        setStatus("Requesting microphone...");
        await enableMic();
      }
    } catch (err) {
      setStatus(`Error: ${err.message}`);
      setEnabled(false);
    }
  });

  stopBtn.addEventListener("click", () => {
    stoppedManually = true;
    if (currentSource) {
      try {
        currentSource.stop();
      } catch (err) {
        // already stopped/ended -- harmless
      }
      currentSource = null;
    }
    sendJson({ type: "stop" });
    setStatus(enabled ? "Listening..." : "Ready -- tap Enable");
  });

  setStatus("Ready -- tap Enable");
})();
