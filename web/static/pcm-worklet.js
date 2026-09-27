// Runs on the audio rendering thread. Deliberately dumb: just hands raw
// mono Float32 samples up to the main thread every ~128-sample block,
// at whatever the device's native sample rate is. No resampling here --
// app.js batches these into slightly larger frames and sends them as-is;
// audio/phone_stream.py resamples to 16kHz server-side (reusing
// audio/echo_cancel.py's existing resample_linear), so this file stays
// tiny and has nothing device/version-specific to get wrong.
class PCMCaptureProcessor extends AudioWorkletProcessor {
  process(inputs) {
    const channelData = inputs[0] && inputs[0][0];
    if (channelData && channelData.length > 0) {
      // slice(): the underlying buffer is reused by the audio thread
      // between calls, so the posted copy must be independent.
      this.port.postMessage(channelData.slice(0));
    }
    return true; // keep the processor alive for the life of the node
  }
}

registerProcessor("pcm-capture-processor", PCMCaptureProcessor);
