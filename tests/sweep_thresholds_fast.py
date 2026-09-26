"""Fast trace-sweep variant of tune_barge_in_thresholds.py's
collect_traces/analyze: reuses ONE MicStream + SileroVAD + TurnDetector
across all trials instead of recreating them (which re-triggers
torch.hub.load() every trial -- observed to add ~150-180s of overhead
per trial, making a 15-trial sweep take over 20 minutes).

This is also more representative of real usage, not just faster:
main.py's actual loop creates its SileroVAD once for the whole
session and reuses it across every reply, so a single long-running
model instance across trials matches production better than a fresh
one per trial. Only the EchoCanceller is reset between trials (each
reply's echo path should start unconverged, matching a real fresh
reply).

Built specifically to re-derive barge_in.prob_threshold/
consecutive_frames after fixing the grace-period VAD-starvation bug
(main.py: speech_prob() must be called every chunk, not skipped
during grace_period_chunks, to keep Silero's streaming state warm).
That fix restored real barge-in (previously it never fired at all)
but also reintroduced self-triggering on residual echo for at least
one reply text ("Let's talk through this slowly...", 2/3 false
triggers live) that the original Monty Hall-only tuning never
exercised -- this sweeps candidate thresholds against fresh traces of
that specific text so the next choice is backed by data for the text
that actually broke, not just the one that happened to pass before.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/sweep_thresholds_fast.py --trials 10
    venv/bin/python tests/sweep_thresholds_fast.py --trials 10 --text "..."
"""

import argparse
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio import playback  # noqa: E402
from audio.capture import MicStream  # noqa: E402
from audio.echo_cancel import EchoCanceller, resample_linear  # noqa: E402
from audio.vad import SileroVAD  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

SAMPLE_RATE = 16000
DEFAULT_TEXT = (
    "Let's talk through this slowly, one step at a time, so there is "
    "plenty of room to interrupt whenever you want to. First, imagine "
    "a long, rambling explanation that just keeps going, sentence "
    "after sentence, the way a real answer sometimes does when there "
    "is a lot of ground to cover."
)


def collect_traces_fast(wav_bytes, n_trials, cfg):
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]

    mic = MicStream(sample_rate=sample_rate, chunk_samples=chunk, channels=1, device=None)
    vad = SileroVAD(sample_rate=sample_rate)  # created once, reused -- matches main.py's real lifecycle
    mic.start()
    time.sleep(0.3)

    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    ref_audio = resample_linear(raw_ref, ref_sr, sample_rate)

    all_traces = []
    for trial in range(n_trials):
        ec = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg["expected_delay_ms"],
            filter_length_ms=echo_cfg["filter_length_ms"],
            mu=echo_cfg["mu"],
        )
        mic.drain()
        playback.play_wav_bytes_async(wav_bytes)

        probs = []
        ref_pos = 0
        chunk_index = 0
        t0 = time.time()
        for mic_chunk in mic.chunks():
            if not playback.is_playing():
                break
            n = len(mic_chunk)
            ref_slice = ref_audio[ref_pos: ref_pos + n]
            if len(ref_slice) < n:
                ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
            ref_pos += n
            ec.push_reference(ref_slice)
            scored_chunk = ec.cancel(mic_chunk)

            # Matches the FIXED main.py: score every chunk, gate only the action.
            prob = vad.speech_prob(scored_chunk)
            in_grace = chunk_index < grace_period_chunks
            probs.append((chunk_index, in_grace, float(prob)))
            chunk_index += 1

        elapsed = time.time() - t0
        all_traces.append(probs)
        print(f"trial {trial}: {len(probs)} chunks logged, elapsed={elapsed:.2f}s", flush=True)

    mic.stop()
    return all_traces


def would_trigger(trace, prob_threshold, consecutive_frames):
    consecutive = 0
    for chunk_index, in_grace, prob in trace:
        if in_grace:
            consecutive = 0
            continue
        if prob >= prob_threshold:
            consecutive += 1
            if consecutive >= consecutive_frames:
                return chunk_index
        else:
            consecutive = 0
    return None


def analyze(traces, cfg):
    print()
    print("=== baseline (current config.yaml settings) ===")
    base_prob = cfg["barge_in"]["prob_threshold"]
    base_frames = cfg["barge_in"]["consecutive_frames"]
    triggers = [would_trigger(t, base_prob, base_frames) for t in traces]
    n_triggered = sum(1 for x in triggers if x is not None)
    print(f"{base_prob}/{base_frames}: {n_triggered}/{len(traces)} false-triggered "
          f"(at chunks {[x for x in triggers if x is not None]})")

    print()
    print("=== sweep ===")
    for prob_threshold in [0.9, 0.93, 0.95, 0.97, 0.98, 0.99]:
        for consecutive_frames in [16, 24, 32, 48, 64, 96]:
            triggers = [would_trigger(t, prob_threshold, consecutive_frames) for t in traces]
            n_triggered = sum(1 for x in triggers if x is not None)
            ms = consecutive_frames * 32
            print(f"prob>={prob_threshold} frames>={consecutive_frames} (~{ms}ms): "
                  f"{n_triggered}/{len(traces)} triggered")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--text", default=DEFAULT_TEXT)
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(args.text)

    traces = collect_traces_fast(wav_bytes, args.trials, cfg)
    analyze(traces, cfg)


if __name__ == "__main__":
    main_cli()
