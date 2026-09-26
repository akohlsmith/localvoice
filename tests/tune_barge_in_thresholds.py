"""Diagnoses and tunes speak_with_barge_in's false-trigger thresholds
(barge_in.prob_threshold / consecutive_frames in config.yaml).

Problem this solves: with speakers and mic on the same machine, the mic
picks up the assistant's own voice as echo. Even after echo cancellation
(audio/echo_cancel.py) and a startup grace period (barge_in.grace_period_ms),
a sufficiently long or acoustically varied reply can occasionally still
cross the barge-in confidence threshold for long enough to falsely
interrupt itself. This happened live on a real "explain the Monty Hall
paradox" reply (TEST_TEXT below), measured at 1/15 (~6.7%) false triggers
at the settings in place at the time. Guessing new threshold values isn't
reliable -- this script instead:

  1. Synthesizes TEST_TEXT via Piper (real reply text known to have
     triggered a false interruption, kept verbatim including its
     punctuation/paragraph breaks -- those turned out not to matter, but
     that was verified, not assumed).
  2. Runs --trials independent live trials of the real
     speak_with_barge_in echo-cancellation + VAD-scoring pipeline
     against real mic input and real speaker playback (not synthetic or
     offline data), logging every post-echo-cancellation chunk's VAD
     probability without acting on it -- so the reply always plays to
     completion regardless of what the real thresholds would have done.
     This makes each trial informative for every candidate threshold at
     once, rather than needing a separate live run per candidate.
  3. Sweeps candidate (prob_threshold, consecutive_frames) pairs against
     the logged traces and reports which combinations would have caught
     zero of the false triggers actually observed, so new config.yaml
     values come from measured headroom instead of a guess.

Needs a running whisper.cpp server? No. Needs LM Studio? No. Just Piper
synthesis plus a real mic and real speaker output -- run it with the
room/mic setup you actually care about tuning for.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/tune_barge_in_thresholds.py
    venv/bin/python tests/tune_barge_in_thresholds.py --trials 30
    venv/bin/python tests/tune_barge_in_thresholds.py --text "some other reply"
"""

import argparse
import os
import pickle
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import main  # noqa: E402
from audio import playback  # noqa: E402
from audio.capture import MicStream  # noqa: E402
from audio.echo_cancel import EchoCanceller, resample_linear  # noqa: E402
from audio.vad import SileroVAD, TurnDetector  # noqa: E402
from tts.tts_client import PiperTTS  # noqa: E402

TEST_TEXT = (
    "Imagine you're on a game show with three doors. One has a car, and "
    "the others have goats.\n\nYou pick a door, and the host—who knows "
    "what's behind them—opens one of the *other* doors to reveal a "
    "goat. Now, the big question is: should you switch your choice?\n\n"
    "Surprisingly, you should! If you stay with your original pick, you "
    "only have a one-third chance of winning. But if you switch, your "
    "odds jump to two-thirds. It sounds counterintuitive, but switching "
    "essentially bets that your first choice was wrong, which it "
    "usually is!\n"
)

OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")
SAMPLE_RATE = 16000


def collect_traces(wav_bytes, n_trials: int):
    cfg = main.load_config()
    sample_rate = 16000
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]

    all_traces = []
    for trial in range(n_trials):
        mic = MicStream(sample_rate=sample_rate, chunk_samples=chunk, channels=1, device=None)
        vad = SileroVAD(sample_rate=sample_rate)
        turn_detector = TurnDetector(
            vad,
            sample_rate=sample_rate,
            chunk_samples=chunk,
            prob_threshold=cfg["vad"]["speech_prob_threshold"],
            silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
            min_speech_ms=cfg["vad"]["min_speech_ms"],
        )
        ec = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg["expected_delay_ms"],
            filter_length_ms=echo_cfg["filter_length_ms"],
            mu=echo_cfg["mu"],
        )
        raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
        ref_audio = resample_linear(raw_ref, ref_sr, ec.sample_rate)
        ec.reset()

        mic.start()
        time.sleep(0.3)
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
            ref_slice = ref_audio[ref_pos : ref_pos + n]
            if len(ref_slice) < n:
                ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
            ref_pos += n
            ec.push_reference(ref_slice)
            scored_chunk = ec.cancel(mic_chunk)

            in_grace = chunk_index < grace_period_chunks
            chunk_index += 1
            prob = vad.speech_prob(scored_chunk)
            probs.append((chunk_index, in_grace, float(prob)))

        mic.stop()
        elapsed = time.time() - t0
        all_traces.append(probs)
        print(f"trial {trial}: {len(probs)} chunks logged, elapsed={elapsed:.2f}s", flush=True)

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


def analyze(traces):
    print()
    print("=== baseline (current config.yaml settings) ===")
    cfg = main.load_config()
    base_prob = cfg["barge_in"]["prob_threshold"]
    base_frames = cfg["barge_in"]["consecutive_frames"]
    triggers = [would_trigger(t, base_prob, base_frames) for t in traces]
    n_triggered = sum(1 for x in triggers if x is not None)
    print(f"{base_prob}/{base_frames}: {n_triggered}/{len(traces)} false-triggered "
          f"(at chunks {[x for x in triggers if x is not None]})")

    print()
    print("=== sweep ===")
    for prob_threshold in [0.9, 0.93, 0.95, 0.97]:
        for consecutive_frames in [16, 24, 32, 48, 64]:
            triggers = [would_trigger(t, prob_threshold, consecutive_frames) for t in traces]
            n_triggered = sum(1 for x in triggers if x is not None)
            ms = consecutive_frames * 32
            print(f"prob>={prob_threshold} frames>={consecutive_frames} (~{ms}ms): "
                  f"{n_triggered}/{len(traces)} triggered")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=20)
    parser.add_argument("--text", default=TEST_TEXT)
    parser.add_argument("--voice", default=None, help="Override tts.piper_model from config.yaml")
    parser.add_argument(
        "--confirm",
        action="store_true",
        help="Skip tracing/sweeping; instead run the real speak_with_barge_in "
        "(with actual stopping enabled, using config.yaml's current settings) "
        "--trials times and report the live pass rate. Use this to confirm a "
        "new prob_threshold/consecutive_frames chosen from a trace sweep "
        "actually holds up live, not just against the traces it was chosen from.",
    )
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=args.voice or cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(args.text)

    if args.confirm:
        confirm_live(wav_bytes, args.trials, cfg)
        return

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    traces = collect_traces(wav_bytes, args.trials)
    with open(os.path.join(OUTPUT_DIR, "traces.pkl"), "wb") as f:
        pickle.dump(traces, f)

    analyze(traces)


def confirm_live(wav_bytes, n_trials, cfg):
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    echo_cfg = cfg["barge_in"]["echo_cancel"]

    print(f"Confirming live at prob_threshold={cfg['barge_in']['prob_threshold']} "
          f"consecutive_frames={cfg['barge_in']['consecutive_frames']} "
          f"over {n_trials} real trials (actual stopping enabled)...")
    results = []
    for trial in range(n_trials):
        mic = MicStream(sample_rate=sample_rate, chunk_samples=chunk, channels=1, device=None)
        vad = SileroVAD(sample_rate=sample_rate)
        turn_detector = TurnDetector(
            vad,
            sample_rate=sample_rate,
            chunk_samples=chunk,
            prob_threshold=cfg["vad"]["speech_prob_threshold"],
            silence_hangover_ms=cfg["vad"]["silence_hangover_ms"],
            min_speech_ms=cfg["vad"]["min_speech_ms"],
        )
        ec = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg["expected_delay_ms"],
            filter_length_ms=echo_cfg["filter_length_ms"],
            mu=echo_cfg["mu"],
        )
        mic.start()
        time.sleep(0.3)
        t0 = time.time()
        interrupted = main.speak_with_barge_in(
            wav_bytes,
            mic,
            turn_detector,
            cfg["barge_in"]["prob_threshold"],
            cfg["barge_in"]["consecutive_frames"],
            ec,
            grace_period_chunks,
            None,
        )
        elapsed = time.time() - t0
        mic.stop()
        status = "FAIL (false-triggered)" if interrupted else "PASS"
        print(f"trial {trial}: {status} (elapsed={elapsed:.2f}s)", flush=True)
        results.append(not interrupted)

    print()
    print(f"{sum(results)}/{len(results)} passed")


if __name__ == "__main__":
    main_cli()
