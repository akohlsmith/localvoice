"""Measures whether real user speech reliably clears barge-in's
prob_threshold/consecutive_frames (config.yaml's barge_in section)
while the assistant is actually talking -- the "true positive" side of
barge-in tuning that tune_barge_in_thresholds.py never measured.

Problem this solves: tune_barge_in_thresholds.py (and the 0.93
prob_threshold / 24 consecutive_frames chosen from it) was tuned
entirely against the *false positive* side -- letting a reply play to
completion untouched and logging how confident Silero got on leftover
echo of the reply itself. It never checked whether a real human
actually clears that same bar while genuinely talking over playback.
After shipping 0.93/24, live use found barge-in no longer interrupts
at all. Hypothesis: real speech, unlike the smooth echo trace of the
reply itself, naturally dips below any fixed probability threshold
between syllables/consonants/breaths, so demanding an unbroken
24-frame (~768ms) run at a stricter 0.93 confidence may never get
fully satisfied even when someone is clearly, continuously talking.

Unlike the other tests/ scripts here, this one needs a real person
reacting live to a prompt -- it can't be fully unattended. Run it in a
terminal you're watching: for each trial it plays TEST_TEXT via Piper,
waits past the current barge_in.grace_period_ms plus --speak-after
seconds, then prints "SPEAK NOW" -- talk continuously (a sentence or
two, normal volume, no need to shout) until playback stops or the
trial ends. It logs the full post-echo-cancellation VAD probability
trace during that window (same measurement tune_barge_in_thresholds.py
uses for the false-positive side) and reports, per trial, the longest
sustained run at several candidate probability thresholds -- so the
final choice is backed by a real interruption attempt, not the
reply's own echo.

Usage (from anywhere; it locates the project root itself):
    venv/bin/python tests/validate_true_barge_in.py
    venv/bin/python tests/validate_true_barge_in.py --trials 8 --speak-after 1.5
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

TEST_TEXT = (
    "Here is a reasonably long reply, so there is plenty of time to "
    "interrupt it. The quick brown fox jumps over the lazy dog, and "
    "then does it again, and again, while we wait and see whether you "
    "can actually cut it off by talking over it right now."
)

CANDIDATE_THRESHOLDS = [0.8, 0.85, 0.9, 0.93, 0.95]


def run_trial(wav_bytes, cfg, trial_num, speak_after_s):
    sample_rate = SAMPLE_RATE
    chunk = main.CHUNK_SAMPLES
    chunk_ms = 1000 * chunk / sample_rate
    echo_cfg = cfg["barge_in"]["echo_cancel"]
    grace_period_chunks = int(cfg["barge_in"]["grace_period_ms"] / chunk_ms)
    speak_at_chunk = grace_period_chunks + int(1000 * speak_after_s / chunk_ms)

    mic = MicStream(sample_rate=sample_rate, chunk_samples=chunk, channels=1, device=None)
    vad = SileroVAD(sample_rate=sample_rate)
    ec = None
    if echo_cfg.get("enabled", True):
        ec = EchoCanceller(
            sample_rate=sample_rate,
            expected_delay_ms=echo_cfg["expected_delay_ms"],
            filter_length_ms=echo_cfg["filter_length_ms"],
            mu=echo_cfg["mu"],
        )

    raw_ref, ref_sr = playback.decode_wav_bytes(wav_bytes)
    ref_audio = resample_linear(raw_ref, ref_sr, sample_rate)
    if ec is not None:
        ec.reset()

    mic.start()
    time.sleep(0.3)
    mic.drain()
    playback.play_wav_bytes_async(wav_bytes)

    wait_s = grace_period_chunks * chunk_ms / 1000 + speak_after_s
    print(f"trial {trial_num}: playing reply, speak in ~{wait_s:.1f}s...", flush=True)

    probs = []
    ref_pos = 0
    chunk_index = 0
    prompted = False
    for mic_chunk in mic.chunks():
        if not playback.is_playing():
            break
        n = len(mic_chunk)
        ref_slice = ref_audio[ref_pos: ref_pos + n]
        if len(ref_slice) < n:
            ref_slice = np.concatenate([ref_slice, np.zeros(n - len(ref_slice), dtype=np.float32)])
        ref_pos += n

        if ec is not None:
            ec.push_reference(ref_slice)
            scored_chunk = ec.cancel(mic_chunk)
        else:
            scored_chunk = mic_chunk

        if not prompted and chunk_index >= speak_at_chunk:
            print("  >>> SPEAK NOW (talk continuously) <<<", flush=True)
            prompted = True

        in_grace = chunk_index < grace_period_chunks
        prob = vad.speech_prob(scored_chunk)
        probs.append((chunk_index, in_grace, float(prob)))
        chunk_index += 1

    mic.stop()
    playback.stop()
    print(f"  {len(probs)} chunks logged", flush=True)
    return probs


def longest_run(trace, prob_threshold):
    best = 0
    cur = 0
    for chunk_index, in_grace, prob in trace:
        if in_grace:
            cur = 0
            continue
        if prob >= prob_threshold:
            cur += 1
            best = max(best, cur)
        else:
            cur = 0
    return best


def analyze(traces, cfg):
    print()
    print("=== longest sustained real-speech run per trial, by candidate threshold ===")
    for prob_threshold in CANDIDATE_THRESHOLDS:
        runs = [longest_run(t, prob_threshold) for t in traces]
        ms = [r * 32 for r in runs]
        print(f"prob>={prob_threshold}: frames={runs}  (~ms)={ms}")

    base_prob = cfg["barge_in"]["prob_threshold"]
    base_frames = cfg["barge_in"]["consecutive_frames"]
    hits = [longest_run(t, base_prob) >= base_frames for t in traces]
    print()
    print(f"=== current config.yaml settings ({base_prob}/{base_frames}) ===")
    print(f"{sum(hits)}/{len(hits)} trials would have triggered barge-in")

    print()
    print("=== frame counts needed at each threshold to catch every trial above ===")
    for prob_threshold in CANDIDATE_THRESHOLDS:
        runs = [longest_run(t, prob_threshold) for t in traces]
        min_run = min(runs) if runs else 0
        print(f"prob>={prob_threshold}: worst trial sustained only {min_run} frames "
              f"(~{min_run * 32}ms) -- consecutive_frames must be <= that to catch every trial")


def main_cli():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--text", default=TEST_TEXT)
    parser.add_argument("--speak-after", type=float, default=1.0,
                         help="Seconds after grace_period_ms ends before prompting you to speak")
    args = parser.parse_args()

    cfg = main.load_config()
    tts = PiperTTS(
        binary=cfg["tts"]["piper_binary"],
        model_path=cfg["tts"]["piper_model"],
        config_path=cfg["tts"].get("piper_model_config"),
        speaker=cfg["tts"].get("speaker"),
    )
    wav_bytes = tts.synthesize(args.text)

    traces = []
    for trial in range(args.trials):
        traces.append(run_trial(wav_bytes, cfg, trial + 1, args.speak_after))
        time.sleep(1.0)

    analyze(traces, cfg)


if __name__ == "__main__":
    main_cli()
