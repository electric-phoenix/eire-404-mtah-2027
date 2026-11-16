"""Per-frame sound measures from the tactical video's audio track, for finding the referee's whistle.

    python -m pipeline.audio            # writes data/processed/audio_frames.npz

WHY (plain words)
-----------------
Dead ball is our weakest class. Many missed stoppages are fouls and injuries: the ball is still
visible and there is no set-piece tag, so the tracking barely changes. But the referee WHISTLES.
The tactical video carries the stadium sound, and video frame i is row i of the submission, so
second t of the sound is frame round(t * 25). No syncing is needed.

STEPS
-----
1. ffmpeg turns the video's sound into a mono 16 kHz WAV (data/processed/audio/, gitignored).
2. For each video frame, a 2048-sample window (128 ms) centred on the frame's 40 ms slice is
   Fourier-transformed (Hann window). 128 ms gives ~8 Hz pitch resolution, enough to separate
   the whistle's two notes.
3. Measures saved per frame (one value per video frame, 199,319):
     rms_db     overall loudness (only used by the rejected sound-features experiment)
     low_db     strongest pitch in 3.65-4.0 kHz above the median level of 2.5-6 kHz
     high_db    strongest pitch in 4.15-4.4 kHz above the same median
     whistle_db min(low_db, high_db): BOTH notes must be present

THE REFEREE'S FINGERPRINT (found on this match's kickoffs, where a whistle is certain)
------------------------------------------------------------------------------------
This referee's whistle sounds two notes at once: one near 3.7-3.9 kHz and one about 450 Hz higher
(4.2-4.3 kHz), each 20-44 dB above the background. Crowd noise is loud below 3.5 kHz but almost
never has two narrow peaks in exactly those places. So `whistle_db` (the weaker of the two notes)
is the measure we use. The first try, "any single sharp pitch in 2-4.5 kHz", caught too much crowd
sound and was removed (it was also the slowest part of this step).
Nothing here looks at labels or the ground truth. All of it is computed locally.
"""
import os
import subprocess
import sys
import wave

import numpy as np
from scipy import fft as sfft

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
PROC = os.path.join(ROOT, "data", "processed")
VIDEO = os.path.join(ROOT, "data", "tactical", "BAR-ATH_tactical-002.mp4")
WAV = os.path.join(PROC, "audio", "tactical_16k.wav")
OUT = os.path.join(PROC, "audio_frames.npz")

SR = 16000            # samples per second after extraction
FPS = 25              # video frames per second
HOP = SR // FPS       # 640 samples = one video frame
N_FRAMES = 199319
FFMPEG = os.environ.get("FFMPEG", "ffmpeg")


def extract_wav():
    """Video sound -> mono 16 kHz 16-bit WAV (only if not already there)."""
    if os.path.exists(WAV):
        return
    os.makedirs(os.path.dirname(WAV), exist_ok=True)
    subprocess.run([FFMPEG, "-hide_banner", "-loglevel", "error", "-y", "-i", VIDEO, "-vn",
                    "-ac", "1", "-ar", str(SR), "-c:a", "pcm_s16le", WAV], check=True)


def read_wav():
    with wave.open(WAV, "rb") as w:
        assert w.getframerate() == SR and w.getnchannels() == 1
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


WIN2 = 2048                    # 128 ms window: ~8 Hz bins, enough to separate the two notes
LOW = (3650.0, 4000.0)
HIGH = (4150.0, 4400.0)
REF = (2500.0, 6000.0)


def whistle_measures(x, chunk=20000):
    """rms_db, low_db, high_db, whistle_db per video frame (see "the referee's fingerprint").

    Efficiency: only the 2.5-6 kHz bins are converted to decibels (the rest are never used), and
    the Fourier transforms run on all CPU cores (scipy.fft, workers=-1) in float64, which gives the
    same numbers as a single-core numpy transform.
    """
    freqs = np.fft.rfftfreq(WIN2, 1.0 / SR)
    ref = (freqs >= REF[0]) & (freqs <= REF[1])
    lo = ((freqs >= LOW[0]) & (freqs <= LOW[1]))[ref]       # positions inside the ref band
    hi = ((freqs >= HIGH[0]) & (freqs <= HIGH[1]))[ref]
    win = np.hanning(WIN2).astype(np.float32)
    # The sound is ~70 ms shorter than the video; pad the end with silence so every frame has a window.
    need = N_FRAMES * HOP + WIN2
    pad = np.pad(x, (WIN2 // 2, max(WIN2, need - len(x))))
    view = np.lib.stride_tricks.sliding_window_view(pad, WIN2)
    out = {k: np.zeros(N_FRAMES, np.float32) for k in ("rms_db", "low_db", "high_db")}
    for a in range(0, N_FRAMES, chunk):
        idx = np.arange(a, min(a + chunk, N_FRAMES))
        seg = (view[idx * HOP + HOP // 2] * win).astype(np.float64)   # frame i, window centred
        power = np.abs(sfft.rfft(seg, axis=1, workers=-1)) ** 2 + 1e-12
        out["rms_db"][idx] = 10 * np.log10(power.mean(axis=1))
        db = 10 * np.log10(power[:, ref])
        base = np.median(db, axis=1)
        out["low_db"][idx] = db[:, lo].max(axis=1) - base
        out["high_db"][idx] = db[:, hi].max(axis=1) - base
    out["whistle_db"] = np.minimum(out["low_db"], out["high_db"])
    return out


AUDIO_FEATURES = ["wh_now", "wh_max_25", "wh_max_125", "since_whistle", "rms_25", "rms_rise"]


def causal_features(m=None, shift=0):
    """Sound features for the classifier, using only the past (like every other feature).

    In plain words, for every frame:
      wh_now        the whistle score right now
      wh_max_25     the loudest whistle score in the last 1 s
      wh_max_125    ... in the last 5 s
      since_whistle frames since the last whistle (score >= 20 for 2 frames), capped at 250 (10 s)
      rms_25        how loud the stadium is (mean over the last 1 s)
      rms_rise      loudness now minus loudness over the last 10 s: the crowd getting excited
                    (attacks, chances) or going quiet (stoppages)
    The analysis window looks up to 64 ms ahead, so every value is moved 2 frames later to keep
    the features strictly causal. `shift` rolls the whole sound track in time: used only by the
    placebo test, which checks that any gain comes from the real timing of the sound.
    """
    import pandas as pd
    m = load() if m is None else m
    w = np.roll(m["whistle_db"], shift + 2); w[:2] = w[2]
    rms = np.roll(m["rms_db"], shift + 2); rms[:2] = rms[2]
    ws, rs = pd.Series(w), pd.Series(rms)
    hit = w >= 20
    onset = hit & np.r_[False, hit[:-1]] & ~np.r_[False, False, hit[:-2]]   # 2nd frame of a run
    last = np.where(onset, np.arange(len(w)), -10 ** 9)
    last = np.maximum.accumulate(last)
    return {
        "wh_now": w.astype(np.float32),
        "wh_max_25": ws.rolling(25, min_periods=1).max().to_numpy(np.float32),
        "wh_max_125": ws.rolling(125, min_periods=1).max().to_numpy(np.float32),
        "since_whistle": np.minimum(np.arange(len(w)) - last, 250).astype(np.float32),
        "rms_25": rs.rolling(25, min_periods=1).mean().to_numpy(np.float32),
        "rms_rise": (rs.rolling(25, min_periods=1).mean()
                     - rs.rolling(250, min_periods=1).mean()).to_numpy(np.float32),
    }


def load():
    d = np.load(OUT)
    return {k: d[k] for k in d.files}


def main():
    extract_wav()
    x = read_wav()
    print("audio seconds {:.2f}  video seconds {:.2f}".format(len(x) / SR, N_FRAMES / FPS))
    m = whistle_measures(x)
    np.savez_compressed(OUT, **m)
    print("wrote", os.path.relpath(OUT, ROOT))


if __name__ == "__main__":
    main()
