"""Kick-sound feasibility test (negative result, 16 Nov 2026). Not part of the pipeline.

    python scripts/kick_feasibility.py tactical
    python scripts/kick_feasibility.py broadcast     # needs data/processed/audio/broadcast_16k.wav

QUESTION (plain words)
----------------------
Can we hear the ball being kicked? If so, we could time possession changes to the touch.

HOW IT IS TESTED WITHOUT ANSWERS
--------------------------------
"Touch" examples come from the tracking: moments where the ball's velocity changes sharply
(smoothed over 4 frames each side, top 10%). "Non-touch" examples are random frames at least 1 s
from any such moment. A small LightGBM model learns, from 16 sound bands around each frame, to tell
them apart. It is scored on 5-minute blocks it never trained on (AUC: 0.5 = coin flip, 1 = perfect).
A PLACEBO run shifts the sound 2 minutes against the tracking: a real kick signal must vanish.

RESULT
------
  tactical  AUC 0.599, placebo 0.561   -> the model learns crowd context, not kicks
  broadcast AUC 0.473, placebo 0.490   -> chance
The broadcast is aligned to tactical frames with the referee's whistle (pipeline/audio.py):
tactical frame = broadcast frame + 4011 (1st half) / + 26917 (2nd half); about 40% of whistles
match within 1 frame, loudness correlates 0.14-0.15 at that offset vs <= 0.09 elsewhere.
Also tried and equally null: hand-made click scores (2 ms energy over a 250 ms background; AUC
0.40-0.63) and moving the model's team switches to the strongest click within 0.5 s (no better
than the time-shifted placebo). Conclusion: the stadium microphones do not resolve single kicks
over crowd noise. See reports/experiments.md.
"""

import sys, wave
sys.path.insert(0, ".")
import numpy as np, pandas as pd, lightgbm as lgb
from scipy import fft as sfft
from scipy.ndimage import median_filter
from sklearn.model_selection import GroupKFold
from sklearn.metrics import roc_auc_score

n = 199319
offs = np.array([(11185, 90651, 4011), (110808, 190197, 26917)])   # see docstring
SRC = sys.argv[1] if len(sys.argv) > 1 else "broadcast"
wav = "data/processed/audio/%s_16k.wav" % ("broadcast" if SRC == "broadcast" else "tactical")
with wave.open(wav, "rb") as w:
    x = np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768
HOP, WIN = 160, 512                                  # 10 ms sub-frames
m = len(x) // HOP
pad = np.pad(x, (WIN // 2, WIN))
view = np.lib.stride_tricks.sliding_window_view(pad, WIN)
edges = np.geomspace(60, 7500, 17)
fq = np.fft.rfftfreq(WIN, 1 / 16000)
bidx = [(fq >= edges[i]) & (fq < edges[i + 1]) for i in range(16)]
win = np.hanning(WIN).astype(np.float32)
B = np.zeros((m, 16), np.float32)
for a in range(0, m, 200000):
    idx = np.arange(a, min(a + 200000, m))
    P = np.abs(sfft.rfft((view[idx * HOP] * win).astype(np.float64), axis=1, workers=-1)) ** 2
    for k, bm in enumerate(bidx):
        B[idx, k] = np.log10(P[:, bm].mean(1) + 1e-12)
C = B - median_filter(B, size=(51, 1), mode="nearest")      # contrast vs +-0.25 s background

# map tactical frame t -> sub-frame index of its start in this audio
def sub_index(t):
    if SRC != "broadcast":
        return t * 4
    s = np.full(len(t), -1)
    for a, b, o in offs:
        sel = (t >= a) & (t <= b)
        s[sel] = (t[sel] - o) * 4
    return s

f = pd.read_parquet("data/processed/frame_features.parquet",
                    columns=["ball_x", "ball_y", "ball_invisible", "ball_interpolated"])
ok = ((f["ball_invisible"] == 0) & (f["ball_interpolated"] == 0)).to_numpy()
bx, by = f["ball_x"].to_numpy(), f["ball_y"].to_numpy(); K = 4
def sh(a_, s):
    o = np.zeros_like(a_)
    if s > 0: o[:-s] = a_[s:]
    else: o[-s:] = a_[:n + s]
    return o
okw = np.convolve(ok.astype(int), np.ones(9, int), "same") == 9
dv = np.hypot(sh(bx, K) - 2 * bx + sh(bx, -K), sh(by, K) - 2 * by + sh(by, -K)) / K * 25
lm = okw & (dv >= sh(dv, -1)) & (dv >= sh(dv, 1)) & (dv > np.percentile(dv[okw], 90))
pos = np.flatnonzero(lm); pos = pos[np.r_[True, np.diff(pos) > 12]]
far = np.ones(n, bool)
for p in pos:
    far[max(0, p - 25):p + 26] = False
rng = np.random.RandomState(0)
neg = rng.choice(np.flatnonzero(okw & far), 8 * len(pos), replace=False)
T = np.r_[pos, neg]; y = np.r_[np.ones(len(pos)), np.zeros(len(neg))]
si = sub_index(T)
keep = (si >= 12) & (si + 16 < m)
T, y, si = T[keep], y[keep], si[keep]
# features: contrast of 16 bands over sub-frames -12..+15 (frame-3 .. frame+3), plus max over that span
offsets = np.arange(-12, 16)
X = np.stack([C[si + o] for o in offsets], 1).reshape(len(si), -1)
X = np.c_[X, C[si[:, None] + offsets].max(1)]
groups = T // 7500                                         # 5-minute blocks
oof = np.zeros(len(y))
for tr, va in GroupKFold(5).split(X, y, groups):
    mdl = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
                     "min_data_in_leaf": 30, "feature_fraction": 0.5, "verbose": -1, "seed": 0},
                    lgb.Dataset(X[tr], y[tr]), 300)
    oof[va] = mdl.predict(X[va])
print("%s audio: positives %d negatives %d  out-of-fold AUC %.3f" % (SRC, int(y.sum()), int((1 - y).sum()), roc_auc_score(y, oof)))
# placebo: shift the sound 3000 frames relative to the tracking
si2 = si + 3000 * 4; ok2 = si2 + 16 < m
X2 = np.stack([C[si2[ok2] + o] for o in offsets], 1).reshape(ok2.sum(), -1)
X2 = np.c_[X2, C[si2[ok2][:, None] + offsets].max(1)]
oof2 = np.zeros(ok2.sum())
for tr, va in GroupKFold(5).split(X2, y[ok2], groups[ok2]):
    mdl = lgb.train({"objective": "binary", "learning_rate": 0.05, "num_leaves": 15,
                     "min_data_in_leaf": 30, "feature_fraction": 0.5, "verbose": -1, "seed": 0},
                    lgb.Dataset(X2[tr], y[ok2][tr]), 300)
    oof2[va] = mdl.predict(X2[va])
print("%s audio PLACEBO (sound shifted 2 min): out-of-fold AUC %.3f" % (SRC, roc_auc_score(y[ok2], oof2)))
