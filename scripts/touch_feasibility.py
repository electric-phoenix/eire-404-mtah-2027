"""Who touched the ball? Tracking-based touch and control tests (negative result, 16 Nov 2026).
Not part of the pipeline. Writes ctl_*.csv into evaluation/results/ for scoring with
    python scripts/compare_candidates.py --files lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle ctl_...
(delete them afterwards so they do not clutter the scoreboard).

    python scripts/touch_feasibility.py

PLAIN WORDS
-----------
1. KICKS AS ACCELERATION PEAKS (Hoernig et al. 2015): the ball path is smoothed with a
   Savitzky-Golay filter (7 frames, quadratic) and kicks are local peaks of acceleration.
   RESULT: at those peaks the nearest player is NOT closer than at random moments (median 2.1 m vs
   2.3 m; the strongest peaks 5.6 m). Ball jitter and airborne balls (projected to the ground, so
   they land off-pitch) swamp real kicks. The tracking cannot time individual kicks.
2. CONTROL ONSETS (Vidal-Codina et al. 2022 "possession zones"): a team starts sole control when
   its nearest player is within R of the ball, the nearest opponent is >= 0.5 m further, for 3+
   frames. RESULT: the model's team switches already sit at those onsets (median 0 to -3 frames),
   so the model is NOT late. The 12 public-answer switches come ~6-16 frames AFTER the onset.
3. WHAT THE TEAM ERRORS ARE (public window, diagnosis only): the long ones are duels/pressing.
   The ball is tracked and an OPPONENT is nearest for seconds (true team nearest only 16-42%).
4. CONFIRMED CONTROL (revert a switch unless the new team gets sole control; or move the switch to
   that moment): 12 settings, none better than v3 + whistle (spellF1 0.500-0.600 vs 0.600,
   macroF1 0.772-0.793 vs 0.793). Reverting removes real short possessions too; moving switches
   later breaks the correct ones. Conclusion: proximity alone cannot tell "pressing" from
   "won the ball"; it would need body orientation, contact or the next pass direction.
"""
import os
import sys
sys.path.insert(0, "."); sys.path.insert(0, "evaluation")
import numpy as np, pandas as pd
from scipy.signal import savgol_filter, find_peaks
from possession import atd as ATD
from games import GAME, half_video_blocks
from pipeline import run as RUN

# ======== 1. acceleration peaks ========


w = ATD.load()
n = len(w["bx"])
f = pd.read_parquet("data/processed/frame_features.parquet", columns=["ball_interpolated"])
interp = f["ball_interpolated"].to_numpy().astype(bool)
bx, by = w["bx"].astype(np.float64), w["by"].astype(np.float64)
good = np.isfinite(bx) & ~interp & (bx > -2) & (bx < 107.3) & (by > -2) & (by < 70)   # on/near pitch
team = w["team"]
lab = pd.read_csv("evaluation/results/lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle.csv")["label"].to_numpy().astype(str)

# 1. Savitzky-Golay smoothing per clean run, acceleration magnitude
WL, PO = 7, 2
acc = np.full(n, np.nan); spd = np.full(n, np.nan)
r = np.diff(np.r_[0, good.astype(int), 0]); st, en = np.flatnonzero(r == 1), np.flatnonzero(r == -1)
for s, e in zip(st, en):
    if e - s < WL + 2:
        continue
    vx = savgol_filter(bx[s:e], WL, PO, deriv=1) * 25; vy = savgol_filter(by[s:e], WL, PO, deriv=1) * 25
    ax = savgol_filter(bx[s:e], WL, PO, deriv=2) * 625; ay = savgol_filter(by[s:e], WL, PO, deriv=2) * 625
    acc[s:e] = np.hypot(ax, ay); spd[s:e] = np.hypot(vx, vy)
a0 = np.nan_to_num(acc, nan=0)
print("acc m/s^2 pct 50/90/99 on clean:", np.nanpercentile(acc, [50, 90, 99]).round(1))

# nearest player per team at every frame
d = np.hypot(w["px"] - bx[:, None], w["py"] - by[:, None])
home = team == 0
dH = np.nanmin(np.where(np.isfinite(d[:, home]), d[:, home], 1e9), 1)
dA = np.nanmin(np.where(np.isfinite(d[:, ~home]), d[:, ~home], 1e9), 1)
dmin = np.minimum(dH, dA)

for thr in (10, 20, 30):
    pk, _ = find_peaks(a0, height=thr, distance=5)
    pk = pk[good[pk]]
    print("\nacc peaks > %d m/s^2: %d" % (thr, len(pk)))
    print("  nearest-player distance at peaks pct 10/25/50: %s | at random clean frames: %s" % (
        np.percentile(dmin[pk], [10, 25, 50]).round(2),
        np.percentile(dmin[np.random.RandomState(0).choice(np.flatnonzero(good), 20000)], [10, 25, 50]).round(2)))
    for R in (1.0, 1.5, 2.0):
        tch = pk[dmin[pk] <= R]
        tteam = np.where(dH[tch] < dA[tch], "H", "A")
        clear = np.abs(dH[tch] - dA[tch]) >= 0.5
        agree = (tteam == lab[tch])[clear & np.isin(lab[tch], ["H", "A"])].mean()
        inplay = np.isin(lab[tch], ["H", "A"]).mean()
        # sequences: share of consecutive touches by same team
        same = (tteam[1:] == tteam[:-1]).mean()
        print("  R=%.1f m: touches %d (%.1f per live minute), agree with model team %.2f, in model-live %.2f, consecutive same team %.2f"
              % (R, len(tch), len(tch) / (np.isin(lab, ["H", "A"]).sum() / 1500), agree, inplay, same))
np.savez_compressed("data/processed/touch_explore.npz", acc=a0.astype(np.float32), dH=dH.astype(np.float32),
                    dA=dA.astype(np.float32), good=good)

# ======== 2. control onsets vs switches ========
z = np.load("data/processed/touch_explore.npz")
dH, dA, good = z["dH"], z["dA"], z["good"]


n = len(dH)
lab = pd.read_csv("evaluation/results/lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle.csv")["label"].to_numpy().astype(str)
gt = pd.read_csv("evaluation/gt/BAR-ATH_possession_gt_public.csv")["label"].to_numpy().astype(str)


def onsets(R=1.0, hold=3, gap=0.5):
    """frames where team T starts sole control: nearest T player within R, opponent >= R+gap, for `hold` frames."""
    cH = good & (dH <= R) & (dA >= dH + gap)
    cA = good & (dA <= R) & (dH >= dA + gap)
    out = {}
    for k, c in (("H", cH), ("A", cA)):
        run = np.convolve(c.astype(int), np.ones(hold, int), "full")[:n] == hold   # c held for `hold` frames ending here
        start = run & ~np.r_[False, run[:-1]]
        out[k] = np.flatnonzero(start) - (hold - 1)
    return out


def switches(l):
    return np.flatnonzero((l[1:] != l[:-1]) & np.isin(l[1:], ["H", "A"]) & np.isin(l[:-1], ["H", "A"])) + 1


for R, hold in ((1.0, 3), (1.5, 3), (1.0, 5)):
    on = onsets(R, hold)
    print("R=%.1f hold=%d: control onsets H %d A %d" % (R, hold, len(on["H"]), len(on["A"])))
    for name, l in (("model", lab), ("GT", np.where(gt == "X", "Z", gt))):
        lags = []
        for i in switches(l):
            new = l[i]
            o = on[new]
            c = o[(o >= i - 75) & (o <= i + 25)]
            lags.append(i - c[np.argmin(np.abs(i - c))] if len(c) else np.nan)
        lags = np.array(lags, float)
        found = np.isfinite(lags)
        print("   %-5s switches %3d, with new-team onset within -3s..+1s: %3d; switch minus onset (frames) pct 10/25/50/75/90: %s"
              % (name, len(lags), found.sum(), np.percentile(lags[found], [10, 25, 50, 75, 90]).round(0) if found.any() else "-"))

# ======== 4. confirmed-control variants ========


n = len(dH)
lab0 = pd.read_csv("evaluation/results/lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle.csv")["label"].to_numpy().astype(str)


def sole(team, Rr, gap, hold):
    d_own, d_opp = (dH, dA) if team == "H" else (dA, dH)
    c = good & (d_own <= Rr) & (d_opp >= d_own + gap)
    run = np.convolve(c.astype(int), np.ones(hold, int), "full")[:n] == hold
    return run                     # True at the LAST frame of a held stretch


def confirm(lab, Rr=1.0, gap=0.5, hold=5, shift=False, revert=True):
    out = lab.copy()
    conf = {"H": sole("H", Rr, gap, hold), "A": sole("A", Rr, gap, hold)}
    ha = np.isin(lab, ["H", "A"])
    r = np.diff(np.r_[0, (lab[1:] != lab[:-1]).astype(int)])
    starts = np.r_[0, np.flatnonzero(lab[1:] != lab[:-1]) + 1]
    ends = np.r_[starts[1:], n]
    n_rev = n_shift = 0
    for k in range(1, len(starts) - 1):
        s, e = starts[k], ends[k]
        B = lab[s]
        if B not in ("H", "A"):
            continue
        A_ = lab[s - 1]
        if A_ not in ("H", "A") or A_ == B:
            continue                                  # only direct team-to-team switches
        c = np.flatnonzero(conf[B][s:e])
        if len(c) == 0:
            # never confirmed: revert only when the team before also follows (A-B-A sandwich)
            if revert and lab[e] == A_:
                out[s:e] = A_; n_rev += 1
        elif shift:
            first = s + c[0] - (hold - 1)               # start of the first held sole-control stretch
            if first > s:
                out[s:first] = A_; n_shift += 1
    return out, n_rev, n_shift


rows = []
for Rr, gap, hold in ((1.0, 0.5, 5), (1.5, 0.5, 5), (1.0, 0.5, 10), (1.0, 1.0, 5)):
    for shift, revert in ((False, True), (True, False), (True, True)):
        lab, nr, ns = confirm(lab0, Rr, gap, hold, shift, revert)
        tag = "ctl_R%.1f_g%.1f_h%d_%s%s" % (Rr, gap, hold, "rev" if revert else "", "shift" if shift else "")
        RUN.write(os.path.join("evaluation/results", tag + ".csv"), lab)
        print("%-28s reverted %3d spells, shifted %3d switches, frames changed %5d" % (tag, nr, ns, (lab != lab0).sum()))