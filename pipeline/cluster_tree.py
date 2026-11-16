"""Crowd rule: when the ball is untracked inside a pack of players, the team with more players in
the pack gets the ball.

Why: the ball goes missing most often in crowded boxes, aerial duels and tackles. The decoder's
fallback there is to hold the last team (decode.rule_invisible). This rule replaces that hold, but
only where a dense pack exists and one team clearly outnumbers the other, so it spends nothing on
frames that are not scrums.

    frame
      |-- ball tracked on this frame?                          yes -> untouched
      |-- decoder says dead ball (D)?                          yes -> untouched
      |-- fewer than 18 players tracked?                       yes -> untouched
      |-- densest pack: >= 6 players within 5 m of one point?  no  -> untouched (hold stays)
      |-- ball last seen within 2 s?
      |     |-- yes: pack centre within 10 m of that last position?   no -> untouched
      |     `-- leaf 1  crowd_near_ball
      |-- no: ball lost for longer
      |     `-- needs a bigger pack, >= 8 players               leaf 2  crowd_blind
      |-- one team outnumbers the other by >= 2 in the pack?   no  -> untouched
      `-- assign that team, but only if the vote holds for about a second (+/-25 frame mean
          of the per-frame votes, |mean| >= 0.5), so a pack that shifts by one player does not
          make the label flicker

All thresholds are fixed in advance from football reasoning and were not tuned on any ground
truth. The per-frame pack counts are built from the tracking alone.
"""
import os

import numpy as np

H, A, D, X = 0, 1, 2, 3
LEAVES = ["crowd_near_ball", "crowd_blind"]

RADIUS = 5.0              # metres: "immediate vicinity"
MIN_PACK = 6              # players of both teams within RADIUS of the pack centre
MIN_PACK_BLIND = 8
MARGIN = 2                # players one team must lead by
NEAR_BALL_M = 10.0        # pack centre vs the ball's last known position
LAST_SEEN_MAX = 50        # 2 s: the last known position is still informative
HOLD = 25                 # +/- frames the vote must persist for
MIN_PLAYERS = 18
CHUNK = 4000


def build_crowd(px, py, team):
    """Per frame: the densest pack of players (any team) and each team's count in it.

    The pack centre is the player with the most players (any team, itself included) within
    RADIUS. Returns dict of arrays: nH, nA (counts in that pack), cx, cy (its centre, NaN if no
    players). Pure function of the tracking.
    """
    n = len(px)
    out = {"nH": np.zeros(n, np.int8), "nA": np.zeros(n, np.int8),
           "cx": np.full(n, np.nan, np.float32), "cy": np.full(n, np.nan, np.float32)}
    home = team == 0
    for a in range(0, n, CHUNK):
        b = min(a + CHUNK, n)
        x, y = px[a:b], py[a:b]
        ok = np.isfinite(x)
        dx = x[:, :, None] - x[:, None, :]
        dy = y[:, :, None] - y[:, None, :]
        near = (dx * dx + dy * dy <= RADIUS * RADIUS) & ok[:, :, None] & ok[:, None, :]
        count = near.sum(axis=2)                       # players within RADIUS of each centre
        count[~ok] = 0
        c = count.argmax(axis=1)
        rows = np.arange(b - a)
        mine = near[rows, c]                           # (chunk, 46): who is in the pack
        has = ok.any(axis=1)
        out["nH"][a:b] = (mine & home[None, :]).sum(axis=1)
        out["nA"][a:b] = (mine & ~home[None, :]).sum(axis=1)
        out["cx"][a:b] = np.where(has, x[rows, c], np.nan)
        out["cy"][a:b] = np.where(has, y[rows, c], np.nan)
    return out


def load_or_build(proc_dir):
    """Cache the per-frame packs next to the other processed data (not committed)."""
    path = os.path.join(proc_dir, "crowd.npz")
    if os.path.exists(path):
        z = np.load(path)
        return {k: z[k] for k in z.files}
    z = np.load(os.path.join(proc_dir, "atd.npz"))
    out = build_crowd(z["px"], z["py"], z["team"])
    np.savez_compressed(path, **out)
    return out


def votes(f, crowd, path, disable=()):
    """Per-frame vote (+1 H, -1 A, 0 none) and leaf id (0 none, 1..2) before smoothing."""
    n = len(path)
    nH, nA = crowd["nH"].astype(int), crowd["nA"].astype(int)
    pack = nH + nA
    eligible = ((f["ball_invisible"] > 0) & (path != D) & (f["n_tracked"] >= MIN_PLAYERS)
                & (f["tracking_gap"] == 0))
    lead = nH - nA
    clear = np.abs(lead) >= MARGIN

    seen_recently = (f["frames_since_ball"] <= LAST_SEEN_MAX) & (f["ball_x"] >= 0)
    gap = np.hypot(crowd["cx"] - f["ball_x"], crowd["cy"] - f["ball_y"])
    near_ball = eligible & seen_recently & (pack >= MIN_PACK) & (gap <= NEAR_BALL_M) & clear
    blind = eligible & ~seen_recently & (pack >= MIN_PACK_BLIND) & clear

    leaf = np.zeros(n, np.int8)
    if "crowd_near_ball" not in disable:
        leaf[near_ball] = 1
    if "crowd_blind" not in disable:
        leaf[blind & (leaf == 0)] = 2
    vote = np.where(leaf > 0, np.sign(lead), 0).astype(np.float32)
    return vote, leaf


def smooth_sign(vote):
    """Sign of the +/-HOLD frame mean of the votes where it is at least 0.5 in size, else 0."""
    c = np.r_[0.0, np.cumsum(vote)]
    i = np.arange(len(vote))
    lo, hi = np.maximum(0, i - HOLD), np.minimum(len(vote), i + HOLD + 1)
    mean = (c[hi] - c[lo]) / (2 * HOLD + 1)
    return np.where(np.abs(mean) >= 0.5, np.sign(mean), 0)


def apply(path, f, crowd, disable=(), log=None):
    """Relabel crowded, ball-untracked alive frames with the team that outnumbers the other."""
    vote, leaf = votes(f, crowd, path, disable)
    s = smooth_sign(vote)
    hit = (leaf > 0) & (s != 0)
    p = path.copy()
    p[hit & (s > 0)] = H
    p[hit & (s < 0)] = A
    if log is not None:
        log.append({"frames": int(hit.sum()), "changed": int((p != path).sum()),
                    "leaf1": int((hit & (leaf == 1)).sum()), "leaf2": int((hit & (leaf == 2)).sum())})
    return p
