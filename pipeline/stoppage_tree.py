"""Stoppage clean-up: merge fragments, drop noise, place the edges, optionally weigh the evidence.

Runs on the decoded path AFTER the rule layer (offline, so it may look at future frames; the
per-frame features themselves stay causal). Every threshold is fixed in advance from football
reasoning or from a distribution over the whole match, never from a ground truth.

    D runs in the decoded path
      1. merge        two stoppages with under 3 s of play between them are one stoppage
      2. short        a stoppage under 2 s with no set-piece tag within 6 s is noise: removed
      3. midpoint     a tagged stoppage ends at its tag's midpoint (the tag window straddles
                      the restart; the offline teacher in possession/model.py does the same)
      4. edges        otherwise the stoppage ends when the players start running again, and starts
                      where the ball was last seen leaving over a line
    soft evidence     (hybrid_stoppage_soft) dead-ball leaves become log-odds bonuses on D in the
                      Viterbi emissions instead of a hard override; see LEAF_BONUS.

Not done, on purpose: fouls and injuries with the ball in view and no tag (weak signal, about 9
stoppages).
"""
import numpy as np

from . import dead_tree as DT

H, A, D, X = 0, 1, 2, 3
FPS = 25

MERGE_GAP = 3 * FPS         # play shorter than this between two stoppages is not play
SHORT_MIN = 2 * FPS
TAG_WINDOW = 150            # a tag within 6 s before the stoppage starts, or inside it
SHIFT_MAX = 100             # an edge moves at most 4 s
RESUME_WINDOW = 25          # players "are running again" when the next 1 s mean is fast enough
RESUME_PCT = 25             # ...fast enough = the 25th percentile of the same mean in live play
OOB_LOOKBACK = 100
RESTART_TYPES = ("throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)")

# Log-odds added to D by each dead_tree leaf in the soft variant, in the leaf order (certainty
# order). Fixed a priori; leaf 1 (pre-kickoff) is certain and is also applied as a hard override.
LEAF_BONUS = {1: 8.0, 2: 3.0, 3: 3.0, 4: 1.5, 5: 2.5, 6: 1.5}


def _alive(path, i):
    return path[i] in (H, A)


def fill_label(path, s, e):
    """Label that replaces a removed stoppage path[s:e]: the team before it, else the one after."""
    if s > 0 and _alive(path, s - 1):
        return path[s - 1]
    if e < len(path) and _alive(path, e):
        return path[e]
    return None


def merge_close(path, max_gap=MERGE_GAP):
    """Join stoppages that are less than 3 seconds apart.

    In plain words: if the game "restarts" for under 3 seconds and then stops again, that is not
    really play, it is one long stoppage that our detector chopped in two. Label the short gap as
    dead ball too, so one stoppage is one stoppage. This also stops the possession around it from
    being cut into extra little pieces. Returns (new labels, number of joins made).
    """
    p = path.copy()
    s, e = DT.runs(p == D)
    n = 0
    for i in range(len(s) - 1):
        if 0 < s[i + 1] - e[i] <= max_gap:
            p[e[i]:s[i + 1]] = D
            n += 1
    return p, n


def has_tag(tags_h, s, e, types=None):
    return any(t["code"] == "set pieces" and (types is None or t["type"] in types)
               and s - TAG_WINDOW <= t["a"] <= e for t in tags_h)


def drop_short(path, tags_h, min_len=SHORT_MIN):
    """Remove stoppages shorter than 2 seconds that have no set-piece tag near them.

    In plain words: real stoppages (throw-ins, corners, fouls) last several seconds. A dead-ball
    label that lasts under 2 seconds with nothing in the match tags to explain it is almost always
    a glitch (a few frames where the ball vanished). We delete it and hand those frames back to the
    team that had the ball just before. A short stoppage that does sit next to a set-piece tag is
    kept, because the tag is a reason to believe it. Returns (new labels, number removed).
    """
    p = path.copy()
    s, e = DT.runs(p == D)
    n = 0
    for s0, e0 in zip(s, e):
        if e0 - s0 >= min_len or s0 == 0 or e0 >= len(p) or has_tag(tags_h, s0, e0):
            continue
        lab = fill_label(p, s0, e0)
        if lab is not None:
            p[s0:e0] = lab
            n += 1
    return p, n


def forward_mean(x, w=RESUME_WINDOW):
    c = np.r_[0.0, np.cumsum(x)]
    i = np.arange(len(x))
    hi = np.minimum(len(x), i + w)
    return (c[hi] - c[i]) / np.maximum(hi - i, 1)


def refine_edges(path, f, tags_h, use_midpoint=True, use_edges=True):
    """Move each stoppage's start and end. Returns (path, number of edges moved)."""
    p = path.copy()
    n = len(p)
    speed = np.nan_to_num(f["speed_all"], nan=0.0)
    fm = forward_mean(speed)
    live = np.isin(p, [H, A]) & (f["n_tracked"] >= DT.MIN_PLAYERS)
    resume = np.percentile(fm[live], RESUME_PCT) if live.any() else np.inf
    s, e = DT.runs(p == D)
    moved = 0
    for i, (s0, e0) in enumerate(zip(s, e)):
        if s0 == 0 or e0 >= n:
            continue
        lo = e[i - 1] if i > 0 else 0
        hi = s[i + 1] if i + 1 < len(s) else n
        new_s, new_e = s0, e0

        tags = [t for t in tags_h if t["code"] == "set pieces" and t["type"] in RESTART_TYPES
                and s0 - TAG_WINDOW <= t["a"] <= e0 and t.get("b") is not None]
        if use_midpoint and tags:
            t = min(tags, key=lambda t: abs(t["a"] - s0))
            mid = (t["a"] + t["b"]) // 2
            if abs(mid - e0) <= TAG_WINDOW:
                new_e = mid
        elif use_edges:
            win = np.arange(max(e0 - SHIFT_MAX, s0 + RESUME_WINDOW), min(e0 + SHIFT_MAX, hi - 1))
            ok = np.flatnonzero(fm[win] >= resume) if len(win) else []
            if len(ok):
                new_e = int(win[ok[0]])
        if use_edges:
            a = max(s0 - OOB_LOOKBACK, lo)
            seen_out = (f["ball_invisible"][a:s0] == 0) & (f["ball_oob"][a:s0] > 0)
            if seen_out.any():
                new_s = a + int(np.flatnonzero(seen_out)[0])

        new_s, new_e = max(new_s, lo), min(new_e, hi)
        if new_e <= new_s + 1:
            continue
        lab = fill_label(path, s0, e0)
        if new_s < s0:
            p[new_s:s0] = D
        elif new_s > s0 and lab is not None:
            p[s0:new_s] = lab
        if new_e > e0:
            p[e0:new_e] = D
        elif new_e < e0 and lab is not None:
            p[new_e:e0] = lab
        moved += int(new_s != s0) + int(new_e != e0)
    return p, moved


def clean(path, f, tags_h, disable=(), log=None):
    """merge -> short -> edges/midpoint. `disable` may hold merge, short, midpoint, edges."""
    p, stats = path.copy(), {}
    if "merge" not in disable:
        p, stats["merged"] = merge_close(p)
    if "short" not in disable:
        p, stats["dropped"] = drop_short(p, tags_h)
    if "midpoint" not in disable or "edges" not in disable:
        p, stats["moved"] = refine_edges(p, f, tags_h, use_midpoint="midpoint" not in disable,
                                         use_edges="edges" not in disable)
    if log is not None:
        log.append(stats)
    return p


def soft_bonus(leaf):
    """Per-frame log-odds bonus for D from the dead_tree leaf ids."""
    b = np.zeros(len(leaf), np.float64)
    for k, v in LEAF_BONUS.items():
        b[leaf == k] = v
    return b
