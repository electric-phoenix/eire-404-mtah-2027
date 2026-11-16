"""Dead-ball decision tree.

One ordered list of tests ("leaves"). Each frame gets the FIRST leaf whose test fires, so the
order is the ranking: the most reliable evidence of a stoppage comes first, the weakest last.
Leaf 0 means "no dead-ball evidence", and the frame keeps whatever the decoder gave it.

The order is set by how certain each piece of evidence is by construction, NOT by scores on the
public window. Seventeen possessions cannot rank anything (EVALUATION.md, RULES.md), so
scripts/dead_tree_report.py prints per-leaf numbers as a check on the order, never as a way to
choose it.

    rank  leaf                evidence                                           run-level
    ----  ------------------  -------------------------------------------------  ---------
     1    pre_kickoff         before this half's kickoff tag has started          no
     2    setpiece_stopped    set-piece tag just started, players stopped or      no
                              ball off the pitch
     3    ball_off_pitch      ball tracked outside the touchlines for >= 2 s      yes
     4    setpiece_ball_still set-piece tag within 5 s, ball visible and still    no
     5    ball_missing_slow   ball untracked for >= 5 s while players are slow    yes
     6    players_still       players AND ball nearly motionless for >= 3 s,      yes
                              no tag needed (fouls, injuries)

"Run-level" leaves test a whole stretch of frames and label all of it, so the first seconds of a
stoppage are labelled too. Every input is a per-frame feature from pipeline/features.py. A leaf
that depends on tracked players is skipped on frames with a tracking gap, because a missing
player reads as "not moving".

Brief idea this tree covers: "no ball for a long time means the ball is not in play" (leaf 5).
It is deliberately conditioned on slow players. The FAQ says the ball is missing most often in
crowded boxes, aerial duels and tackles, which are live play: a long scramble has fast players
and stays alive.
"""
import numpy as np

FPS = 25

SLOW_MS = 1.6                 # mean player speed (m/s) below which play looks stopped
STILL_MS = 1.0                # ...and below which it looks standing still
BALL_STILL_MS = 1.0           # trailing max ball speed (m/s) for "the ball is not moving"
SETPIECE_STILL_MS = 1.5
SETPIECE_WINDOW = 125         # 5 s after a set-piece tag starts
OOB_MIN = 2 * FPS             # ball outside the lines this long is not an error of a metre
MISSING_MIN = 5 * FPS         # the 5 s threshold
STILL_MIN = 3 * FPS
MIN_PLAYERS = 18

# "rework" profile (fixed in advance from football reasoning, not tuned on any ground truth):
#   - setpiece_ball_still is dropped: it covers 0.3% of frames and the tags already drive leaf 2
#   - players_still no longer needs a motionless ball. Players at walking pace (1.3 m/s) and a ball
#     that is being carried or rolled (under 3 m/s) for 5 s is a stoppage with the ball visible
STILL2_MS = 1.3
BALL_CARRIED_MS = 3.0
STILL2_MIN = 5 * FPS

LEAVES = ["pre_kickoff", "setpiece_stopped", "ball_off_pitch", "setpiece_ball_still",
          "ball_missing_slow", "players_still"]


def runs(mask):
    """(start, end_exclusive) arrays of the True runs in a boolean array."""
    padded = np.r_[False, np.asarray(mask, bool), False].astype(np.int8)
    change = np.diff(padded)
    return np.flatnonzero(change == 1), np.flatnonzero(change == -1)


def fill_long_runs(mask, min_len, keep=None):
    """True over every run of `mask` that is at least min_len frames long.

    If `keep` is given it is called as keep(start, end) and a run is only filled when it returns
    True (used to ask "were the players slow across the whole run?").
    """
    out = np.zeros(len(mask), bool)
    for s, e in zip(*runs(mask)):
        if e - s >= min_len and (keep is None or keep(s, e)):
            out[s:e] = True
    return out


def pre_kickoff_mask(n, tags_h):
    """Frames before this half's kickoff tag has started. Same rule as flags.build.

    tags_h: Smart Tagging events with `a` relative to the half's first frame. With no kickoff tag
    in range nothing is marked (the safe direction: no tag, no claim).
    """
    kicks = [t["a"] for t in tags_h
             if t["code"] == "set pieces" and "kick off (start)" in t["type"]]
    out = np.zeros(n, bool)
    if kicks:
        out[:max(0, min(kicks))] = True
    return out


def classify(f, pre_kickoff, disable=(), rework=False):
    """Leaf id per frame (0 = no evidence, 1..6 = LEAVES order, first match wins).

    f: dict of per-frame feature arrays for ONE half. pre_kickoff: boolean array, same length.
    disable: leaf names to switch off (ablation).
    rework: use the reworked leaves 4 and 6 (see STILL2_MS above).
    """
    n = len(pre_kickoff)
    have_players = (f["tracking_gap"] == 0) & (f["n_tracked"] >= MIN_PLAYERS)
    visible = f["ball_invisible"] == 0
    slow = f["speed_all_50"] < SLOW_MS
    ball_still = visible & (f["ball_speed_max_12"] < BALL_STILL_MS)

    tests = {
        "pre_kickoff": lambda: np.asarray(pre_kickoff, bool),

        "setpiece_stopped": lambda: (f["dead_ball_interval"] > 0)
                                    & ((slow & have_players) | (f["ball_oob"] > 0)),

        "ball_off_pitch": lambda: fill_long_runs((f["ball_oob"] > 0) & visible, OOB_MIN),

        "setpiece_ball_still": lambda: (f["sp_age"] < SETPIECE_WINDOW) & (f["sp_type"] > 0)
                                       & visible & (f["ball_speed_max_12"] < SETPIECE_STILL_MS)
                                       & have_players,

        "ball_missing_slow": lambda: fill_long_runs(
            ~visible, MISSING_MIN,
            keep=lambda s, e: bool(have_players[s:e].all())
            and float(np.mean(f["speed_all"][s:e])) < SLOW_MS),

        "players_still": lambda: fill_long_runs(
            have_players & (f["speed_all_50"] < STILL_MS) & ball_still, STILL_MIN),
    }

    if rework:
        tests["players_still"] = lambda: fill_long_runs(
            have_players & visible & (f["speed_all_50"] < STILL2_MS)
            & (f["ball_speed_max_12"] < BALL_CARRIED_MS), STILL2_MIN)
        disable = tuple(disable) + ("setpiece_ball_still",)

    leaf = np.zeros(n, np.int8)
    for i, name in enumerate(LEAVES, start=1):
        if name in disable:
            continue
        hit = tests[name]() & (leaf == 0)
        leaf[hit] = i
    return leaf
