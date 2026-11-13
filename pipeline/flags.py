"""Per-frame flags (brief Step 1). CAUSAL: the value at frame t uses frames <= t only.

    ball_invisible      ball not tracked on this frame
    ball_interpolated   ball not tracked, position held from the last tracked frame (<= HOLD_MAX ago)
    tracking_gap        fewer than MIN_PLAYERS players tracked
    dead_ball_interval  Smart Tagging says the ball is (probably) dead: a SET PIECES tag started in
                        the last SETPIECE_DEAD frames, or this half's kickoff tag has not started yet,
                        or the frame is outside both halves.
"""
import numpy as np

HOLD_MAX = 50          # 2 s: longer than this, a held ball position is not evidence of anything
MIN_PLAYERS = 18
SETPIECE_DEAD = 75     # tag windows straddle the restart; their first ~3 s are the stoppage


def hold_last(values, max_age):
    """Forward-fill NaNs with the last finite value, at most max_age frames old. Also returns age
    (0 when observed, -1 when nothing usable). Causal."""
    n = len(values)
    seen = np.isfinite(values)
    idx = np.where(seen, np.arange(n), -1)
    last = np.maximum.accumulate(idx)
    age = np.where(last >= 0, np.arange(n) - last, -1)
    ok = (last >= 0) & (age <= max_age)
    out = np.full(n, np.nan, values.dtype)
    out[ok] = values[last[ok]]
    age = np.where(ok, age, -1)
    return out, age


def build(w, tags, halves):
    """w: wide ATD arrays; tags: list of tag dicts (code, a, team, type); halves: [(h, a, b)]."""
    n = len(w["bx"])
    vis = np.isfinite(w["bx"])
    _, age = hold_last(w["bx"], HOLD_MAX)
    n_players = np.isfinite(w["px"]).sum(axis=1)

    dead = np.ones(n, bool)
    for _, a, b in halves:
        dead[a:b + 1] = False
        kick = [t["a"] for t in tags if t["code"] == "set pieces" and "kick off (start)" in t["type"]
                and a - 3000 <= t["a"] <= b]
        # Dead until this half's kickoff tag has STARTED. Causal: at frame t the question is
        # "has a kickoff tag started yet?", so with no kickoff tag seen the half stays dead.
        stop = min(kick) if kick else b + 1
        dead[a:max(a, stop)] = True
    for t in tags:
        if t["code"] == "set pieces":
            s = max(t["a"], 0)
            dead[s:min(s + SETPIECE_DEAD, n)] = True
    return {
        "ball_invisible": ~vis,
        "ball_interpolated": (~vis) & (age > 0),
        "tracking_gap": n_players < MIN_PLAYERS,
        "dead_ball_interval": dead,
    }
