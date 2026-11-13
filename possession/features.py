"""Per-frame features from the ATD and the Smart Tagging. All indexed by video frame.

This is an offline labeller over a recorded match, so features may look at both past and future
frames (the decoder does anyway). Every feature is computed from provided data only -- never from
any ground truth.
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "starter"))
from games import GAME, GAMES, half_video_blocks  # noqa: E402

FPS = GAMES[GAME]["fps"]
LENGTH, WIDTH = GAMES[GAME]["pitch"]


def nearest_by_team(atd):
    """Distance (m) from the ball to the nearest home / away player; inf if none tracked."""
    dx = atd["px"] - atd["bx"][:, None]
    dy = atd["py"] - atd["by"][:, None]
    d = np.sqrt(dx * dx + dy * dy)
    d = np.where(np.isfinite(d), d, np.inf)
    home = atd["team"] == 0
    return d[:, home].min(axis=1), d[:, ~home].min(axis=1)


def ball_speed_ms(atd):
    """Ball speed from positions (m/s), centred difference; NaN where either side is missing."""
    bx, by = atd["bx"], atd["by"]
    v = np.full(len(bx), np.nan, np.float32)
    dx = bx[2:] - bx[:-2]
    dy = by[2:] - by[:-2]
    v[1:-1] = np.sqrt(dx * dx + dy * dy) * FPS / 2.0
    return v


def out_of_bounds(atd, margin=0.5):
    bx, by = atd["bx"], atd["by"]
    with np.errstate(invalid="ignore"):
        return ((bx < -margin) | (bx > LENGTH + margin) |
                (by < -margin) | (by > WIDTH + margin)) & np.isfinite(bx)


def player_activity(atd):
    """Mean and count of tracked players' speed per frame (stoppages are slow)."""
    ps = atd["ps"]
    n = np.isfinite(ps).sum(axis=1)
    with np.errstate(invalid="ignore"):
        mean = np.nanmean(np.where(np.isfinite(ps), ps, np.nan), axis=1)
    return mean, n


def graded_mask(n_frames, game=GAME):
    m = np.zeros(n_frames, bool)
    for _, a, b in half_video_blocks(game):
        m[a:b + 1] = True
    return m


def half_index(n_frames, game=GAME):
    h = np.zeros(n_frames, np.int8)
    for half, a, b in half_video_blocks(game):
        h[a:b + 1] = half
    return h


def build(atd):
    dh, da = nearest_by_team(atd)
    pmean, pn = player_activity(atd)
    return {
        "ball_vis": np.isfinite(atd["bx"]),
        "d_home": dh, "d_away": da,
        "ball_v": ball_speed_ms(atd),
        "ball_s_file": atd["bs"],
        "oob": out_of_bounds(atd),
        "p_speed": pmean, "p_n": pn,
    }
