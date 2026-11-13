"""Feature store (brief Step 1). CAUSAL: every value at frame t is computed from frames <= t and
from Smart Tagging tags whose START is <= t (tag ends are never used -- they are future info).

Groups (used for ablation in Step 4):
  kinematics   ball position/speed, nearest-player distances per team, team speeds
  team         players within 2 m / 5 m per team, team centroids, ball position in H's attacking frame
  uncertainty  ball visibility, frames since ball seen, tracking gap, players tracked
  tagging      last tag code, its possession vote, frames since it started, set-piece context
  temporal     trailing means of control/margin/visibility/speed; frames since last H/A control
"""
import numpy as np
import pandas as pd

from . import flags as FL

FPS = 25
LENGTH, WIDTH = 105.3, 68.0
CAP = 250  # frames-since features are capped at 10 s
REACH = 1.5

TAG_CODES = ["none", "set pieces", "build up", "progression", "creating chances", "long ball",
             "attacking transition", "defensive transition", "defending in defensive third",
             "defending in middle third", "defending in attacking third", "ball in final third",
             "ball in the box", "players in the box", "goals"]
TAG_INDEX = {c: i for i, c in enumerate(TAG_CODES)}
IN_POSS = {"build up", "progression", "creating chances", "long ball", "attacking transition"}
OUT_POSS = {"defensive transition", "defending in defensive third", "defending in middle third",
            "defending in attacking third"}
SP_TYPES = ["none", "throw-in", "corner kick", "goal kick", "free kick", "kick off (after goal)",
            "kick off (start)"]

GROUPS = {
    "kinematics": ["ball_x", "ball_y", "ball_speed", "d_H1", "d_A1", "d_H2", "d_A2", "margin",
                   "speed_H", "speed_A", "speed_all", "ball_oob"],
    "team": ["n_H_2m", "n_A_2m", "n_H_5m", "n_A_5m", "cx_H_att", "cx_A_att", "ball_x_att_H"],
    "uncertainty": ["ball_invisible", "ball_interpolated", "frames_since_ball", "tracking_gap",
                    "n_tracked"],
    "tagging": ["tag_code", "tag_vote", "tag_age", "sp_type", "sp_team", "sp_age",
                "dead_ball_interval"],
    "temporal": ["ctrl_H_25", "ctrl_A_25", "ctrl_H_75", "ctrl_A_75", "margin_12", "margin_50",
                 "vis_25", "speed_all_50", "since_ctrl_H", "since_ctrl_A", "ball_speed_max_12"],
}
ALL_FEATURES = [c for g in GROUPS.values() for c in g]


def _since(event):
    """Frames since the last True (capped). Causal."""
    n = len(event)
    idx = np.where(event, np.arange(n), -10 ** 9)
    last = np.maximum.accumulate(idx)
    return np.minimum(np.arange(n) - last, CAP).astype(np.float32)


def _trail_mean(x, k):
    return pd.Series(x.astype(np.float64)).rolling(k, min_periods=1).mean().to_numpy(np.float32)


def _trail_max(x, k):
    return pd.Series(x.astype(np.float64)).rolling(k, min_periods=1).max().to_numpy(np.float32)


def _tag_state(tags, n):
    """Latest started tag (code, possession vote) and latest started set piece (type, team, age)."""
    code = np.zeros(n, np.int8)
    vote = np.zeros(n, np.int8)
    start = np.full(n, -10 ** 9)
    sp_type = np.zeros(n, np.int8)
    sp_team = np.zeros(n, np.int8)          # 0 none / N/A, 1 home, 2 away
    sp_start = np.full(n, -10 ** 9)
    for t in sorted(tags, key=lambda t: t["a"]):
        a = max(t["a"], 0)
        if a >= n:
            continue
        c = t["code"]
        v = 0
        if t["team"] is not None and c in IN_POSS:
            v = 1 if t["team"] == 0 else -1
        elif t["team"] is not None and c in OUT_POSS:
            v = -1 if t["team"] == 0 else 1
        code[a:] = TAG_INDEX.get(c, 0)
        vote[a:] = v
        start[a:] = a
        if c == "set pieces":
            sp_type[a:] = SP_TYPES.index(t["type"]) if t["type"] in SP_TYPES else 0
            sp_team[a:] = 0 if t["team"] is None else t["team"] + 1
            sp_start[a:] = a
    idx = np.arange(n)
    return {
        "tag_code": code, "tag_vote": vote,
        "tag_age": np.minimum(idx - start, CAP).astype(np.float32),
        "sp_type": sp_type, "sp_team": sp_team,
        "sp_age": np.minimum(idx - sp_start, CAP).astype(np.float32),
    }


def _two_nearest(d):
    s = np.sort(np.where(np.isfinite(d), d, np.inf), axis=1)
    return s[:, 0], s[:, 1] if s.shape[1] > 1 else np.full(len(s), np.inf)


def frame_features(w, tags, halves):
    """Frame-level feature table (one row per video frame)."""
    n = len(w["bx"])
    fl = FL.build(w, tags, halves)
    bx, age = FL.hold_last(w["bx"], FL.HOLD_MAX)
    by, _ = FL.hold_last(w["by"], FL.HOLD_MAX)

    # ball speed from tracked positions, backward difference over 2 frames (causal)
    spd = np.full(n, np.nan, np.float32)
    spd[2:] = np.hypot(w["bx"][2:] - w["bx"][:-2], w["by"][2:] - w["by"][:-2]) * FPS / 2
    spd, _ = FL.hold_last(spd, 5)

    home = w["team"] == 0
    d = np.hypot(w["px"] - bx[:, None], w["py"] - by[:, None])
    dH1, dH2 = _two_nearest(d[:, home])
    dA1, dA2 = _two_nearest(d[:, ~home])
    big = 60.0
    clip = lambda a: np.where(np.isfinite(a), np.minimum(a, big), big).astype(np.float32)  # noqa: E731
    dH1, dH2, dA1, dA2 = clip(dH1), clip(dH2), clip(dA1), clip(dA2)
    margin = np.clip(dA1 - dH1, -10, 10)

    with np.errstate(invalid="ignore"):
        ps = w["ps"]
        speed_H = np.nanmean(np.where(home, ps, np.nan), axis=1)
        speed_A = np.nanmean(np.where(~home, ps, np.nan), axis=1)
        speed_all = np.nanmean(ps, axis=1)
        cxH = np.nanmean(np.where(home, w["px"], np.nan), axis=1)
        cxA = np.nanmean(np.where(~home, w["px"], np.nan), axis=1)

    # attacking frame for H: 1st half H attacks left->right (ATD metadata); flips in the 2nd half
    flip = np.zeros(n, bool)
    for h, a, b in halves:
        if h == 2:
            flip[a:] = True
    att = lambda x: np.where(flip, LENGTH - x, x)  # noqa: E731

    dfin = np.isfinite(d)
    within = lambda r, m: (dfin & (d < r) & m[None, :]).sum(axis=1).astype(np.float32)  # noqa: E731
    with np.errstate(invalid="ignore"):
        oob = (bx < -0.5) | (bx > LENGTH + 0.5) | (by < -0.5) | (by > WIDTH + 0.5)
    ctrl_H = (dH1 < REACH) & (dH1 < dA1) & np.isfinite(bx)
    ctrl_A = (dA1 < REACH) & (dA1 < dH1) & np.isfinite(bx)

    f = {
        "ball_x": np.nan_to_num(bx, nan=-1), "ball_y": np.nan_to_num(by, nan=-1),
        "ball_speed": np.nan_to_num(spd, nan=-1),
        "d_H1": dH1, "d_A1": dA1, "d_H2": dH2, "d_A2": dA2, "margin": margin.astype(np.float32),
        "speed_H": np.nan_to_num(speed_H, nan=0), "speed_A": np.nan_to_num(speed_A, nan=0),
        "speed_all": np.nan_to_num(speed_all, nan=0), "ball_oob": oob.astype(np.float32),
        "n_H_2m": within(2, home), "n_A_2m": within(2, ~home),
        "n_H_5m": within(5, home), "n_A_5m": within(5, ~home),
        "cx_H_att": np.nan_to_num(att(cxH), nan=-1), "cx_A_att": np.nan_to_num(att(cxA), nan=-1),
        "ball_x_att_H": np.nan_to_num(att(bx), nan=-1),
        "ball_invisible": fl["ball_invisible"].astype(np.float32),
        "ball_interpolated": fl["ball_interpolated"].astype(np.float32),
        "frames_since_ball": np.where(age < 0, CAP, np.minimum(age, CAP)).astype(np.float32),
        "tracking_gap": fl["tracking_gap"].astype(np.float32),
        "n_tracked": np.isfinite(w["px"]).sum(axis=1).astype(np.float32),
        "dead_ball_interval": fl["dead_ball_interval"].astype(np.float32),
        "ctrl_H_25": _trail_mean(ctrl_H, 25), "ctrl_A_25": _trail_mean(ctrl_A, 25),
        "ctrl_H_75": _trail_mean(ctrl_H, 75), "ctrl_A_75": _trail_mean(ctrl_A, 75),
        "margin_12": _trail_mean(margin, 12), "margin_50": _trail_mean(margin, 50),
        "vis_25": _trail_mean(~fl["ball_invisible"], 25),
        "speed_all_50": _trail_mean(np.nan_to_num(speed_all, nan=0), 50),
        "since_ctrl_H": _since(ctrl_H), "since_ctrl_A": _since(ctrl_A),
        "ball_speed_max_12": _trail_max(np.nan_to_num(spd, nan=0), 12),
    }
    f.update(_tag_state(tags, n))
    df = pd.DataFrame({k: np.asarray(v) for k, v in f.items()})
    df.insert(0, "frame", np.arange(n, dtype=np.int32))
    return df[["frame"] + ALL_FEATURES]


def player_features(long_df, frame_df):
    """Per-(frame, player) columns the brief names: dist_to_ball, speed, plus frame-level
    broadcasts ball_speed, team_players_within_5m, ball_invisible, smart_tag_event_type."""
    fr = frame_df.set_index("frame")
    f = long_df["frame"].to_numpy()
    bx = fr["ball_x"].to_numpy()[f]
    by = fr["ball_y"].to_numpy()[f]
    have = bx >= 0
    dist = np.where(have & ~long_df["is_ball"].to_numpy(),
                    np.hypot(long_df["x"].to_numpy() - bx, long_df["y"].to_numpy() - by), np.nan)
    out = long_df.copy()
    out["dist_to_ball"] = dist.astype(np.float32)
    out["ball_speed"] = fr["ball_speed"].to_numpy()[f]
    side = out["team_side"].to_numpy()
    out["team_players_within_5m"] = np.where(
        side == "H", fr["n_H_5m"].to_numpy()[f],
        np.where(side == "A", fr["n_A_5m"].to_numpy()[f], np.nan)).astype(np.float32)
    out["ball_invisible"] = fr["ball_invisible"].to_numpy()[f].astype(bool)
    out["ball_interpolated"] = fr["ball_interpolated"].to_numpy()[f].astype(bool)
    out["tracking_gap"] = fr["tracking_gap"].to_numpy()[f].astype(bool)
    out["dead_ball_interval"] = fr["dead_ball_interval"].to_numpy()[f].astype(bool)
    out["smart_tag_event_type"] = pd.Categorical.from_codes(
        fr["tag_code"].to_numpy()[f].astype(int), TAG_CODES)
    return out
