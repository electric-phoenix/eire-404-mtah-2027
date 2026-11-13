"""Smart Tagging -> per-frame evidence arrays. Uses the official starter loader.

Team semantics (football reading of the codes, checked for sign only on the public window):
  in-possession codes  (BUILD UP, PROGRESSION, ...)     Team = team WITH the ball
  out-of-possession    (DEFENDING IN ...)                Team = team WITHOUT the ball
  SET PIECES           restart; Team = restarting team, N/A for every throw-in
"""

import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "starter"))
import load_smart_tagging  # noqa: E402

TEAM_SIDE = {"barcelona": 0, "atletic club": 1, "athletic club": 1}
SET_PIECES = "set pieces"
# Codes whose Team is the side in possession / out of possession. Location codes (ball in final
# third, ball in the box, players in the box) say where the ball is, not who has it: excluded.
IN_POSSESSION = {"build up", "progression", "creating chances", "long ball", "attacking transition"}
OUT_OF_POSSESSION = {"defending in defensive third", "defending in middle third",
                     "defending in attacking third", "defensive transition"}


def events():
    out = []
    for e in load_smart_tagging.load():
        code = (e["code"] or "").strip().lower()
        team = TEAM_SIDE.get((e["tags"].get("Team") or "").strip().lower())
        out.append({"code": code, "a": e["start_frame"], "b": e["end_frame"], "team": team,
                    "type": (e["tags"].get("Type") or "").strip().lower()})
    return out


def possession_votes(evs, n_frames):
    """Per frame: (#tags saying H has the ball) - (#tags saying A has the ball), per code group."""
    vote = np.zeros(n_frames, np.float32)
    for e in evs:
        if e["team"] is None:
            continue
        if e["code"] in IN_POSSESSION:
            side = e["team"]
        elif e["code"] in OUT_OF_POSSESSION:
            side = 1 - e["team"]
        else:
            continue
        vote[max(e["a"], 0):min(e["b"], n_frames)] += 1.0 if side == 0 else -1.0
    return vote


def set_pieces(evs):
    return [e for e in evs if e["code"] == SET_PIECES]


def code_sign_table(evs, gt):
    """For the sign check: per code, share of H among H/A GT frames inside its intervals,
    split by which side the tag names."""
    rows = {}
    for e in evs:
        if e["team"] is None:
            continue
        seg = gt[max(e["a"], 0):e["b"]]
        h, a = int((seg == "H").sum()), int((seg == "A").sum())
        if h + a == 0:
            continue
        key = e["code"]
        r = rows.setdefault(key, [0, 0])  # agree, disagree with "Team has the ball"
        tagged_h = e["team"] == 0
        r[0] += h if tagged_h else a
        r[1] += a if tagged_h else h
    return rows
