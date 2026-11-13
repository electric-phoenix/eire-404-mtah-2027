"""ATD parsers (brief Step 1).

    parse_atd_metadata(xml_path) -> dict
    parse_atd_data(txt_path)     -> long DataFrame: frame, player_id (0 = ball), x, y, speed, is_ball
    load_wide()                  -> dense arrays used by the feature code (cached)

Verified facts about this export (see reports/model.md):
  - frame index: video_frame = ATD frame_id - 1
  - x, y normalised 0-1; metres = x * 105.3, y * 68.0 (games.py). Checked: the 1st-half kickoff
    ball lands 1.4 m from the centre spot.
  - player speed column: labelled kph in the XML but is m/s (ratio 1.00, r = 0.95 against
    position-derived speed).
  - ball speed column: uncorrelated with ball movement (r = -0.03) -> derive from positions.
"""
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402
from possession import atd as _wide  # noqa: E402

N_SLOTS = _wide.N_SLOTS
BALL_ID = 0


def parse_atd_metadata(xml_path=None):
    xml_path = xml_path or require(["atd_meta"])[0]
    root = ET.parse(xml_path).getroot()
    teams = []
    for t in root.iter("Team"):
        params = {p.findtext("Name"): p.findtext("Value") for p in t.iter("ProviderParameter")}
        teams.append({"id": t.get("id"), "name": t.findtext("Name"),
                      "attack_direction_first_half": params.get("attack_direction_first_half")})
    home_id = teams[0]["id"]
    players = []
    for p in root.iter("Player"):
        params = {q.findtext("Name"): q.findtext("Value") for q in p.iter("ProviderParameter")}
        players.append({"player_id": int(p.get("id")),
                        "team_side": "H" if p.get("teamId") == home_id else "A",
                        "position_type": params.get("position_type")})
    global_params = {p.findtext("Name"): p.findtext("Value")
                     for p in root.find(".//ProviderGlobalParameters").iter("ProviderParameter")}
    return {
        "fps": float(root.findtext(".//FrameRate")),
        "frame_range": (int(global_params["first_half_start"]) + 1,
                        int(global_params["first_half_end"])),
        "pitch_metadata": (float(root.findtext(".//FieldSize/Width")),
                           float(root.findtext(".//FieldSize/Height"))),
        "pitch_metres": GAMES[GAME]["pitch"],
        "home": teams[0], "away": teams[1],
        "players": players,
        "goalkeepers": {pl["team_side"]: pl["player_id"] for pl in players
                        if pl["position_type"] == "Goalkeeper"},
    }


def wide_to_long(w):
    """Dense arrays -> long table, one row per (frame, player_id), ball as player_id 0.
    x, y in METRES; NaN rows (object not tracked) are kept so every (frame, id) exists."""
    n = len(w["bx"])
    frames = np.repeat(np.arange(n, dtype=np.int32), N_SLOTS + 1)
    pid = np.tile(np.arange(N_SLOTS + 1, dtype=np.int16), n)
    x = np.concatenate([w["bx"][:, None], w["px"]], axis=1).ravel()
    y = np.concatenate([w["by"][:, None], w["py"]], axis=1).ravel()
    s = np.concatenate([w["bs"][:, None], w["ps"]], axis=1).ravel()
    return pd.DataFrame({"frame": frames, "player_id": pid, "x": x, "y": y, "speed": s,
                         "is_ball": pid == BALL_ID})


def parse_atd_data(txt_path=None, n_frames=None, pitch=None):
    """Long format as specified by the brief."""
    txt_path = txt_path or require(["atd_raw"])[0]
    with open(txt_path) as fh:
        text = fh.read()
    w = _wide.parse_lines(text, n_frames or source_frames("tactical"), pitch or GAMES[GAME]["pitch"])
    return wide_to_long(w)


def load_wide(rebuild=False):
    return _wide.load(rebuild=rebuild)
