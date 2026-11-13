"""Parse the tactical ATD into dense numpy arrays, indexed by VIDEO frame (= submission row).

Raw line: "<frame_id>:<x,y,s>;<x,y,s>;...(46 slots):<bx,by,bs>", NaN for missing.
video_frame = frame_id - 1 (verified by starter/load_atd.py).

Output (cached to data/processed/atd.npz, which is gitignored with the rest of data/):
    px, py   float32 (N, 46)  player positions in METRES (x along the 105.3 m axis)
    ps       float32 (N, 46)  player speed as given by the file
    bx, by   float32 (N,)     ball position in metres, NaN when not tracked
    bs       float32 (N,)     ball speed as given by the file
    team     int8   (46,)     0 = home (H), 1 = away (A) for each slot
"""

import io
import os
import sys
import xml.etree.ElementTree as ET

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import GAME, GAMES, require, source_frames  # noqa: E402

N_SLOTS = 46
CACHE = os.path.join(ROOT, "data", "processed", "atd.npz")


def slot_teams(meta_path):
    """0 = home, 1 = away per slot; first <Team> in the metadata is home (EPTS convention)."""
    root = ET.parse(meta_path).getroot()
    team_ids = [t.get("id") for t in root.iter("Team")]
    assert len(team_ids) == 2, "expected two teams in ATD metadata"
    slots = [p.get("teamId") for p in root.iter("Player")]
    assert len(slots) == N_SLOTS, "expected {} slots, got {}".format(N_SLOTS, len(slots))
    return np.array([0 if s == team_ids[0] else 1 for s in slots], dtype=np.int8)


def parse_lines(text, n_frames, pitch):
    """Parse raw ATD text into the dict of arrays described above. Pure; used by tests too."""
    flat = text.replace(":", ",").replace(";", ",")
    width = 1 + 3 * N_SLOTS + 3
    df = pd.read_csv(io.StringIO(flat), header=None, dtype=np.float64,
                     na_values=["NaN"], engine="c")
    if df.shape[1] != width:
        raise ValueError("ATD line has {} fields, expected {}".format(df.shape[1], width))
    arr = df.to_numpy()
    video = arr[:, 0].astype(np.int64) - 1
    if video.min() < 0 or video.max() >= n_frames:
        raise ValueError("ATD frame ids outside 1..{}".format(n_frames))

    length, wide = pitch
    players = arr[:, 1:1 + 3 * N_SLOTS].reshape(len(arr), N_SLOTS, 3)
    ball = arr[:, 1 + 3 * N_SLOTS:]

    out = {k: np.full((n_frames, N_SLOTS), np.nan, np.float32) for k in ("px", "py", "ps")}
    out.update({k: np.full(n_frames, np.nan, np.float32) for k in ("bx", "by", "bs")})
    out["px"][video] = players[:, :, 0] * length
    out["py"][video] = players[:, :, 1] * wide
    out["ps"][video] = players[:, :, 2]
    out["bx"][video] = ball[:, 0] * length
    out["by"][video] = ball[:, 1] * wide
    out["bs"][video] = ball[:, 2]
    return out


def load(game=GAME, rebuild=False):
    """Arrays for the tactical ATD, parsed once and cached."""
    if os.path.isfile(CACHE) and not rebuild:
        with np.load(CACHE) as z:
            return {k: z[k] for k in z.files}
    meta, raw = require(["atd_meta", "atd_raw"], game)
    with open(raw) as fh:
        text = fh.read()
    out = parse_lines(text, source_frames("tactical", game), GAMES[game]["pitch"])
    out["team"] = slot_teams(meta)
    os.makedirs(os.path.dirname(CACHE), exist_ok=True)
    np.savez_compressed(CACHE, **out)
    return out


if __name__ == "__main__":
    d = load(rebuild="--rebuild" in sys.argv)
    n = len(d["bx"])
    print("frames", n, "| home slots", int((d["team"] == 0).sum()),
          "| away slots", int((d["team"] == 1).sum()))
    print("ball tracked on {:.1f}% of frames".format(100 * np.isfinite(d["bx"]).mean()))
