"""Report for the throw-in / corner restart tree (pipeline/restart_tree.py).

    python scripts/restart_report.py

Three things, in order:
  1. How often each leaf runs on the full match and how much it changes (needs no labels).
  2. How each submission file labels the team right after a corner, against the Smart Tagging
     corner team. The tag is automatic, not ground truth, and there are only 12 corners.
  3. How often a file gives the ball straight back to the team that last controlled it after a
     throw-in. There is no reference for throw-ins, so this is a rate, not an accuracy.

The public ground truth contains NO throw-in or corner, so it cannot score any of this; the
public spellF1 / macroF1 columns come from scripts/board.py and only show nothing else moved.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, half_video_blocks  # noqa: E402
from pipeline import dead_tree as DT  # noqa: E402
from pipeline import restart_tree as RT  # noqa: E402
from possession import tagging as T  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
RES = os.path.join(ROOT, "evaluation", "results")
FILES = ["baseline_smooth50", "lgbm_full_hybrid", "lgbm_full_hybrid_tree",
         "lgbm_full_hybrid_tree_restart", "lgbm_full_hybrid_tree_restart-literal_corner"]
CODE = {"H": 0, "A": 1, "D": 2, "X": 3}


def load(name):
    return pd.read_csv(os.path.join(RES, name + ".csv"))["label"].to_numpy()


def first_team_after_stoppage(lab, a, lo=150, hi=250):
    """Team (0/1) on the first alive frame after the first D run near frame a, else None."""
    seg = lab[max(a - lo, 0):a + hi] == "D"
    idx = np.flatnonzero(seg)
    if not len(idx):
        return None
    e = max(a - lo, 0) + idx[0]
    while e < len(lab) and lab[e] == "D":
        e += 1
    after = lab[e:e + 75]
    after = after[np.isin(after, ["H", "A"])]
    return CODE[after[0]] if len(after) else None


def main():
    df = pd.read_parquet(os.path.join(PROC, "frame_features.parquet"))
    f = {c: df[c].to_numpy() for c in df.columns if c != "gt_public"}
    tags = T.events()
    blocks = half_video_blocks(GAME)

    # ---- 1. leaf activity, replayed on the hybrid_tree labels
    base = np.array([CODE[x] for x in load("lgbm_full_hybrid_tree")], np.int8)
    log = []
    for _, a, b in blocks:
        tags_h = [dict(t, a=t["a"] - a) for t in tags if a - 500 <= t["a"] <= b]
        fs = {k: v[a:b + 1] for k, v in f.items()}
        part = []
        RT.apply(base[a:b + 1], fs, tags_h, log=part)
        log += part
    lg = pd.DataFrame(log)
    print("1. LEAF ACTIVITY (full match, replayed on hybrid_tree labels)\n")
    print("   {:<20} {:>10} {:>9} {:>11} {:>14} {:>9}".format(
        "leaf", "stoppages", "team set", "ball blind", "frames changed", "mean s"))
    for leaf in RT.LEAVES:
        g = lg[lg.leaf == leaf]
        print("   {:<20} {:>10} {:>9} {:>11} {:>14} {:>9}".format(
            leaf, len(g), int(g.team.notna().sum()), int((g.blind > 0).sum()),
            int(g.changed.sum()), "{:.1f}".format(g.changed.sum() / max(1, (g.changed > 0).sum()) / 25)))
    total = int(sum(len(x) for x in [lg]) and (base != 2).sum())
    print("\n   D stoppages in the file: {}   entered the tree: {}   frames changed: {} "
          "({:.2%} of alive frames)\n".format(
              sum(len(DT.runs(base[a:b + 1] == 2)[0]) for _, a, b in blocks), len(lg),
              int(lg.changed.sum()), lg.changed.sum() / max(total, 1)))

    # ---- 2 and 3. what each file does after a corner / throw-in
    corners = [t for t in tags if t["code"] == "set pieces" and t["type"] == "corner kick"
               and t["team"] is not None]
    throws = [t for t in tags if t["code"] == "set pieces" and t["type"] == "throw-in"]
    print("2. CORNERS: team on the first alive frame vs the Smart Tagging corner team "
          "(n={}, tag is automatic)\n".format(len(corners)))
    print("3. THROW-INS: share of throw-ins where the SAME team as the last controller gets the "
          "ball back (n={})\n".format(len(throws)))
    print("   {:<48} {:>10} {:>10} {:>12}".format("file", "corner==tag", "(found)", "throw: same"))
    for name in FILES:
        lab = load(name)
        ok = tot = 0
        for t in corners:
            team = first_team_after_stoppage(lab, t["a"])
            if team is None:
                continue
            tot += 1
            ok += team == t["team"]
        same = n = 0
        for t in throws:
            team = first_team_after_stoppage(lab, t["a"])
            for h, a, b in blocks:
                if a <= t["a"] <= b:
                    seg = lab[a:t["a"]]
                    s = np.flatnonzero(f["since_ctrl_H"][a:t["a"]] == 0)
                    r = np.flatnonzero(f["since_ctrl_A"][a:t["a"]] == 0)
                    last = None
                    if len(s) or len(r):
                        last = 0 if (len(s) and (not len(r) or s[-1] > r[-1])) else 1
                    if team is not None and last is not None:
                        n += 1
                        same += team == last
        print("   {:<48} {:>10} {:>10} {:>12}".format(
            name, "{}/{}".format(ok, tot), "{:.0%}".format(ok / max(tot, 1)),
            "{}/{} = {:.0%}".format(same, n, same / max(n, 1))))


if __name__ == "__main__":
    main()
