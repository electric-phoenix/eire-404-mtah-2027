"""Per-leaf report for the dead-ball decision tree (pipeline/dead_tree.py).

    python scripts/dead_tree_report.py

The leaf ORDER is fixed in advance by how certain each kind of evidence is (see dead_tree.py).
This script is a check on that order, not a way to choose it: the public ground truth has 17
possessions, and tuning on it is forbidden (RULES.md). It prints, for every leaf:

  frames/share   how much of the graded match it labels dead (full match, no labels needed)
  segs/mean s    how many separate stoppages it makes and how long they are
  new            share of its frames the frozen `hybrid` file does not already call D
  pub frames     its frames inside the public 5 minutes
  pub prec       of those, the share that really are dead (a sanity check, small sample)
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
from possession import tagging as T  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
HYBRID = os.path.join(ROOT, "evaluation", "results", "lgbm_full_hybrid.csv")
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def main():
    df = pd.read_parquet(os.path.join(PROC, "frame_features.parquet"))
    f = {c: df[c].to_numpy() for c in df.columns if c != "gt_public"}
    tags = T.events()
    n = len(df)

    leaf = np.zeros(n, np.int8)
    graded = np.zeros(n, bool)
    for _, a, b in half_video_blocks(GAME):
        sl = slice(a, b + 1)
        tags_h = [dict(t, a=t["a"] - a) for t in tags if a - 500 <= t["a"] <= b]
        fs = {k: v[sl] for k, v in f.items()}
        leaf[sl] = DT.classify(fs, DT.pre_kickoff_mask(b + 1 - a, tags_h))
        graded[sl] = True

    hybrid = pd.read_csv(HYBRID)["label"].to_numpy() if os.path.exists(HYBRID) else None
    gt = pd.read_csv(GT).set_index("frame")["label"] if os.path.exists(GT) else None
    in_pub = np.zeros(n, bool)
    if gt is not None:
        gt_full = np.full(n, "X", dtype="<U1")
        gt_full[gt.index.to_numpy()] = gt.to_numpy()
        in_pub = gt_full != "X"                    # the public window = frames with a public label
        has_gt = in_pub

    print("graded frames: {}   ({:.1f} min)\n".format(graded.sum(), graded.sum() / 25 / 60))
    head = "{:>4}  {:<20} {:>8} {:>7} {:>5} {:>7} {:>6} {:>10} {:>8}"
    print(head.format("rank", "leaf", "frames", "share", "segs", "mean s", "new", "pub frames",
                      "pub prec"))
    total = 0
    for i, name in enumerate(DT.LEAVES, start=1):
        m = leaf == i
        starts, ends = DT.runs(m)
        k = int(m.sum())
        total += k
        new = ("{:.0%}".format(float((hybrid[m] != "D").mean())) if hybrid is not None and k
               else "-")
        pub = m & in_pub
        prec = ("{:.0%}".format(float((gt_full[pub & has_gt] == "D").mean()))
                if gt is not None and (pub & has_gt).any() else "-")
        print(head.format(i, name, k, "{:.1%}".format(k / graded.sum()), len(starts),
                          "{:.1f}".format((ends - starts).mean() / 25) if len(starts) else "-",
                          new, int(pub.sum()), prec))
    print("\n  all leaves: {} frames = {:.1%} of graded".format(total, total / graded.sum()))
    print("  note: 'pub prec' is a sanity check on a tiny sample; it did not set the order.")


if __name__ == "__main__":
    main()
