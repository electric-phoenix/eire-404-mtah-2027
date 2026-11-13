"""Official public-window board (run_evaluation.py --bootstrap 1000) + full-match spell stats.

    python scripts/board.py [substring filter]
"""
import glob
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession.features import graded_mask  # noqa: E402

subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                "--bootstrap", "1000", "--no-detail"], check=True, stdout=subprocess.DEVNULL)
board = pd.read_csv(os.path.join(ROOT, "evaluation", "scoreboard.csv"))
g = graded_mask(199319)


def full_stats(name):
    lab = pd.read_csv(os.path.join(ROOT, "evaluation", "results", name + ".csv"))["label"].to_numpy()
    lab = lab.astype(str)
    # scorer-style fill of X with the previous label inside each half
    s = pd.Series(np.where(lab == "X", None, lab))
    lab = s.ffill().bfill().to_numpy().astype(str)
    lab = np.where(g, lab, "D")
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    st, en = np.r_[0, change], np.r_[change, len(lab)]
    d = np.array([(e - s_) / 25 for s_, e in zip(st, en) if lab[s_] in ("H", "A")])
    return len(d), float(np.median(d)) if len(d) else 0.0


rows = []
flt = sys.argv[1] if len(sys.argv) > 1 else ""
for r in board.itertuples():
    if flt not in r.submission:
        continue
    n, med = full_stats(r.submission)
    rows.append({"submission": r.submission, "spellF1": r.spell_f1_mean,
                 "CI90": "[{:.2f}, {:.2f}]".format(r.spell_bootstrap_low, r.spell_bootstrap_high),
                 "macroF1": r.macro_f1, "pub_poss": r.poss, "full_poss": n, "full_median_s": med})
print(pd.DataFrame(rows).round(3).to_string(index=False))
