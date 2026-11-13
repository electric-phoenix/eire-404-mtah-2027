"""One command: raw data -> validated submission CSV.

    .venv/Scripts/python scripts/make_submission.py [--name hmm_v1]

Writes evaluation/results/<name>.csv (gitignored), validates it with the official checks, and
prints unsupervised plausibility stats for the FULL match (possession count / duration), which is
what we compare versions on -- not the public 5-minute score.
"""
import argparse
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import atd as atd_mod, model  # noqa: E402
from possession.features import graded_mask, FPS  # noqa: E402


def spell_stats(labels, graded):
    lab = np.where(graded, labels, "D")
    alive = np.isin(lab, ["H", "A"])
    change = np.flatnonzero(lab[1:] != lab[:-1]) + 1
    starts = np.r_[0, change]
    ends = np.r_[change, len(lab)]
    runs = [(s, e) for s, e in zip(starts, ends) if alive[s]]
    dur = np.array([(e - s) / FPS for s, e in runs])
    return len(runs), dur


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--name", default="hmm_v1")
    args = ap.parse_args()

    labels = model.predict(atd_mod.load())
    path = os.path.join(ROOT, "evaluation", "results", args.name + ".csv")
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))

    frame = pm.read_label_csv(path)
    problems = pm.validate_submission(frame, len(labels))
    print("validation:", "OK" if not problems else problems)

    g = graded_mask(len(labels))
    n, dur = spell_stats(labels, g)
    share = {k: float((labels[g] == k).mean()) for k in "HAD"}
    print("full match: {} possessions, mean {:.1f} s, median {:.1f} s, <2 s: {}".format(
        n, dur.mean(), np.median(dur), int((dur < 2).sum())))
    print("graded share  H {H:.3f}  A {A:.3f}  D {D:.3f}".format(**share))
    print("wrote", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
