"""Side-by-side comparison of the models we might submit. Prints a Markdown table.

    python scripts/compare_candidates.py

WHAT THIS IS FOR (plain words)
------------------------------
We have several versions ("variants") of the model. This script puts the ones we are choosing
between in one table, using two kinds of evidence:

  1. The OFFICIAL scorer on the public 5 minutes (evaluation/run_evaluation.py), with
     --bootstrap 1000. "Bootstrap" re-scores the file 1000 times on random resamples of the
     possessions and reports the range the score falls in 90% of the time (the "CI90").
     If two models' ranges overlap a lot, the public window cannot tell them apart.
     The public window has only 17 possessions, so this is a sanity check, NOT a way to choose.

  2. Label-free numbers over the WHOLE match (no answers needed, so they are allowed and they are
     what we choose on):
       possessions     how many separate possessions the model says the match had
       median s        typical possession length; real football is nearer 14 s than 6 s
       stoppages       how many separate dead-ball stretches it found
       <2 s, gap<3 s   tiny stoppages, and stoppages almost touching each other: both are signs
                       that one real stoppage has been chopped into pieces

  Two public dead-ball numbers are also shown, as a check only:
       D prec          of the frames we call dead, the share that really are dead
       D rec           of the really dead frames, the share we found

Edit CANDIDATES below to change what is compared. Each entry is (readable name, file name in
evaluation/results/ without ".csv"). Build the files first with
    python -m pipeline.run --proba full --variants <variant> ...
"""
import os
import subprocess
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
from games import GAME, half_video_blocks  # noqa: E402
from pipeline import dead_tree as DT  # noqa: E402

RES = os.path.join(ROOT, "evaluation", "results")
GT = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
FPS = 25

CANDIDATES = [
    ("Baseline (organisers' floor)", "baseline_smooth50"),
    ("Model v1 (frozen)", "lgbm_full_hybrid"),
    ("Model v2 (dead-ball tree)", "lgbm_full_hybrid_tree"),
    ("Model v2 + tidy", "lgbm_full_hybrid_tree_tidy"),
    ("Model v2 + tidy + 1 s minimum", "lgbm_full_hybrid_tree_tidy-min25"),
    ("Reworked tree, hard, merge+short", "lgbm_full_hybrid_stoppage-midpoint-edges"),
    ("Reworked tree, SOFT, merge+short", "lgbm_full_hybrid_stoppage_soft-midpoint-edges"),
    ("Original tree, SOFT, merge+short", "lgbm_full_hybrid_stoppage_soft-midpoint-edges-rework"),
    ("Model v3 (frozen): soft, merge+short, 1 s min", "lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25"),
    ("v3 + whistle (threshold 20)", "lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle"),
    ("v3 + whistle, looser (16)", "lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle16"),
    ("v3 + whistle, stricter (24)", "lgbm_full_hybrid_stoppage_soft-midpoint-edges-min25-whistle24"),
]


def official_board():
    """Run the official scorer once on everything in evaluation/results/ and read its table."""
    subprocess.run([sys.executable, os.path.join(ROOT, "evaluation", "run_evaluation.py"),
                    "--bootstrap", "1000", "--no-detail"], check=True, stdout=subprocess.DEVNULL)
    return pd.read_csv(os.path.join(ROOT, "evaluation", "scoreboard.csv")).set_index("submission")


def load(name):
    return pd.read_csv(os.path.join(RES, name + ".csv"))["label"].to_numpy().astype(str)


def possessions(lab):
    """Possession count and median length over the whole match, counted the way the scorer does.

    The scorer first replaces every X ("no answer") with the previous label in the same half, so we
    do the same. A possession is then an unbroken run of H, or of A, inside a half.
    """
    lengths = []
    for _, a, b in half_video_blocks(GAME):
        seg = pd.Series(np.where(lab[a:b + 1] == "X", None, lab[a:b + 1])).ffill().bfill()
        seg = seg.to_numpy().astype(str)
        change = np.flatnonzero(seg[1:] != seg[:-1]) + 1
        starts, ends = np.r_[0, change], np.r_[change, len(seg)]
        lengths += [(e - s) / FPS for s, e in zip(starts, ends) if seg[s] in ("H", "A")]
    return len(lengths), float(np.median(lengths)) if lengths else 0.0


def stoppages(lab):
    """Number of dead-ball runs, how many are under 2 s, and how many are under 3 s apart."""
    lengths, gaps = [], []
    for _, a, b in half_video_blocks(GAME):
        s, e = DT.runs(lab[a:b + 1] == "D")
        lengths += list((e - s) / FPS)
        gaps += list((s[1:] - e[:-1]) / FPS)
    lengths, gaps = np.array(lengths), np.array(gaps)
    return len(lengths), int((lengths < 2).sum()), int((gaps < 3).sum())


def dead_check(lab, gt):
    """Public-window dead-ball precision and recall (a check, never used to choose)."""
    m = gt != "X"
    hit = ((lab[m] == "D") & (gt[m] == "D")).sum()
    return hit / max((lab[m] == "D").sum(), 1), hit / max((gt[m] == "D").sum(), 1)


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", help="compare these result files (names without .csv) "
                    "instead of CANDIDATES; each is labelled by its file name")
    args = ap.parse_args()
    candidates = [(n, n) for n in args.files] if args.files else CANDIDATES
    board = official_board()
    gt = pd.read_csv(GT)["label"].to_numpy().astype(str)
    print("| Model | file | public spellF1 [CI90] | public macroF1 | possessions | median s |"
          " stoppages | <2 s | gap<3 s | D prec | D rec |")
    print("|---|---|---|---|---|---|---|---|---|---|---|")
    for readable, name in candidates:
        if not os.path.exists(os.path.join(RES, name + ".csv")):
            print("| {} | `{}` | (file missing: build it first) |||||||||".format(readable, name))
            continue
        lab = load(name)
        r = board.loc[name]
        n_poss, med = possessions(lab)
        n_stop, short, close = stoppages(lab)
        prec, rec = dead_check(lab, gt)
        print("| {} | `{}` | {:.3f} [{:.2f}, {:.2f}] | {:.3f} | {} | {:.1f} | {} | {} | {} | {:.2f} | {:.2f} |"
              .format(readable, name, r.spell_f1_mean, r.spell_bootstrap_low,
                      r.spell_bootstrap_high, r.macro_f1, n_poss, med, n_stop, short, close,
                      prec, rec))


if __name__ == "__main__":
    main()
