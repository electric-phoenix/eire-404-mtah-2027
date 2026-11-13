"""Probabilities -> labels -> validated CSV, for each decoder variant (brief Steps 2-4).

    python -m pipeline.run --proba full --variants classical viterbi hybrid
    python -m pipeline.run --proba full --variants hybrid --out submission.csv

Variants
  classical        argmax of the LightGBM probabilities
  classical_s50    argmax + centred majority vote +/-50 (same smoothing as the published floor)
  viterbi          4-state Viterbi with Smart Tagging modifiers, no rules
  viterbi_nomod    Viterbi with uniform (unmodified) transitions
  hybrid           Viterbi + all four rules
  hybrid-<rule>    hybrid without one rule (ablation): dead_ball, pass_flight, invisible, contested
  hybrid_min25     hybrid + 1 s minimum spell length
"""
import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from games import GAME, half_video_blocks  # noqa: E402
from pipeline import decode as DC  # noqa: E402
from possession import tagging as T  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "evaluation", "results")
RULE_ORDER = ["dead_ball", "pass_flight", "invisible", "contested"]


def majority(codes, k):
    out = codes.copy()
    cs = [np.r_[0, np.cumsum(codes == c)] for c in range(4)]
    i = np.arange(len(codes))
    lo, hi = np.maximum(0, i - k), np.minimum(len(codes), i + k + 1)
    counts = np.stack([c[hi] - c[lo] for c in cs], axis=1)
    out[:] = counts.argmax(axis=1)
    return out


def produce(variant, proba, f, tags, prior):
    n = len(proba)
    labels = np.full(n, "X", dtype="<U1")
    log_em = np.log(np.maximum(proba, 1e-6)) - np.log(prior)
    for _, a, b in half_video_blocks(GAME):
        sl = slice(a, b + 1)
        fs = {k: v[sl] for k, v in f.items()}
        tags_h = [dict(t, a=t["a"] - a) for t in tags if a - 500 <= t["a"] <= b]
        if variant.startswith("classical"):
            path = proba[sl].argmax(axis=1).astype(np.int8)
            if variant == "classical_s50":
                path = majority(path, 50)
        else:
            start = np.log(np.array([0.02, 0.02, 0.94, 0.02]))     # halves begin dead
            path = DC.viterbi(log_em[sl], tags_h, use_modifiers=(variant != "viterbi_nomod"),
                              start=start)
            if variant.startswith("hybrid"):
                skip = variant.split("-", 1)[1] if "-" in variant else None
                for r in RULE_ORDER:
                    if r != skip:
                        path = DC.RULES[r](path, fs)
                if variant == "hybrid_min25":
                    path = DC.enforce_min_spell(path, 25)
        labels[sl] = DC.STATES[path]
    return labels


def write(path, labels):
    with open(path, "w", newline="\n") as fh:
        fh.write("frame,label\n")
        fh.writelines("{},{}\n".format(i, l) for i, l in enumerate(labels))
    problem = pm.validate_submission(pm.read_label_csv(path), pm_n())
    if problem:
        raise SystemExit("INVALID {}: {}".format(path, problem))
    return path


def pm_n():
    return 199319


def load(proba_tag):
    proba = np.load(os.path.join(PROC, "proba_{}.npy".format(proba_tag)))
    df = pd.read_parquet(os.path.join(PROC, "frame_features.parquet"))
    f = {c: df[c].to_numpy() for c in df.columns if c not in ("gt_public",)}
    with open(os.path.join(PROC, "train_{}.json".format(proba_tag))) as fh:
        counts = json.load(fh)["label_counts"]
    prior = np.array([counts[c] for c in "HADX"], float)
    prior = prior / prior.sum()
    return proba, f, T.events(), prior


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--proba", default="full")
    ap.add_argument("--variants", nargs="+", default=["hybrid"])
    ap.add_argument("--out", help="write the (single) variant here instead of evaluation/results/")
    args = ap.parse_args()
    proba, f, tags, prior = load(args.proba)
    for v in args.variants:
        labels = produce(v, proba, f, tags, prior)
        path = args.out or os.path.join(RESULTS, "lgbm_{}_{}.csv".format(args.proba, v))
        write(path, labels)
        print("valid  {}".format(os.path.relpath(path, ROOT)))


if __name__ == "__main__":
    main()
