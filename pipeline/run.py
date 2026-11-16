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
  hybrid_tree      hybrid with the dead-ball override replaced by the dead-ball decision tree
  hybrid_tree-<leaf>  the tree without one leaf (ablation), see pipeline/dead_tree.py LEAVES
  hybrid_tree_crowd            hybrid_tree + crowd rule (pipeline/cluster_tree.py): ball untracked in a
                               pack of players -> the team with more players in the pack
  hybrid_stoppage              hybrid_tree with the reworked leaves + stoppage clean-up (merge close
                               stoppages, drop short ones, tag-midpoint and player-speed edges)
  hybrid_stoppage_soft         same, but the dead-ball leaves are soft log-odds on D in the Viterbi
                               emissions instead of a hard override (pipeline/stoppage_tree.py)
  hybrid_stoppage[_soft]-<x>   ablation, x in merge, short, midpoint, edges, rework
  hybrid_tree_tidy             hybrid_tree + two clean-ups of the stoppages, nothing else:
                               merge stoppages less than 3 s apart, drop untagged ones under 2 s
  hybrid_tree_tidy-<x>         ablation, x is merge or short (switches that one step off)
  hybrid_tree_restart          hybrid_tree + throw-in / corner restart tree (pipeline/restart_tree.py)

Combining options (read this if you are new here)
  A variant name is a BASE name, then optional "-option" pieces, e.g.
      hybrid_stoppage_soft-midpoint-edges
      `-----------------' `------' `---'
          base model        option  option
  Most options SWITCH A STEP OFF (an "ablation": run the model without one piece to see what that
  piece was worth). Any number of them can be chained. One option ADDS a step instead:
      min25   after everything else, no possession or stoppage may be shorter than 1 second
              (25 frames); a shorter one is absorbed into the label before it.
      whistle referee-whistle evidence for dead ball (pipeline/whistle.py, needs
              python -m pipeline.audio first). Only on hybrid_stoppage_soft*.
              whistle16 / whistle24 use a looser / stricter whistle threshold (sensitivity check).
      foulteam  with a whistle option: after a foul whistle, nudge the restart towards the team
              that had the ball before it (the fouled team usually takes the free kick).
  Examples
      hybrid_tree_tidy-min25                 tidy model + 1 s minimum spell
      hybrid_stoppage_soft-midpoint-edges    soft dead-ball evidence + merge + drop-short only
  See variant_options() below for exactly how a name is split.
  hybrid_tree_restart-<leaf>   ...without one restart leaf; "-literal_corner" applies the plain
                               "other team restarts" idea to corners (ablation, conflicts with tags)
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
from pipeline import cluster_tree as CT  # noqa: E402
from pipeline import dead_tree as DT  # noqa: E402
from pipeline import decode as DC  # noqa: E402
from pipeline import restart_tree as RT  # noqa: E402
from pipeline import stoppage_tree as ST  # noqa: E402
from pipeline import whistle as WH  # noqa: E402
from possession import tagging as T  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
RESULTS = os.path.join(ROOT, "evaluation", "results")
RULE_ORDER = ["dead_ball", "pass_flight", "invisible", "contested"]

# Options that ADD a step rather than switch one off. Everything else after a "-" in a variant
# name is treated as "switch this step off".
ADD_ONS = {"min25", "whistle", "whistle16", "whistle24", "foulteam"}
MIN_SPELL_1S = 25          # frames; 25 frames = 1 second of video


def variant_options(variant):
    """Split a variant name into its option pieces.

    In plain words: everything after the FIRST "-" is a list of options separated by "-".
        "hybrid_tree_tidy"                     -> []
        "hybrid_tree_tidy-min25"               -> ["min25"]
        "hybrid_stoppage_soft-midpoint-edges"  -> ["midpoint", "edges"]
    Option names themselves use underscores (e.g. "literal_corner"), never "-", so splitting on
    "-" is safe. Before this helper existed only ONE option could be given per name.
    """
    return variant.split("-")[1:]


def switched_off(opts):
    """The options that switch a step off (all options except the add-ons like min25)."""
    return tuple(o for o in opts if o not in ADD_ONS)


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
    crowd = CT.load_or_build(PROC) if variant.startswith("hybrid_tree_crowd") else None
    whistle_opt = [o for o in variant_options(variant) if o.startswith("whistle")]
    if whistle_opt:
        from pipeline import audio as AU
        whistle_db = AU.load()["whistle_db"]
        whistle_thr = float(whistle_opt[0][7:] or WH.THRESHOLD_DB)
    labels = np.full(n, "X", dtype="<U1")
    log_em = np.log(np.maximum(proba, 1e-6)) - np.log(prior)
    for _, a, b in half_video_blocks(GAME):
        sl = slice(a, b + 1)
        fs = {k: v[sl] for k, v in f.items()}
        tags_h = [dict(t, a=t["a"] - a, b=t["b"] - a) for t in tags if a - 500 <= t["a"] <= b]
        if variant.startswith("classical"):
            path = proba[sl].argmax(axis=1).astype(np.int8)
            if variant == "classical_s50":
                path = majority(path, 50)
        else:
            start = np.log(np.array([0.02, 0.02, 0.94, 0.02]))     # halves begin dead
            em = log_em[sl]
            stoppage = variant.startswith("hybrid_stoppage")
            opts = variant_options(variant)          # e.g. ["midpoint", "edges", "min25"]
            if stoppage:
                fs["pre_kickoff"] = DT.pre_kickoff_mask(b + 1 - a, tags_h)
                leaf = DT.classify(fs, fs["pre_kickoff"], rework="rework" not in opts)
                if variant.startswith("hybrid_stoppage_soft"):
                    em = em.copy()
                    em[:, DC.D] += ST.soft_bonus(leaf)
                    if whistle_opt:
                        # Add-on: the referee's whistle (a "stop" whistle) nudges the frames up to
                        # the restart kick towards dead ball. See pipeline/whistle.py.
                        em[:, DC.D] += WH.bonus(whistle_db[sl], fs, whistle_thr)
                        if "foulteam" in opts:
                            em[:, [DC.H, DC.A]] += WH.restart_bonus(
                                whistle_db[sl], fs, em[:, [DC.H, DC.A]], whistle_thr)
            path = DC.viterbi(em, tags_h, use_modifiers=(variant != "viterbi_nomod"),
                              start=start)
            if stoppage:
                soft = variant.startswith("hybrid_stoppage_soft")
                for r in RULE_ORDER:
                    if r == "dead_ball":
                        path[(fs["pre_kickoff"] if soft else leaf > 0)] = DC.D
                    else:
                        path = DC.RULES[r](path, fs)
                # Clean up the stoppages (merge / drop short / move edges). Any of those steps
                # named in the options is skipped.
                path = ST.clean(path, fs, tags_h, disable=switched_off(opts))
                if "min25" in opts:
                    # Add-on: no spell (possession or stoppage) shorter than 1 s survives; it is
                    # relabelled with whatever came just before it.
                    path = DC.enforce_min_spell(path, MIN_SPELL_1S)
            elif variant.startswith("hybrid_tree"):
                restart = variant.startswith("hybrid_tree_restart")
                crowded = variant.startswith("hybrid_tree_crowd")
                tidy = variant.startswith("hybrid_tree_tidy")
                opts = variant_options(variant)
                # For plain "hybrid_tree-<leaf>" the options name dead-ball tree leaves to switch
                # off. For the other hybrid_tree_* models the options belong to their own add-on
                # step instead, so the dead-ball tree runs in full.
                off = () if (restart or crowded or tidy) else switched_off(opts)
                fs["pre_kickoff"] = DT.pre_kickoff_mask(len(path), tags_h)
                for r in RULE_ORDER:
                    path = (DC.rule_dead_tree(path, fs, off) if r == "dead_ball"
                            else DC.RULES[r](path, fs))
                if tidy:
                    # hybrid_tree_tidy, in plain words
                    # --------------------------------
                    # Start from our current best model (hybrid_tree). Its stoppage ("dead ball")
                    # labels come out in too many small pieces, so we tidy them in two steps:
                    #   1. MERGE: if two stoppages are less than 3 seconds apart, the "play" between
                    #      them is not real play. Treat both as one stoppage (label the gap D).
                    #   2. DROP SHORT: a stoppage shorter than 2 seconds, with no throw-in, corner
                    #      or other set-piece tag near it, is almost certainly a glitch. Remove it
                    #      and give those frames back to the team that had the ball.
                    # That is all. Nothing else moves: we do NOT shift where stoppages start or end
                    # (the tag-midpoint idea hurt dead-ball precision, so it is left out). It reads
                    # no new data: only the labels we already made and the tags we already have.
                    # Use "-merge" or "-short" to switch one step off and see what each one does.
                    path = ST.clean(path, fs, tags_h,
                                    disable=("midpoint", "edges") + switched_off(opts))
                    if "min25" in opts:
                        # Add-on "-min25": after tidying, any possession or stoppage shorter than
                        # 1 second is absorbed into the label before it. Real possessions and
                        # stoppages almost never last under a second, so these are leftovers.
                        path = DC.enforce_min_spell(path, MIN_SPELL_1S)
                if crowded:
                    path = CT.apply(path, fs, {k: v[sl] for k, v in crowd.items()},
                                    disable=tuple(opts))
                if restart:
                    path = RT.apply(path, fs, tags_h,
                                    disable=tuple(o for o in opts if o != "literal_corner"),
                                    literal_corner="literal_corner" in opts)
            elif variant.startswith("hybrid"):
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
        # Show the path relative to the repo when possible. On Windows relpath fails when the file
        # is on another drive (e.g. C: vs D:), so fall back to the full path.
        try:
            shown = os.path.relpath(path, ROOT)
        except ValueError:
            shown = path
        print("valid  {}".format(shown))


if __name__ == "__main__":
    main()
