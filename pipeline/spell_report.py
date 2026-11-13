"""Per-spell diff report: where does a submission lose spell continuity on the public window?

For every GT transition: its type (H<->A, alive->D, D->alive), whether the ball was visible
around it, and whether the prediction has a transition of the same target within +/-1 s.
For every predicted transition with no GT transition nearby (spurious): ball visible?, ball speed
(pass flight?), contested (both teams within 2 m)?

    python pipeline/spell_report.py evaluation/results/baseline_smooth50.csv [more.csv ...]
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "evaluation"))
import possession_metrics as pm  # noqa: E402
from possession import atd as atd_mod, features as F  # noqa: E402

GT_PATH = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")
TOL = 25  # 1 s


def _filled(pred, gt):
    """Apply the scorer's own abstention fill so we analyse what spellF1 actually sees."""
    gt_codes = pm.label_codes(gt)
    runs = pm.valid_runs(gt_codes)
    return pm.fill_abstentions(pm.label_codes(pred), runs), gt_codes, runs


def transitions(lab, lo, hi):
    idx = np.flatnonzero(lab[lo + 1:hi + 1] != lab[lo:hi]) + lo + 1
    return [(int(t), lab[t - 1], lab[t]) for t in idx]


def kind(a, b):
    if "D" == a:
        return "D->alive"
    if "D" == b:
        return "alive->D"
    return "H<->A"


def report(pred_path, ctx):
    gt = pd.read_csv(GT_PATH)["label"].to_numpy()
    pred = pd.read_csv(pred_path)["label"].astype(str).str.strip().str.upper().to_numpy()
    filled, _, _ = _filled(pred, gt)
    lab_pred = np.array(pm.LABELS)[filled]
    lo, hi = 11185, 18684
    gt_t = transitions(gt, lo, hi)
    pr_t = transitions(lab_pred, lo, hi)
    vis, near_h, near_a, bv = ctx

    rows = []
    for t, a, b in gt_t:
        w = slice(t - TOL, t + TOL)
        hit = any(abs(p - t) <= TOL and pb == b for p, _, pb in pr_t)
        rows.append({"frame": t, "kind": kind(a, b), "to": b,
                     "ball_vis_1s": round(float(vis[w].mean()), 2), "detected": hit})
    gtd = pd.DataFrame(rows)

    spur = []
    for p, a, b in pr_t:
        if any(abs(p - t) <= TOL for t, _, _ in gt_t):
            continue
        w = slice(p - TOL, p + 1)
        spur.append({
            "frame": p, "kind": kind(a, b),
            "ball_invisible": not vis[p],
            "pass_flight": bool(np.nanmax(np.r_[bv[w], 0]) > 12),
            "contested": bool(vis[p] and near_h[p] < 2 and near_a[p] < 2),
        })
    spd = pd.DataFrame(spur, columns=["frame", "kind", "ball_invisible", "pass_flight", "contested"])
    return gtd, spd


def context():
    atd = atd_mod.load()
    f = F.build(atd)
    return f["ball_vis"], f["d_home"], f["d_away"], f["ball_v"]


def summarise(name, gtd, spd):
    out = ["### {}".format(name), "",
           "GT transitions detected within 1 s: **{}/{}**".format(int(gtd.detected.sum()), len(gtd)), ""]
    by = gtd.groupby("kind").agg(n=("detected", "size"), detected=("detected", "sum"),
                                 ball_vis=("ball_vis_1s", "mean")).reset_index()
    out.append("| GT transition | n | detected | mean ball visibility ±1 s |")
    out.append("|---|---|---|---|")
    for r in by.itertuples():
        out.append("| {} | {} | {} | {:.2f} |".format(r.kind, r.n, int(r.detected), r.ball_vis))
    missed = gtd[~gtd.detected]
    out.append("")
    out.append("Missed GT transitions (frame, kind, ball visibility): " + (", ".join(
        "{} {} ({:.2f})".format(r.frame, r.kind, r.ball_vis_1s) for r in missed.itertuples()) or "none"))
    out.append("")
    n = len(spd)
    out.append("Spurious predicted transitions: **{}**".format(n))
    if n:
        out.append("")
        out.append("| cause flag at the spurious flip | count | share |")
        out.append("|---|---|---|")
        for col, label in (("ball_invisible", "ball invisible"), ("pass_flight", "ball in fast flight (>12 m/s) in previous 1 s"),
                           ("contested", "contested (both teams within 2 m)")):
            out.append("| {} | {} | {:.0%} |".format(label, int(spd[col].sum()), spd[col].mean()))
        kinds = spd["kind"].value_counts()
        out.append("| kind: " + ", ".join("{} {}".format(k, v) for k, v in kinds.items()) + " | | |")
    out.append("")
    return "\n".join(out)


if __name__ == "__main__":
    ctx = context()
    for path in sys.argv[1:]:
        gtd, spd = report(path, ctx)
        print(summarise(os.path.splitext(os.path.basename(path))[0], gtd, spd))
