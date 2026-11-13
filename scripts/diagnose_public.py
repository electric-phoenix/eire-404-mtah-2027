"""Correctness check: how do the raw signals behave inside each GT class on the public window?

Prints aggregates only. Used to confirm what H/A/D mean in terms of the tracking -- not to fit
thresholds (RULES.md forbids tuning on the public window).
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import atd as atd_mod, features as F  # noqa: E402

atd = atd_mod.load()
f = F.build(atd)
gt = pd.read_csv(os.path.join(ROOT, "evaluation/gt/BAR-ATH_possession_gt_public.csv"))["label"].to_numpy()
graded = F.graded_mask(len(gt))
print("graded-frame summary over the whole match (no GT):")
print("  ball visible {:.1%}  oob {:.1%}".format(f["ball_vis"][graded].mean(), f["oob"][graded].mean()))
vv = f["ball_v"][graded & f["ball_vis"]]
print("  ball speed from positions m/s: median {:.2f} p90 {:.2f}".format(np.nanmedian(vv), np.nanpercentile(vv, 90)))
bs = f["ball_s_file"][graded & f["ball_vis"]]
print("  ball speed from file:          median {:.2f} p90 {:.2f}".format(np.nanmedian(bs), np.nanpercentile(bs, 90)))
print("  player speed file: median {:.2f}".format(np.nanmedian(f["p_speed"][graded])))

rows = []
for lab in "HAD":
    m = gt == lab
    vis = m & f["ball_vis"]
    near_h = (f["d_home"] < f["d_away"])[vis]
    rows.append({
        "label": lab, "n": int(m.sum()),
        "ball_vis": f["ball_vis"][m].mean(),
        "oob|vis": f["oob"][vis].mean(),
        "nearest_is_H|vis": near_h.mean(),
        "min_d_med": np.median(np.minimum(f["d_home"], f["d_away"])[vis]),
        "ball_v_med": np.nanmedian(f["ball_v"][vis]),
        "p_speed_med": np.nanmedian(f["p_speed"][m]),
    })
print(pd.DataFrame(rows).round(3).to_string(index=False))
