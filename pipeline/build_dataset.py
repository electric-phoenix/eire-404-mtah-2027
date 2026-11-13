"""Frame-aligned dataset builder (brief Step 1).

    python -m pipeline.build_dataset

Writes (both under data/, gitignored -- never commit or publish them):
  data/processed/frames.parquet          one row per (frame, player_id); ball is player_id 0
  data/processed/frame_features.parquet  one row per frame; model features + gt_public (VALIDATION
                                         ONLY -- excluded from features.ALL_FEATURES)
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import half_video_blocks, GAME  # noqa: E402
from mtah_io import atd as A, homography as Hm  # noqa: E402
from possession import tagging as T  # noqa: E402
from pipeline import features as FE  # noqa: E402

OUT = os.path.join(ROOT, "data", "processed")
GT_PUBLIC = os.path.join(ROOT, "evaluation", "gt", "BAR-ATH_possession_gt_public.csv")


def team_side_by_id(meta):
    side = {0: "BALL"}
    side.update({p["player_id"]: p["team_side"] for p in meta["players"]})
    return side


def image_coords(w, H, pitch):
    """Homography: normalised pitch -> image pixels (DATA.md; the decimetre hypothesis failed the
    landmark check). Returns (N, 47) arrays, column 0 = ball."""
    L, W = pitch
    x = np.concatenate([w["bx"][:, None], w["px"]], axis=1) / L
    y = np.concatenate([w["by"][:, None], w["py"]], axis=1) / W
    Hx = H[:, None, :, :]
    u = Hx[..., 0, 0] * x + Hx[..., 0, 1] * y + Hx[..., 0, 2]
    v = Hx[..., 1, 0] * x + Hx[..., 1, 1] * y + Hx[..., 1, 2]
    s = Hx[..., 2, 0] * x + Hx[..., 2, 1] * y + Hx[..., 2, 2]
    return (u / s).astype(np.float32), (v / s).astype(np.float32)


def build(write=True):
    meta = A.parse_atd_metadata()
    w = A.load_wide()
    H, reproj, valid = Hm.load_homography()
    tags = T.events()
    halves = half_video_blocks(GAME)
    n = len(w["bx"])
    assert len(H) == n, "homography and ATD frame counts differ"

    frame_df = FE.frame_features(w, tags, halves)
    gt = pd.read_csv(GT_PUBLIC)["label"].to_numpy()
    frame_df["gt_public"] = gt                      # validation only
    frame_df["half"] = 0
    for h, a, b in halves:
        frame_df.loc[a:b, "half"] = h
    frame_df["homography_reprojection_error"] = reproj.astype(np.float32)
    frame_df["homography_valid"] = valid

    long_df = A.wide_to_long(w)
    side = team_side_by_id(meta)
    long_df["team_side"] = pd.Categorical(long_df["player_id"].map(side), ["H", "A", "BALL"])
    u, v = image_coords(w, H, meta["pitch_metres"])
    long_df["x_img"] = u.ravel()
    long_df["y_img"] = v.ravel()
    long_df = FE.player_features(long_df, frame_df)
    long_df["gt_public"] = pd.Categorical(gt[long_df["frame"].to_numpy()], ["H", "A", "D", "X"])

    if write:
        os.makedirs(OUT, exist_ok=True)
        long_df.to_parquet(os.path.join(OUT, "frames.parquet"), index=False)
        frame_df.to_parquet(os.path.join(OUT, "frame_features.parquet"), index=False)
    return long_df, frame_df


if __name__ == "__main__":
    long_df, frame_df = build()
    print("frames.parquet          rows {:,}  cols {}".format(len(long_df), len(long_df.columns)))
    print("frame_features.parquet  rows {:,}  cols {}".format(len(frame_df), len(frame_df.columns)))
    print("columns:", list(long_df.columns))
