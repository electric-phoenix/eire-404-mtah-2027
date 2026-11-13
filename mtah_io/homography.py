"""Homography loader and projector (brief Step 1).

The brief's working hypothesis was that H maps normalised pitch coordinates to DECIMETRES, so a
0.1 scale gives metres. `landmark_check()` tests that against the kickoff spot, as the brief asks.
DATA.md says H maps normalised pitch coordinates to IMAGE PIXELS. See reports/model.md for the
verdict. Pitch metres for modelling come from `normalised * pitch size` (mtah_io.atd), which needs
no homography.
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from games import require  # noqa: E402

HOMOGRAPHY_SCALE_TO_METERS = 0.1
KICKOFF_FRAME_1ST = 11185          # DATA.md / FAQ.md
PITCH_CENTRE_M = (52.5, 34.0)


def load_homography(csv_path=None):
    """-> (H (N,3,3) float64, reprojection_error (N,), valid (N,) bool), indexed by video frame."""
    csv_path = csv_path or require(["homography"])[0]
    df = pd.read_csv(csv_path)
    if not np.array_equal(df["frame"].to_numpy(), np.arange(len(df))):
        raise ValueError("homography frames are not 0..N-1 in order")
    H = df[["h00", "h01", "h02", "h10", "h11", "h12", "h20", "h21", "h22"]].to_numpy().reshape(-1, 3, 3)
    return H, df["reprojection_error"].to_numpy(), df["valid"].to_numpy().astype(bool)


def project(x_norm, y_norm, H, scale=HOMOGRAPHY_SCALE_TO_METERS):
    """Apply H to normalised pitch coordinates. Broadcasts over arrays; H is (3,3) or (N,3,3)."""
    x = np.asarray(x_norm, float)
    y = np.asarray(y_norm, float)
    H = np.asarray(H, float)
    if H.ndim == 2:
        u = H[0, 0] * x + H[0, 1] * y + H[0, 2]
        v = H[1, 0] * x + H[1, 1] * y + H[1, 2]
        w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    else:
        u = H[:, 0, 0] * x + H[:, 0, 1] * y + H[:, 0, 2]
        v = H[:, 1, 0] * x + H[:, 1, 1] * y + H[:, 1, 2]
        w = H[:, 2, 0] * x + H[:, 2, 1] * y + H[:, 2, 2]
    return u / w * scale, v / w * scale


def first_tracked_ball(bx_norm, by_norm, start=KICKOFF_FRAME_1ST, within=50):
    """(frame, x, y) of the first tracked ball in [start, start+within)."""
    for f in range(start, start + within):
        if np.isfinite(bx_norm[f]):
            return f, bx_norm[f], by_norm[f]
    raise ValueError("ball not tracked within {} frames of {}".format(within, start))


def landmark_check(H, bx_norm, by_norm):
    """Project the ball at the 1st-half kickoff; under the brief's hypothesis it lands near the
    pitch centre in metres. Returns a dict with the verdict."""
    f, bx, by = first_tracked_ball(bx_norm, by_norm)
    px, py = project(bx, by, H[f])
    err = float(np.hypot(px - PITCH_CENTRE_M[0], py - PITCH_CENTRE_M[1]))
    centre_px = project(0.5, 0.5, H[KICKOFF_FRAME_1ST], scale=1.0)
    corners = [tuple(int(round(c)) for c in project(cx, cy, H[f], scale=1.0))
               for cx, cy in ((0, 0), (1, 0), (0, 1), (1, 1))]
    return {
        "frame": int(f),
        "ball_norm": (float(bx), float(by)),
        "ball_metres_from_normalised": (float(bx) * 105.3, float(by) * 68.0),
        "pitch_corners_unscaled": corners,
        "projected_scaled": (float(px), float(py)),
        "distance_to_centre_m": err,
        "decimetre_hypothesis_holds": err < 3.0,
        "centre_unscaled": tuple(float(c) for c in centre_px),
        "centre_unscaled_inside_1920x1080": bool(0 <= centre_px[0] <= 1920 and 0 <= centre_px[1] <= 1080),
    }
