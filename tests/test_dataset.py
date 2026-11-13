"""Dataset builder fixture test + no-leakage test (brief Step 1 DoD, Hard Rule 6).
Synthetic data only: runs on a clean clone without the gated bundle."""
import os
import sys

import numpy as np
import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from mtah_io import atd as A, homography as Hm  # noqa: E402
from pipeline import features as FE  # noqa: E402

N = 600
HALVES = [(1, 50, 280), (2, 330, 560)]


def synthetic(seed=0):
    rng = np.random.default_rng(seed)
    w = {
        "px": rng.uniform(0, 105.3, (N, 46)).astype(np.float32),
        "py": rng.uniform(0, 68, (N, 46)).astype(np.float32),
        "ps": rng.uniform(0, 8, (N, 46)).astype(np.float32),
        "bx": rng.uniform(0, 105.3, N).astype(np.float32),
        "by": rng.uniform(0, 68, N).astype(np.float32),
        "bs": rng.uniform(0, 20, N).astype(np.float32),
        "team": np.r_[np.zeros(24), np.ones(22)].astype(np.int8),
    }
    for k in ("px", "py", "ps"):
        w[k][rng.random((N, 46)) < 0.5] = np.nan
    miss = rng.random(N) < 0.4
    w["bx"][miss] = np.nan
    w["by"][miss] = np.nan
    tags = [
        {"code": "set pieces", "a": 60, "b": 180, "team": None, "type": "kick off (start)"},
        {"code": "build up", "a": 120, "b": 300, "team": 0, "type": ""},
        {"code": "defending in middle third", "a": 200, "b": 260, "team": 0, "type": ""},
        {"code": "set pieces", "a": 400, "b": 500, "team": 1, "type": "corner kick"},
        {"code": "set pieces", "a": 345, "b": 420, "team": None, "type": "kick off (start)"},
    ]
    return w, tags


def truncate(w, tags, t):
    """Everything after frame t removed (NaN), and tags that have not started by t dropped."""
    v = {k: (a.copy() if k == "team" else a.copy()) for k, a in w.items()}
    for k in ("px", "py", "ps", "bx", "by", "bs"):
        v[k][t + 1:] = np.nan
    return v, [g for g in tags if g["a"] <= t]


def test_raw_line_roundtrip_to_long_format():
    slots = ";".join(["0.5,0.25,3.0"] + ["NaN,NaN,NaN"] * 45)
    text = "1:{}:0.5,0.5,1.0\n2:{}:NaN,NaN,NaN\n".format(slots, slots)
    w = A._wide.parse_lines(text, n_frames=2, pitch=(105.3, 68.0))
    long_df = A.wide_to_long(w)
    assert len(long_df) == 2 * 47                       # one row per (frame, player_id)
    ball0 = long_df[(long_df.frame == 0) & long_df.is_ball].iloc[0]
    assert ball0.x == pytest.approx(52.65) and ball0.y == pytest.approx(34.0)
    p1 = long_df[(long_df.frame == 0) & (long_df.player_id == 1)].iloc[0]
    assert p1.x == pytest.approx(52.65) and p1.y == pytest.approx(17.0) and p1.speed == 3.0
    assert np.isnan(long_df[(long_df.frame == 1) & long_df.is_ball].iloc[0].x)


def test_frame_features_shape_and_columns():
    w, tags = synthetic()
    df = FE.frame_features(w, tags, HALVES)
    assert len(df) == N
    assert list(df.columns) == ["frame"] + FE.ALL_FEATURES
    assert np.isfinite(df[FE.ALL_FEATURES].to_numpy(np.float64)).all()
    assert "gt_public" not in FE.ALL_FEATURES


def test_project_identity_scale():
    H = np.eye(3)
    x, y = Hm.project(0.5, 0.25, H, scale=1.0)
    assert (x, y) == (0.5, 0.25)


@pytest.mark.parametrize("t", [0, 1, 55, 61, 119, 121, 199, 201, 299, 344, 346, 399, 401, 599])
def test_no_future_leakage(t):
    """Every feature at frame t is identical whether or not frames > t exist."""
    w, tags = synthetic(seed=t)
    full = FE.frame_features(w, tags, HALVES).iloc[t]
    v, past_tags = truncate(w, tags, t)
    cut = FE.frame_features(v, past_tags, HALVES).iloc[t]
    bad = [c for c in FE.ALL_FEATURES if not np.isclose(full[c], cut[c], equal_nan=True)]
    assert not bad, "features use the future at t={}: {}".format(t, bad)
