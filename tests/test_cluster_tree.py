"""Fixture tests for the crowd rule: no gated data needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import cluster_tree as CT  # noqa: E402

H, A, D = CT.H, CT.A, CT.D
TEAM = np.array([0] * 23 + [1] * 23, np.int8)


def tracking(n, n_h, n_a, spread=1.0, far_h=0):
    """n_h home and n_a away players packed around (50, 30); the rest spread over the pitch."""
    px = np.full((n, 46), np.nan, np.float32)
    py = np.full((n, 46), np.nan, np.float32)
    rng = np.random.RandomState(0)
    home_idx, away_idx = np.arange(23), np.arange(23, 46)
    for k, i in enumerate(home_idx):
        px[:, i] = 50 + rng.uniform(-spread, spread) if k < n_h else 10 + 4 * k
        py[:, i] = 30 + rng.uniform(-spread, spread) if k < n_h else 5.0
    for k, i in enumerate(away_idx):
        px[:, i] = 50 + rng.uniform(-spread, spread) if k < n_a else 100 - 4 * k
        py[:, i] = 30 + rng.uniform(-spread, spread) if k < n_a else 60.0
    return px, py


def feats(n, invisible=True, seen_at=(50.0, 30.0), age=10):
    return {"ball_invisible": np.full(n, 1.0 if invisible else 0.0),
            "n_tracked": np.full(n, 22.0), "tracking_gap": np.zeros(n),
            "frames_since_ball": np.full(n, float(age)),
            "ball_x": np.full(n, -1.0 if seen_at is None else seen_at[0]),
            "ball_y": np.full(n, -1.0 if seen_at is None else seen_at[1])}


def run(n_h, n_a, path_val=A, **kw):
    n = 200
    px, py = tracking(n, n_h, n_a)
    crowd = CT.build_crowd(px, py, TEAM)
    path = np.full(n, path_val, np.int8)
    return CT.apply(path, feats(n, **kw), crowd), crowd


def test_pack_counts_per_team():
    _, crowd = run(5, 2)
    assert crowd["nH"][0] == 5 and crowd["nA"][0] == 2


def test_team_with_more_players_in_a_crowd_gets_the_ball():
    out, _ = run(5, 2, path_val=A)
    assert (out == H).all()
    out, _ = run(2, 5, path_val=H)
    assert (out == A).all()


def test_close_numbers_small_packs_and_visible_ball_are_left_alone():
    assert (run(4, 3)[0] == A).all()                        # margin 1: no clear majority
    assert (run(2, 1)[0] == A).all()                        # pack of 3: not a crowd
    assert (run(5, 2, invisible=False)[0] == A).all()       # ball tracked: not this rule's job


def test_dead_ball_is_never_overridden():
    assert (run(5, 2, path_val=D)[0] == D).all()


def test_pack_far_from_where_the_ball_was_last_seen_is_ignored():
    assert (run(5, 2, seen_at=(10.0, 30.0))[0] == A).all()


def test_ball_lost_for_long_needs_a_bigger_pack():
    assert (run(5, 2, seen_at=None, age=250)[0] == A).all()          # 7 players < 8
    assert (run(7, 3, seen_at=None, age=250)[0] == H).all()          # 10 players


def test_disable_switches_a_leaf_off():
    n = 200
    px, py = tracking(n, 5, 2)
    crowd = CT.build_crowd(px, py, TEAM)
    path = np.full(n, A, np.int8)
    out = CT.apply(path, feats(n), crowd, disable=("crowd_near_ball",))
    assert (out == A).all()
