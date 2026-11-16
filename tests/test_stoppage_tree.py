"""Fixture tests for the stoppage clean-up: no gated data needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import dead_tree as DT  # noqa: E402
from pipeline import stoppage_tree as ST  # noqa: E402

H, A, D = ST.H, ST.A, ST.D


def path_of(n, *spans, base=H):
    p = np.full(n, base, np.int8)
    for s, e in spans:
        p[s:e] = D
    return p


def test_merge_joins_stoppages_with_under_three_seconds_of_play_between():
    p, n = ST.merge_close(path_of(600, (100, 200), (250, 300), (500, 560)))
    assert n == 1 and (p[100:300] == D).all() and (p[300:500] == H).all()


def test_short_untagged_stoppage_is_removed_but_a_tagged_one_stays():
    p = path_of(600, (100, 130), (400, 430))
    tags = [{"code": "set pieces", "type": "throw-in", "a": 395, "b": 440, "team": None}]
    out, n = ST.drop_short(p, tags)
    assert n == 1 and (out[100:130] == H).all() and (out[400:430] == D).all()


def test_tag_midpoint_moves_the_end_of_a_tagged_stoppage():
    n = 600
    f = {"speed_all": np.full(n, 3.0), "n_tracked": np.full(n, 22.0),
         "ball_invisible": np.ones(n), "ball_oob": np.zeros(n)}
    p = path_of(n, (200, 300))
    tags = [{"code": "set pieces", "type": "corner kick", "a": 200, "b": 500, "team": 0}]
    out, moved = ST.refine_edges(p, f, tags)             # midpoint = 350
    assert moved >= 1 and (out[200:350] == D).all() and out[350] == H


def test_stoppage_ends_when_players_start_running_again():
    n = 800
    speed = np.full(n, 3.0)
    speed[300:420] = 0.2                                  # players stand for 120 frames
    f = {"speed_all": speed, "n_tracked": np.full(n, 22.0),
         "ball_invisible": np.ones(n), "ball_oob": np.zeros(n)}
    p = path_of(n, (300, 380))                            # decoded end is too early
    out, _ = ST.refine_edges(p, f, [])
    assert (out[300:410] == D).all() and out[500] == H


def test_edges_never_cross_a_neighbouring_stoppage_and_first_run_is_untouched():
    n = 600
    f = {"speed_all": np.full(n, 0.1), "n_tracked": np.full(n, 22.0),
         "ball_invisible": np.ones(n), "ball_oob": np.zeros(n)}
    p = path_of(n, (0, 100), (150, 250))
    out, _ = ST.refine_edges(p, f, [])
    assert (out[:100] == D).all()                                  # first run untouched
    starts, ends = DT.runs(out == D)
    assert len(starts) == 2 and ends[0] <= starts[1]               # still two separate runs


def test_clean_runs_all_steps_and_disable_works():
    p = path_of(600, (100, 200), (250, 300))
    f = {"speed_all": np.full(600, 3.0), "n_tracked": np.full(600, 22.0),
         "ball_invisible": np.ones(600), "ball_oob": np.zeros(600)}
    merged = ST.clean(p, f, [], disable=("edges", "midpoint"))
    assert (merged[100:300] == D).all()
    kept = ST.clean(p, f, [], disable=("edges", "midpoint", "merge"))
    assert (kept[200:250] == H).all()


def test_reworked_players_still_catches_a_walking_stoppage_with_a_carried_ball():
    n = 400
    f = {"tracking_gap": np.zeros(n), "n_tracked": np.full(n, 22.0), "ball_invisible": np.zeros(n),
         "ball_oob": np.zeros(n), "speed_all": np.full(n, 3.0),
         "speed_all_50": np.r_[np.full(100, 3.0), np.full(200, 1.1), np.full(100, 3.0)],
         "ball_speed_max_12": np.r_[np.full(100, 12.0), np.full(200, 2.0), np.full(100, 12.0)],
         "dead_ball_interval": np.zeros(n), "sp_age": np.full(n, 250.0), "sp_type": np.zeros(n)}
    pre = np.zeros(n, bool)
    assert (DT.classify(f, pre) == 0).all()                       # old leaf needs a still ball
    out = DT.classify(f, pre, rework=True)
    assert (out[100:300] == DT.LEAVES.index("players_still") + 1).all()


def test_soft_bonus_follows_the_leaf_order():
    b = ST.soft_bonus(np.array([0, 1, 2, 5]))
    assert b[0] == 0 and b[1] > b[2] > b[3] > 0
