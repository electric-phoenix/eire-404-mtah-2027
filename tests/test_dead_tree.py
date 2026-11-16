"""Fixture tests for the dead-ball decision tree: no gated data needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import dead_tree as DT  # noqa: E402


def frames(n, **over):
    """A live-play half: players running, ball visible and fast, no tags, nothing flagged."""
    f = {
        "tracking_gap": np.zeros(n), "n_tracked": np.full(n, 22.0),
        "ball_invisible": np.zeros(n), "ball_oob": np.zeros(n),
        "speed_all": np.full(n, 3.0), "speed_all_50": np.full(n, 3.0),
        "ball_speed_max_12": np.full(n, 12.0),
        "dead_ball_interval": np.zeros(n), "sp_age": np.full(n, 250.0), "sp_type": np.zeros(n),
    }
    f.update({k: np.asarray(v, float) for k, v in over.items()})
    return f


def leaf(f, pre=None, **kw):
    n = len(f["speed_all"])
    return DT.classify(f, np.zeros(n, bool) if pre is None else pre, **kw)


def test_live_play_has_no_dead_ball_evidence():
    assert (leaf(frames(500)) == 0).all()


def test_long_missing_ball_with_running_players_stays_alive():
    # the scramble in the box: ball gone for 8 s, players still running -> NOT dead
    inv = np.zeros(500)
    inv[100:300] = 1
    assert (leaf(frames(500, ball_invisible=inv)) == 0).all()


def test_long_missing_ball_with_slow_players_is_dead_for_the_whole_run():
    inv = np.zeros(500)
    inv[100:300] = 1
    slow = np.full(500, 0.5)
    out = leaf(frames(500, ball_invisible=inv, speed_all=slow))
    assert (out[100:300] == DT.LEAVES.index("ball_missing_slow") + 1).all()
    assert (out[:100] == 0).all() and (out[300:] == 0).all()


def test_missing_ball_shorter_than_threshold_is_not_dead():
    inv = np.zeros(500)
    inv[100:100 + DT.MISSING_MIN - 1] = 1
    assert (leaf(frames(500, ball_invisible=inv, speed_all=np.full(500, 0.5))) == 0).all()


def test_tracking_gap_inside_a_missing_run_blocks_the_leaf():
    inv = np.zeros(500)
    inv[100:300] = 1
    gap = np.zeros(500)
    gap[200] = 1                       # players vanish for a frame: "not moving" is not evidence
    out = leaf(frames(500, ball_invisible=inv, speed_all=np.full(500, 0.0), tracking_gap=gap))
    assert (out == 0).all()


def test_first_matching_leaf_wins_and_pre_kickoff_ranks_first():
    n = 300
    pre = np.zeros(n, bool)
    pre[:100] = True
    f = frames(n, dead_ball_interval=np.r_[np.ones(150), np.zeros(150)],
               speed_all_50=np.full(n, 0.5))
    out = leaf(f, pre)
    assert (out[:100] == 1).all()                                   # pre_kickoff, not leaf 2
    assert (out[100:150] == DT.LEAVES.index("setpiece_stopped") + 1).all()


def test_ball_off_the_pitch_needs_two_seconds():
    oob = np.zeros(400)
    oob[50:50 + DT.OOB_MIN - 1] = 1                                  # 1 frame short
    oob[200:200 + DT.OOB_MIN] = 1
    out = leaf(frames(400, ball_oob=oob))
    assert (out[50:100] == 0).all()
    assert (out[200:200 + DT.OOB_MIN] == DT.LEAVES.index("ball_off_pitch") + 1).all()


def test_players_still_leaf_catches_a_stoppage_with_no_tag_and_a_visible_ball():
    n = 400
    still = np.full(n, 3.0)
    still[100:300] = 0.2
    ball = np.full(n, 12.0)
    ball[100:300] = 0.1
    out = leaf(frames(n, speed_all_50=still, ball_speed_max_12=ball))
    assert (out[100:300] == DT.LEAVES.index("players_still") + 1).all()
    assert (out[:100] == 0).all() and (out[300:] == 0).all()


def test_disable_switches_one_leaf_off():
    inv = np.zeros(500)
    inv[100:300] = 1
    f = frames(500, ball_invisible=inv, speed_all=np.full(500, 0.5))
    assert (leaf(f, disable=("ball_missing_slow",)) == 0).all()


def test_pre_kickoff_mask_uses_the_kickoff_tag_and_claims_nothing_without_one():
    tags = [{"code": "set pieces", "type": "kick off (start)", "a": 40, "team": 0}]
    m = DT.pre_kickoff_mask(100, tags)
    assert m[:40].all() and not m[40:].any()
    assert not DT.pre_kickoff_mask(100, []).any()
