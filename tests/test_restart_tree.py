"""Fixture tests for the throw-in / corner restart tree: no gated data needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import restart_tree as RT  # noqa: E402

H, A, D = RT.H, RT.A, RT.D
CAP = 250


def scene(n=800, ctrl_team=H, ctrl_until=300, stop=(300, 500), blind_until=560, y_out=False):
    """`ctrl_team` has the ball until frame ctrl_until; D over `stop`; the ball is untracked from
    ctrl_until to blind_until (so the stoppage and the restart are both blind)."""
    sH = np.full(n, float(CAP))
    sA = np.full(n, float(CAP))
    key = sH if ctrl_team == H else sA
    key[:ctrl_until] = 0.0
    key[ctrl_until:] = np.minimum(np.arange(1, n - ctrl_until + 1), CAP)
    inv = np.zeros(n)
    inv[ctrl_until:blind_until] = 1
    by = np.full(n, 30.0)
    if y_out:
        by[ctrl_until:stop[1]] = 69.5
        inv[ctrl_until:stop[1]] = 0           # ball tracked outside the touchline during the stop
    f = {"since_ctrl_H": sH, "since_ctrl_A": sA, "ball_invisible": inv,
         "ball_x": np.full(n, 50.0), "ball_y": by}
    path = np.full(n, int(ctrl_team), np.int8)
    path[stop[0]:stop[1]] = D
    return f, path


def tag(kind, team=None, a=300):
    return [{"code": "set pieces", "type": kind, "a": a, "team": team}]


def test_throw_in_gives_the_restart_to_the_team_that_did_not_have_the_ball():
    f, path = scene(ctrl_team=H)
    out = RT.apply(path, f, tag("throw-in"))
    assert (out[500:560] == A).all()                     # blind stretch relabelled
    assert (out[:300] == H).all() and (out[300:500] == D).all()


def test_ball_visible_at_the_restart_is_left_to_the_decoder():
    f, path = scene(ctrl_team=H, blind_until=500)         # ball reappears exactly at the restart
    assert (RT.apply(path, f, tag("throw-in")) == path).all()


def test_corner_follows_the_tag_not_the_literal_idea():
    f, path = scene(ctrl_team=H)
    out = RT.apply(path, f, tag("corner kick", team=H))   # attackers H keep it: tag team = H
    assert (out[500:560] == H).all()
    literal = RT.apply(path, f, tag("corner kick", team=H), literal_corner=True)
    assert (literal[500:560] == A).all()                  # the ablation does the opposite


def test_other_set_pieces_are_never_touched():
    f, path = scene(ctrl_team=H, y_out=True)
    for kind in ("goal kick", "free kick", "kick off (after goal)"):
        assert (RT.apply(path, f, tag(kind)) == path).all()


def test_untagged_stoppage_only_enters_the_tree_if_the_ball_left_over_a_touchline():
    f, path = scene(ctrl_team=H, y_out=False)
    assert (RT.apply(path, f, []) == path).all()
    f, path = scene(ctrl_team=H, y_out=True)
    out = RT.apply(path, f, [])
    assert (out[500:560] == A).all()


def test_unclear_last_controller_abstains():
    f, path = scene(ctrl_team=H)
    f["since_ctrl_A"][:300] = 0.0                         # both teams on the ball at the end
    assert (RT.apply(path, f, tag("throw-in")) == path).all()
    f2, path2 = scene(ctrl_team=H, ctrl_until=10, stop=(600, 700), blind_until=20)
    assert RT.last_controller(f2, 600) is None            # last control more than 10 s ago


def test_disable_switches_a_leaf_off_and_logs_what_happened():
    f, path = scene(ctrl_team=A)
    assert (RT.apply(path, f, tag("throw-in"), disable=("throw_in_tag",)) == path).all()
    log = []
    out = RT.apply(path, f, tag("throw-in"), log=log)
    assert log[0]["leaf"] == "throw_in_tag" and log[0]["team"] == H
    assert log[0]["blind"] == 60 and log[0]["changed"] == 60
    assert (out[500:560] == H).all()
