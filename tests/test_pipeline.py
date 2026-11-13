"""Fixture tests: run without the gated data."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from possession import atd, model  # noqa: E402


def _line(frame_id, ball=(0.5, 0.5, 1.0), first=(0.5, 0.5, 2.0)):
    slots = ["{},{},{}".format(*first)] + ["NaN,NaN,NaN"] * (atd.N_SLOTS - 1)
    b = "NaN,NaN,NaN" if ball is None else "{},{},{}".format(*ball)
    return "{}:{}:{}".format(frame_id, ";".join(slots), b)


def test_parse_aligns_frame_id_to_video_frame_and_scales_to_metres():
    text = "\n".join([_line(1), _line(3, ball=None)]) + "\n"
    out = atd.parse_lines(text, n_frames=4, pitch=(100.0, 50.0))
    assert out["bx"][0] == 50.0 and out["by"][0] == 25.0     # frame_id 1 -> video frame 0
    assert np.isnan(out["bx"][1])                            # frame never in file stays NaN
    assert np.isnan(out["bx"][2])                            # ball NaN preserved
    assert out["px"][2, 0] == 50.0 and np.isnan(out["px"][2, 1])


def test_parse_rejects_out_of_range_frames():
    try:
        atd.parse_lines(_line(9) + "\n", n_frames=4, pitch=(100.0, 50.0))
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_viterbi_holds_state_through_weak_contrary_evidence():
    em = np.zeros((100, 3))
    em[:60, model.H] = 1.0
    em[30:33, model.H] = 0.0
    em[30:33, model.A] = 1.0       # isolated 3-frame blip inside an H spell: should not flip
    em[60:, model.A] = 1.0         # sustained: should flip
    logT = model.transition_matrix(dict(model.PARAMS, cost_HA=5.0))
    path = model.viterbi(em, logT)
    assert (path[:60] == model.H).all()
    assert (path[60:] == model.A).all()
    assert (path[65:] == model.A).all()
