"""Referee-whistle evidence (pipeline/whistle.py). Synthetic data only, no audio file needed."""
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import whistle as WH  # noqa: E402


def scores(n=1500):
    return np.full(n, 6.0)            # background whistle score (the match's median is ~6 dB)


def test_short_blips_are_not_whistles():
    w = scores(); w[100] = 30                       # one frame only
    assert len(WH.onsets(w)) == 0


def test_a_whistle_needs_two_frames_and_bridges_a_one_frame_dip():
    w = scores(); w[100:102] = 30
    assert WH.onsets(w).tolist() == [100]
    w = scores(); w[200] = 30; w[202] = 30          # dip at 201 is bridged -> one whistle
    assert WH.onsets(w).tolist() == [200]


def test_stop_whistle_needs_players_to_slow_down():
    speed = np.full(1500, 2.0)
    speed[300:] = 1.0                               # players slow after frame 300
    assert WH.is_stop(300, speed)
    assert not WH.is_stop(300, np.full(1500, 2.0))  # nobody slows: not a stop
    assert not WH.is_stop(300, np.full(1500, 0.5))  # already still: a restart whistle


def test_stoppage_ends_at_the_restart_kick_or_the_cap():
    ball = np.zeros(2000)
    ball[700:705] = 15.0                            # restart kick at frame 700
    assert WH.stoppage_end(300, ball) == 700
    assert WH.stoppage_end(300, np.zeros(2000)) == 300 + WH.MAX_S * WH.FPS
    ball = np.zeros(2000); ball[310:315] = 15.0     # inside the first 2 s: ignored
    assert WH.stoppage_end(300, ball) == 300 + WH.MAX_S * WH.FPS


def test_evidence_marks_whistle_to_kick_and_ignores_restart_whistles():
    n = 2000
    w = scores(n); w[300:305] = 30; w[1200:1205] = 30
    speed = np.full(n, 2.0); speed[300:900] = 0.8   # slowdown only after the first whistle
    speed[1100:] = 0.5                              # second whistle: players already still
    ball = np.zeros(n); ball[800:805] = 15.0
    mask = WH.evidence(w, {"speed_all": speed, "ball_speed": ball})
    assert mask[300:800].all() and not mask[:300].any() and not mask[800:].any()
    assert (WH.bonus(w, {"speed_all": speed, "ball_speed": ball})[300:800] == WH.BONUS).all()


def test_foulteam_nudges_the_team_that_had_the_ball_before_the_whistle():
    n = 2000
    w = scores(n); w[300:305] = 30
    speed = np.full(n, 2.0); speed[300:900] = 0.8
    ball = np.zeros(n); ball[800:805] = 15.0
    team = np.zeros((n, 2)); team[:300, 1] = 1.0           # away (column 1) had the ball before
    out = WH.restart_bonus(w, {"speed_all": speed, "ball_speed": ball}, team)
    span = slice(800, 800 + WH.RESTART_S * WH.FPS)
    assert (out[span, 1] == WH.RESTART_BONUS).all() and not out[span, 0].any()
    assert not out[:800].any() and not out[800 + WH.RESTART_S * WH.FPS:].any()


def test_sound_features_only_use_the_past():
    from pipeline import audio as AU
    n = 1000
    m = {"whistle_db": np.full(n, 6.0, np.float32), "rms_db": np.full(n, -30.0, np.float32)}
    m["whistle_db"][500:505] = 30.0
    f = AU.causal_features(m)
    for k, v in f.items():
        assert v.shape == (n,), k
    # nothing changes before the whistle (+2 frames, the analysis window's look-ahead)
    assert (f["wh_max_125"][:502] == 6.0).all() and f["wh_max_125"][502] == 30.0
    assert f["since_whistle"][503] == 0 and f["since_whistle"][504] == 1
