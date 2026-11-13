"""Algorithmic training labels (no human labels, no ground truth). Hard Rule 1 / RULES.md.

Teacher: the hand-specified offline HMM in possession/model.py, which decodes the whole match
(past AND future) -> H / A / D. Labels may use the future; only FEATURES must be causal.
X ("contested") is overlaid where the ball is tracked, both teams have a player within reach,
neither team has had sole control in the previous/next 0.5 s, and the ball is not in fast flight.
"""
import numpy as np

from possession import atd as atd_mod, model as M
from possession import features as PF

CLASSES = np.array(["H", "A", "D", "X"])
REACH = 1.5


def contested(feats):
    vis = feats["ball_vis"]
    dh, da = feats["d_home"], feats["d_away"]
    both = vis & (dh < REACH) & (da < REACH)
    sole_h = vis & (dh < REACH) & (da >= REACH)
    sole_a = vis & (da < REACH) & (dh >= REACH)
    k = 12
    kern = np.ones(2 * k + 1)
    near_sole = (np.convolve(sole_h, kern, "same") > 0) ^ (np.convolve(sole_a, kern, "same") > 0)
    slow = np.nan_to_num(feats["ball_v"], nan=0) < 8
    return both & ~near_sole & slow


def build():
    atd = atd_mod.load()
    teacher = M.predict(atd)                     # H/A/D in halves, X outside
    feats = PF.build(atd)
    y = teacher.copy()
    graded = teacher != "X"
    y[graded & contested(feats)] = "X"
    return y, graded
