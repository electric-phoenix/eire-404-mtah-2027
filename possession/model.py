"""Hand-specified 3-state HMM (H, A, D) decoded with Viterbi. No learned parameters.

Emissions are log-likelihood scores built from football reasoning:
  team    a player within reach of the ball, nearer than any opponent, is evidence his team has it;
          a ball in flight (nobody near) or not tracked is no evidence -> the state holds.
          Smart Tagging phases add a weaker, slower vote.
  dead    ball off the pitch, players walking, ball still, set-piece windows -> D.
          fast ball / running players -> alive.
Transitions carry fixed switching costs, which is what turns flickering frame evidence into
whole possessions -- the unit spellF1 scores.
"""

import numpy as np

from . import features as F
from . import tagging as T

H, A, D = 0, 1, 2
LETTERS = np.array(["H", "A", "D"])

PARAMS = {
    # team evidence
    "reach_m": 2.0,          # a player this close to the ball can be in control of it
    "w_ctrl": 0.6,           # per-frame log-odds for the controlling team
    "ctrl_max_ball_v": 15.0, # m/s; faster than this the ball is passing a player, not controlled
    "w_tag": 0.05,           # per-frame log-odds per Smart Tagging vote
    # dead-ball evidence (log-odds D vs alive)
    "w_oob": 1.5,
    "slow_players_ms": 1.4,  # mean player speed below which play looks stopped
    "w_slow": 0.4,           # per m/s below the threshold
    "w_fast": 0.2,           # per m/s above it (alive evidence)
    "w_invisible_dead": 0.05,
    "w_setpiece_dead": 0.3,
    "alive_bias": 0.25,      # alive is the majority of graded time
    # transition costs (nats)
    "cost_HA": 5.0,
    "cost_to_D": 10.0,
    "cost_from_D": 8.0,
}


def emissions(atd, feats, evs, p=PARAMS):
    n = len(atd["bx"])
    e = np.zeros((n, 3), np.float64)

    # --- team: controlled-ball evidence
    dh, da = feats["d_home"], feats["d_away"]
    near = np.minimum(dh, da)
    bv = np.nan_to_num(feats["ball_v"], nan=0.0)
    ctrl = feats["ball_vis"] & ~feats["oob"] & (near < p["reach_m"]) & (bv < p["ctrl_max_ball_v"])
    margin = np.clip(np.where(np.isfinite(dh) & np.isfinite(da), da - dh, 0.0), -3, 3) / 3.0
    team = np.where(ctrl, p["w_ctrl"] * np.sign(margin) * np.sqrt(np.abs(margin)), 0.0)

    # --- team: Smart Tagging phase votes
    team += p["w_tag"] * np.clip(T.possession_votes(evs, n), -2, 2)
    e[:, H] += team / 2
    e[:, A] -= team / 2

    # --- dead vs alive
    ps = np.nan_to_num(feats["p_speed"], nan=0.0)
    dead = np.full(n, -p["alive_bias"])
    dead += p["w_oob"] * feats["oob"]
    dead += np.where(ps < p["slow_players_ms"], p["w_slow"] * (p["slow_players_ms"] - ps),
                     -p["w_fast"] * (ps - p["slow_players_ms"]))
    dead += p["w_invisible_dead"] * ~feats["ball_vis"]
    for sp in T.set_pieces(evs):
        # The tag window straddles the restart; its first half is the stoppage.
        mid = (sp["a"] + sp["b"]) // 2
        dead[max(sp["a"], 0):max(mid, 0)] += p["w_setpiece_dead"]
        if sp["team"] is not None:  # restarting side gets the ball
            e[mid:sp["b"], sp["team"]] += 0.3
    e[:, D] += dead
    return e


def transition_matrix(p=PARAMS):
    c = np.zeros((3, 3))
    c[H, A] = c[A, H] = p["cost_HA"]
    c[H, D] = c[A, D] = p["cost_to_D"]
    c[D, H] = c[D, A] = p["cost_from_D"]
    return -c  # log-scores


def viterbi(em, logT, start=None):
    n, k = em.shape
    score = em[0] + (np.zeros(k) if start is None else start)
    back = np.zeros((n, k), np.int8)
    for t in range(1, n):
        cand = score[:, None] + logT        # from i to j
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + em[t]
    path = np.empty(n, np.int8)
    path[-1] = score.argmax()
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def predict(atd, p=PARAMS):
    """Letters for every video frame: H/A/D inside the graded halves, X outside."""
    feats = F.build(atd)
    evs = T.events()
    em = emissions(atd, feats, evs, p)
    logT = transition_matrix(p)
    n = len(atd["bx"])
    labels = np.full(n, "X", dtype="<U1")
    for _, a, b in F.half_video_blocks(F.GAME):
        # every half starts from a stoppage (before kickoff)
        path = viterbi(em[a:b + 1], logT, start=np.array([-5.0, -5.0, 0.0]))
        labels[a:b + 1] = LETTERS[path]
    return labels
