"""Structured decoder (brief Step 3) and rule layer (brief [5]).

Viterbi over 4 states H, A, D, X on the classifier's frame probabilities.
  transition(t) = row-normalise( base_priors  x  smart_tagging_modifier(event active at t) )
  emission(t)   = log P(state | x_t) - log prior(state)     (posterior -> scaled likelihood)
Min spell length: possession_metrics.py enforces NONE (spells are matched by temporal IoU only),
so MIN_SPELL_FRAMES defaults to 1; the mechanism is kept and ablated.

Rule layer, applied after decoding, in this order:
  1. dead-ball override     Smart Tagging dead interval (pre-kickoff; set-piece onset with the ball
                            off the pitch or players stopped) -> D
  2. pass-flight logic      an H<->A flip during fast ball flight is moved to the first frame the
                            new team has a player in reach of the ball; if none within 2 s, undone
  3. ball-invisible fallback  no H<->A flip while the ball is unseen: the last label (decided with
                            the last-known ball) holds until the ball reappears
  4. contested -> X         both teams in reach of the ball and neither in control -> X
"""
import numpy as np

STATES = np.array(["H", "A", "D", "X"])
H, A, D, X = range(4)
FPS = 25
MIN_SPELL_FRAMES = 1

# Expected dwell (seconds) per state and where a state goes when it ends.
DWELL_S = {H: 10.0, A: 10.0, D: 12.0, X: 1.0}
EXIT = {
    H: {A: 0.55, D: 0.35, X: 0.10},
    A: {H: 0.55, D: 0.35, X: 0.10},
    D: {H: 0.50, A: 0.50, X: 0.00},
    X: {H: 0.45, A: 0.45, D: 0.10},
}


def base_transition():
    T = np.zeros((4, 4))
    for i in range(4):
        stay = 1.0 - 1.0 / (DWELL_S[i] * FPS)
        T[i, i] = stay
        for j, p in EXIT[i].items():
            T[i, j] = (1 - stay) * p
    return T


def smart_tagging_modifier(event_type, team=None):
    """4x4 multiplicative modifier on transition probabilities while a tag event is active.

    event_type: "setpiece_onset" (first seconds of a SET PIECES tag: ball going dead),
                "setpiece_restart" (later part: restart by `team`, or unknown for throw-ins),
                "possession" (an in/out-of-possession phase starting for `team`), or None.
    team: 0 = H, 1 = A, None = unknown.
    """
    M = np.ones((4, 4))
    if event_type == "setpiece_onset":
        M[H, D] = M[A, D] = M[X, D] = 20.0
        M[D, H] = M[D, A] = 0.2
    elif event_type == "setpiece_restart":
        if team is None:
            M[D, H] = M[D, A] = 10.0
        else:
            M[D, team] = 20.0
            M[D, 1 - team] = 2.0
    elif event_type == "possession" and team is not None:
        M[1 - team, team] = 5.0
        M[D, team] = 3.0
        M[X, team] = 3.0
    return M


def modifier_schedule(tags, n, onset=75, restart=(50, 175), phase=50):
    """Per-frame modifier index + list of matrices (memory-light: most frames use identity)."""
    mats = [np.ones((4, 4))]
    idx = np.zeros(n, np.int32)
    in_poss = {"build up", "progression", "creating chances", "long ball", "attacking transition"}
    out_poss = {"defensive transition", "defending in defensive third", "defending in middle third",
                "defending in attacking third"}
    for t in sorted(tags, key=lambda t: t["a"]):
        a = max(t["a"], 0)
        if t["code"] == "set pieces":
            mats.append(smart_tagging_modifier("setpiece_onset"))
            idx[a:min(a + onset, n)] = len(mats) - 1
            mats.append(smart_tagging_modifier("setpiece_restart", t["team"]))
            idx[min(a + restart[0], n):min(a + restart[1], n)] = len(mats) - 1
        elif t["team"] is not None and (t["code"] in in_poss or t["code"] in out_poss):
            side = t["team"] if t["code"] in in_poss else 1 - t["team"]
            mats.append(smart_tagging_modifier("possession", side))
            seg = idx[a:min(a + phase, n)]
            seg[seg == 0] = len(mats) - 1          # set-piece modifiers take precedence
    return idx, mats


def viterbi(log_em, tags=None, use_modifiers=True, start=None):
    n = len(log_em)
    base = base_transition()
    if use_modifiers and tags is not None:
        idx, mats = modifier_schedule(tags, n)
    else:
        idx, mats = np.zeros(n, np.int32), [np.ones((4, 4))]
    logTs = []
    for M in mats:
        T = base * M
        T /= T.sum(axis=1, keepdims=True)
        logTs.append(np.log(np.maximum(T, 1e-300)))
    score = log_em[0] + (np.log(np.full(4, 0.25)) if start is None else start)
    back = np.zeros((n, 4), np.int8)
    for t in range(1, n):
        cand = score[:, None] + logTs[idx[t]]
        back[t] = cand.argmax(axis=0)
        score = cand.max(axis=0) + log_em[t]
    path = np.empty(n, np.int8)
    path[-1] = int(score.argmax())
    for t in range(n - 1, 0, -1):
        path[t - 1] = back[t, path[t]]
    return path


def enforce_min_spell(path, k=MIN_SPELL_FRAMES):
    if k <= 1:
        return path
    p = path.copy()
    change = np.flatnonzero(np.r_[True, p[1:] != p[:-1]])
    ends = np.r_[change[1:], len(p)]
    for s, e in zip(change, ends):
        if e - s < k and s > 0:
            p[s:e] = p[s - 1]
    return p


# ---------------------------------------------------------------- rule layer
def rule_dead_ball(path, f):
    p = path.copy()
    stopped = (f["speed_all_50"] < 1.6) | (f["ball_oob"] > 0)
    p[(f["dead_ball_interval"] > 0) & stopped] = D
    return p


def rule_dead_tree(path, f, disable=()):
    """Dead-ball decision tree (pipeline/dead_tree.py) replaces the single dead-ball override.
    f must carry "pre_kickoff", the half's before-kickoff mask."""
    from . import dead_tree
    p = path.copy()
    p[dead_tree.classify(f, f["pre_kickoff"], disable) > 0] = D
    return p


def rule_pass_flight(path, f, reach=1.5, horizon=50):
    p = path.copy()
    fast = f["ball_speed_max_12"] > 12
    dH, dA = f["d_H1"], f["d_A1"]
    t_list = np.flatnonzero((p[1:] != p[:-1]) & np.isin(p[1:], [H, A]) & np.isin(p[:-1], [H, A])) + 1
    for t in t_list:
        if not fast[t] or p[t] == p[t - 1]:
            continue
        new, old = p[t], p[t - 1]
        d_new = dH if new == H else dA
        hit = np.flatnonzero(d_new[t:t + horizon] < reach)
        end = t
        while end < len(p) and p[end] == new:
            end += 1
        if len(hit) and t + hit[0] < end:
            p[t:t + hit[0]] = old                  # flip at reception, not at the kick
        else:
            p[t:end] = old                         # no receiver control: not a change
    return p


def rule_invisible(path, f):
    p = path.copy()
    inv = f["ball_invisible"] > 0
    n = len(p)
    t = 1
    while t < n:
        if inv[t] and not inv[t - 1]:
            e = t
            while e < n and inv[e]:
                e += 1
            hold = p[t - 1]
            if hold in (H, A):
                seg = p[t:e]
                seg[np.isin(seg, [H, A])] = hold   # D/X decisions stay; no team flip unseen
            t = e
        t += 1
    return p


def rule_contested(path, f, reach=1.5):
    p = path.copy()
    both = (f["ball_invisible"] == 0) & (f["d_H1"] < reach) & (f["d_A1"] < reach)
    no_ctrl = (f["ctrl_H_25"] < 0.2) & (f["ctrl_A_25"] < 0.2)
    p[both & no_ctrl & np.isin(p, [H, A])] = X
    return p


RULES = {"dead_ball": rule_dead_ball, "dead_tree": rule_dead_tree, "pass_flight": rule_pass_flight,
         "invisible": rule_invisible, "contested": rule_contested}
