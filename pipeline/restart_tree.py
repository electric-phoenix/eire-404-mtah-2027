"""Who has the ball after a throw-in or a corner: a decision tree that only runs where it applies.

The idea: whoever did NOT last have the ball is the team that restarts play. The ball is
untracked for a while after most restarts, and the decoder's "hold the last team" fallback then
gives the restart to the team that just lost it. This tree replaces that held label with the
restart team, for the blind stretch only.

The tree is entered once per stoppage (a run of D in the decoded path), and most stoppages leave
it at the first test, so no work or risk is spent on frames that are not these set pieces:

    stoppage (run of D)
      |-- Smart Tagging set piece nearby?
      |     |-- corner kick  -> leaf 1  corner_tag          restart team = the tag's team
      |     |-- throw-in     -> leaf 2  throw_in_tag        restart team = NOT the last controller
      |     `-- anything else (goal kick, free kick, kickoff) -> untouched
      `-- no tag: ball tracked outside a touchline during the stoppage?
            |-- yes          -> leaf 3  throw_in_touchline  restart team = NOT the last controller
            `-- no           -> untouched
      then, for a leaf with a team:
        last controller unclear (nobody in 10 s, or both teams within 0.5 s)  -> untouched
        ball visible the moment play restarts                                -> untouched
        otherwise relabel the blind stretch: from the restart until the ball is seen again

Corners are the exception to the idea, on purpose. A corner goes to the attacking side because
the DEFENDER touched it last, and the attackers are the team that last had control. Smart
Tagging's corner team agreed with "the team that last controlled the ball" in 8 of 8 corners and
disagreed with "the other team" in 8 of 8. So corners follow the tag. `literal_corner=True`
applies the plain idea to corners instead; it exists only as a labelled ablation.

Evidence (labels are never read): last controller from since_ctrl_*, ball position from the
tracking, set pieces from the Smart Tagging tags.
"""
import numpy as np

from . import dead_tree as DT

H, A, D, X = 0, 1, 2, 3
LEAVES = ["corner_tag", "throw_in_tag", "throw_in_touchline"]

TAG_WINDOW = 150            # a tag within 6 s before the stoppage starts, or inside it
LOOKBACK = 500              # 20 s: how far back to look for the last controlling team
MAX_AGE = 250               # last control must be within 10 s of the stoppage
MIN_GAP = 12                # both teams controlling within 0.5 s of each other: unclear
MAX_BLIND = 150             # relabel at most 6 s of untracked ball after the restart
REAPPEAR = 5                # ball must be tracked this many frames in a row to count as seen
TOUCHLINE_MIN = 25          # ball tracked outside a touchline for >= 1 s
LENGTH, WIDTH = 105.3, 68.0


def last_controller(f, s):
    """0 (H) or 1 (A): the team that most recently had a player on the ball before frame s.

    None when nobody controlled it in the last MAX_AGE frames or both teams did within MIN_GAP
    of each other (a contested ball is not evidence of who lost it).
    """
    lo = max(0, s - LOOKBACK)
    last = {}
    for team, key in ((H, "since_ctrl_H"), (A, "since_ctrl_A")):
        hit = np.flatnonzero(f[key][lo:s] == 0)
        last[team] = lo + hit[-1] if len(hit) else None
    if last[H] is None and last[A] is None:
        return None
    if last[H] is None or last[A] is None:
        team = H if last[A] is None else A
        return team if s - last[team] <= MAX_AGE else None
    if abs(last[H] - last[A]) < MIN_GAP:
        return None
    team = H if last[H] > last[A] else A
    return team if s - last[team] <= MAX_AGE else None


def other(team):
    return None if team is None else 1 - team


def exited_over_touchline(f, s, e):
    """Ball tracked outside a touchline (top or bottom of the pitch) for >= TOUCHLINE_MIN frames."""
    seen = f["ball_invisible"][s:e] == 0
    y, x = f["ball_y"][s:e], f["ball_x"][s:e]
    out = seen & ((y < -0.5) | (y > WIDTH + 0.5)) & (x >= 0) & (x <= LENGTH)
    return int(out.sum()) >= TOUCHLINE_MIN


def find_tag(tags_h, s, e, types):
    near = [t for t in tags_h if t["code"] == "set pieces" and t["type"] in types
            and s - TAG_WINDOW <= t["a"] <= e]
    return min(near, key=lambda t: abs(t["a"] - s)) if near else None


def restart_team(f, tags_h, s, e, disable=(), literal_corner=False):
    """(leaf name, team) for the stoppage path[s:e], or (None, None) if the tree does not apply."""
    corner = find_tag(tags_h, s, e, ("corner kick",))
    if corner is not None and "corner_tag" not in disable:
        if literal_corner:
            return "corner_tag", other(last_controller(f, s))
        tag_team = corner["team"]
        return "corner_tag", tag_team if tag_team is not None else last_controller(f, s)
    if find_tag(tags_h, s, e, ("throw-in",)) is not None and "throw_in_tag" not in disable:
        return "throw_in_tag", other(last_controller(f, s))
    if ("throw_in_touchline" not in disable and find_tag(tags_h, s, e, ("goal kick", "free kick",
            "kick off (start)", "kick off (after goal)")) is None and exited_over_touchline(f, s, e)):
        return "throw_in_touchline", other(last_controller(f, s))
    return None, None


def blind_stretch(f, path, e):
    """Frames [e, end) after a restart during which the ball is untracked. Empty if it is seen."""
    n = len(path)
    inv = f["ball_invisible"]
    end = e
    while end < n and end - e < MAX_BLIND and path[end] != D:
        if end + REAPPEAR <= n and (inv[end:end + REAPPEAR] == 0).all():
            break
        end += 1
    return end


def apply(path, f, tags_h, disable=(), literal_corner=False, log=None):
    """Relabel the blind stretch after each throw-in and corner. Returns a new path."""
    p = path.copy()
    starts, ends = DT.runs(p == D)
    for s, e in zip(starts, ends):
        if e >= len(p) or s == 0:
            continue
        leaf, team = restart_team(f, tags_h, s, e, disable, literal_corner)
        if leaf is None:
            continue
        end = blind_stretch(f, p, e)
        changed = 0
        if team is not None and end > e:
            seg = p[e:end]
            changed = int(((seg == H) | (seg == A) | (seg == X)).sum() - (seg == team).sum())
            seg[(seg == H) | (seg == A) | (seg == X)] = team
        if log is not None:
            log.append({"leaf": leaf, "start": int(s), "end": int(e), "team": team,
                        "blind": int(end - e), "changed": changed})
    return p
