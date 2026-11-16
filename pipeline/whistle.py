"""Referee-whistle evidence for dead ball (soft, like the other dead-ball clues in v3).

WHY (plain words)
-----------------
Our model misses many stoppages for fouls and injuries: the ball stays visible and there is no
set-piece tag, so the tracking barely changes. But the referee whistles. pipeline/audio.py turns
the video's sound into a per-frame "whistle score" (`whistle_db`, how strongly this referee's
two-note whistle is present). This module turns that score into dead-ball evidence.

THE RULE, STEP BY STEP
----------------------
1. FIND WHISTLES. A whistle is at least 2 frames (80 ms) in a row with whistle_db >= THRESHOLD_DB
   (a 1-frame dip inside is bridged).
2. KEEP ONLY "STOP" WHISTLES. Referees also whistle to START play (kickoffs, some free kicks).
   Those come after players have already stopped, so they are dropped. A whistle counts as a stop
   when players were moving before it (mean speed over the previous 3 s >= MOVING_MS) and slow
   down after it (mean over +1..+4 s is at least SLOWDOWN_MS lower).
3. MARK THE STOPPAGE. From the whistle until the ball is kicked again (ball speed >= KICK_MS for
   KICK_FRAMES frames in a row, looked for from 2 s after the whistle), capped at MAX_S seconds.
4. SOFT EVIDENCE. Those frames get BONUS added to the decoder's log-score for D. It nudges, it
   does not force: the decoder can still keep the ball live if the other evidence is strong.

HOW THE NUMBERS WERE SET (no answers used)
------------------------------------------
- The fingerprint and THRESHOLD_DB come from this match's kickoffs (a whistle is certain there):
  5 of 6 kickoff whistles score 28-37; 99% of all frames score under 14.2. 20 sits well above
  the background and finds ~0.7 whistles a minute, a normal rate for a match.
- Check without answers: after "stop" whistles where the model says the ball is live, the
  players' mean speed falls from 2.0 to 1.3 m/s within 2 s and stays there; at random live
  moments it stays near 2.0. That is the signature of a real stoppage.
- BONUS = 3.0 is the strength already used for the strong dead-ball clues (ball off the pitch,
  set piece with players stopped) in stoppage_tree.LEAF_BONUS. Not tuned.
"""
import numpy as np

from pipeline import dead_tree as DT

THRESHOLD_DB = 20.0
MIN_FRAMES = 2
MOVING_MS = 1.5
SLOWDOWN_MS = 0.3
KICK_MS = 8.0
KICK_FRAMES = 3
MAX_S = 25
BONUS = 3.0
FPS = 25


def onsets(whistle_db, threshold=THRESHOLD_DB):
    """Start frame of every whistle (step 1)."""
    hit = whistle_db >= threshold
    bridged = hit.copy()
    bridged[1:-1] |= hit[:-2] & hit[2:]          # bridge a single-frame dip
    s, e = DT.runs(bridged)
    return s[(e - s) >= MIN_FRAMES]


def is_stop(o, speed):
    """Step 2: players moving before the whistle and slower after it."""
    if o < 3 * FPS or o + 4 * FPS > len(speed):
        return False
    before = np.nanmean(speed[o - 3 * FPS:o])
    after = np.nanmean(speed[o + FPS:o + 4 * FPS])
    return before >= MOVING_MS and after <= before - SLOWDOWN_MS


def stoppage_end(o, ball_speed):
    """Step 3: first restart kick from 2 s after the whistle, or the MAX_S cap."""
    cap = min(len(ball_speed), o + MAX_S * FPS)
    fast = ball_speed[o + 2 * FPS:cap] >= KICK_MS
    if len(fast) >= KICK_FRAMES:
        run = np.convolve(fast.astype(int), np.ones(KICK_FRAMES, int), "valid") == KICK_FRAMES
        k = np.flatnonzero(run)
        if len(k):
            return o + 2 * FPS + int(k[0])
    return cap


def evidence(whistle_db, f, threshold=THRESHOLD_DB, log=None):
    """Boolean mask over one half: frames inside a whistle-marked stoppage (steps 1-3).

    whistle_db and every array in f cover the same frames (one half).
    """
    mask = np.zeros(len(whistle_db), bool)
    speed, ball_speed = f["speed_all"], f["ball_speed"]
    for o in onsets(whistle_db, threshold):
        if is_stop(o, speed):
            mask[o:stoppage_end(o, ball_speed)] = True
            if log is not None:
                log.append(int(o))
    return mask


RESTART_BONUS = 1.5     # half the stop-whistle strength: a weaker football prior
RESTART_S = 3


def restart_bonus(whistle_db, f, team_score, threshold=THRESHOLD_DB):
    """Option "-foulteam": after a foul, the fouled team usually takes the free kick.

    In plain words: the team that had the ball just before a stop whistle is usually the team that
    was fouled, and the fouled team restarts. So for the first RESTART_S seconds after the restart
    kick we nudge the decoder towards that team (RESTART_BONUS log-odds, soft, can be overruled).

    team_score: (n, 2) classifier log-scores for H and A on this half. "Who had the ball before"
    is the team with the higher mean score over the 3 s before the whistle. Returns an (n, 2)
    array of bonuses to add to the H and A columns.
    """
    out = np.zeros((len(whistle_db), 2))
    speed, ball_speed = f["speed_all"], f["ball_speed"]
    for o in onsets(whistle_db, threshold):
        if not is_stop(o, speed):
            continue
        team = int(team_score[o - 3 * FPS:o].mean(axis=0).argmax())
        end = stoppage_end(o, ball_speed)
        out[end:end + RESTART_S * FPS, team] += RESTART_BONUS
    return out


def bonus(whistle_db, f, threshold=THRESHOLD_DB):
    """Step 4: log-score added to D on each frame (0 where there is no whistle evidence)."""
    return BONUS * evidence(whistle_db, f, threshold).astype(float)
