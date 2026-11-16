# Éire 404: ball possession, BAR-ATH

Technical document for More than a Hack 2027, Phase 1 (Ball Possession).

| | |
|---|---|
| Possession file | `Eire404.csv`, 199,319 rows, SHA-256 `fea816bf28f8fae28c473919262d603342ffccdded1eff88c4b9fa0ed5da6ca2` |
| Model | `hybrid_stoppage_soft-midpoint-edges-min25-whistle` ("v3 + whistle") |
| Public window (5 min, 17 possessions) | spellF1 0.600 [90% CI 0.48, 0.71], macroF1 0.793 |
| Published floor on the same window | spellF1 0.326 [0.21, 0.45] |
| Reproduce | `python -m pipeline` (section 7) |

## 1. Summary

Every frame gets one of H (home ball), A (away ball), D (dead ball) or X (no answer). The pipeline
is a chain of small, testable modules:

```
tracking + Smart Tagging + the tactical video's sound
   -> 42 causal per-frame features
   -> LightGBM classifier, trained on labels made by an algorithm (no human labels)
   -> dead-ball decision tree + referee-whistle detector, added as soft evidence for D
   -> 4-state Viterbi decoder (turns noisy frame guesses into whole possessions)
   -> rule layer (passes in flight, ball not tracked, contested balls)
   -> stoppage clean-up (merge, drop short) -> 1 s minimum spell
   -> Eire404.csv
```

The decoder is the key step: on its own it lifts the public spellF1 from 0.16 to 0.53, because
spellF1 scores whole possessions and frame-by-frame guesses split them into fragments.

## 2. Approach, and why

**Why a classifier and a decoder, not one or the other.** A frame classifier sees the local
evidence well (who is near the ball, how fast players move) but flickers between labels. spellF1
matches whole possessions, so every flicker costs a possession. A decoder that charges a cost for
every label change removes the flicker. We use both: LightGBM for the per-frame probabilities,
Viterbi to choose the best whole sequence.

**Why labels from an algorithm.** The rules forbid hand labels on this match, and we had no
labelled data from other matches. So a hand-written "teacher" model labels the match, and the
classifier learns from those labels (section 4). The teacher may look at past and future frames,
because it only makes training labels. The classifier's features use only past and present
frames, so the learned model does not depend on seeing the future.

**Why the dead-ball work.** After the first model, dead ball was the weakest class (public recall
0.23), and a missed or fragmented stoppage splits a possession in two. The later stages all target
that: a decision tree of dead-ball evidence, fed in softly, a clean-up of stoppage fragments, and
finally the referee's whistle, which catches fouls and injuries that the tracking cannot see.

**How we chose between versions.** The public window has only 17 possessions; one possession moves
spellF1 by about 0.06, and the 90% intervals of all our later versions overlap almost completely.
We did not tune on it. We chose on label-free numbers over the whole match: possession count,
median possession length, number of stoppages, and how many stoppages were tiny or almost touching
(signs of one stoppage chopped into pieces).

| Version | public spellF1 [CI90] | macroF1 | possessions (whole match) | median s | stoppages |
|---|---|---|---|---|---|
| Published floor | 0.326 [0.21, 0.45] | 0.621 | 594 | 4.6 | 133 |
| v1: LightGBM + Viterbi + rules | 0.550 [0.42, 0.67] | 0.654 | 350 | 9.1 | 114 |
| v2: + dead-ball tree | 0.564 [0.43, 0.69] | 0.718 | 362 | 6.4 | 138 |
| v3: soft tree + clean-up + 1 s minimum | 0.584 [0.43, 0.71] | 0.721 | 290 | 10.5 | 103 |
| **v3 + whistle (submitted)** | **0.600 [0.48, 0.71]** | **0.793** | **289** | **10.2** | **106** |

v3 was chosen over v2 for its structure: fewest stoppages, none under 3 s apart, the longest median
possession, and public dead-ball precision of 0.95. The whistle then raised public dead-ball recall
from 0.37 to 0.56 at the same precision, with the whole-match structure unchanged. A placebo test
(the same whistle scores shifted in time) loses the whole gain, so it comes from the real timing of
the whistle and not from adding dead-ball evidence as such (section 5).

## 3. Assets used, and how

| Asset | Used for | Notes |
|---|---|---|
| Tracking (ATD) | all positional features; the teacher | Metres = normalised × (105.3, 68.0), checked on the kickoff spot. Player speed column is m/s (checked against positions, r = 0.95). The ball speed column was unusable (r = −0.03), so ball speed is derived from positions. Slots 1–24 home, 25–46 away. |
| Smart Tagging | features; decoder transitions; dead-ball tree; stoppage clean-up | Used as evidence, never as the answer (details below). |
| Homography | checks only | It maps the pitch to image pixels, not metres. Loaded and checked; not a model input. |
| Video (tactical): sound track | referee-whistle detector (section 5) | Frame-aligned with the tracking (frame i = second i/25). Extracted with ffmpeg to mono 16 kHz. |
| Video images; broadcast video | not used | The broadcast sound was tried for kick detection (dropped, below). |
| Public ground truth (5 min) | scoring checks only | Excluded from training and never used to pick thresholds. |

**How the Smart Tagging is used.** The FAQ invites its use; it is automatic, not ground truth, and
has errors, so every use is a nudge that other evidence can overrule:

- **Features:** the latest started tag's code and team, and time since it started. Only tags that
  have started by frame t are used; tag end times are never used as features.
- **Decoder:** while a tag is active, the chance of switching state is adjusted (e.g. a set-piece
  start makes "to dead ball" more likely; a possession phase for a team makes "to that team" more
  likely).
- **Dead-ball tree:** the kickoff tag marks the pre-kickoff stretch; a set-piece tag combined with
  stopped players or the ball off the pitch is evidence of a stoppage.
- **Clean-up:** a short stoppage is kept only if a set-piece tag lies near it (section 5).

**Tried and dropped** (all logged in `reports/experiments.md`):

| Idea | Result | Why dropped |
|---|---|---|
| Restart tree: after a throw-in or corner, the team that did not last touch the ball restarts | changed 0.07% of frames | the ball is tracked at the restart in 44 of 48 stoppages, so there was almost nothing to fill; for corners it disagreed with the tag team in 8 of 8 clear cases |
| Crowd rule: ball untracked in a pack -> the team with more players there | anti-predictive | where the ball is tracked inside such a pack, the larger group has it only 41.9% of the time (782 frames) |
| End tagged stoppages at the tag's midpoint | public D precision 0.94 -> 0.75 | moved stoppage ends into live play |
| Move stoppage edges using player speed | no gain | added complexity without a structural benefit |
| Dead-ball tree as a hard override | same score as soft | soft gave fewer stoppages (103 vs 112) and longer possessions |
| Dropping the tagging feature group | +0.029, inside noise | kept: the gain is noise, and the decoder uses the tags anyway |
| Foul restart: after a whistle, the team that had the ball restarts | changed 28 frames | the ball is tracked at restarts, so the model already gets the team right |
| Sound features (whistle, crowd loudness) inside LightGBM | spellF1 0.600 -> 0.524 | a time-shifted placebo scored better than the real sound: the teacher's labels carry no whistle signal to learn, so the whistle works as a rule, not a feature |
| Self-training: retrain LightGBM on our own best output | spellF1 -0.015 | more fragments (112 stoppages, median possession 9.4 s) |
| Kick sounds: hear touches to time possession changes | per-touch AUC 0.47 (broadcast), 0.60 vs placebo 0.56 (tactical) | the microphones do not resolve single kicks over crowd noise (`scripts/kick_feasibility.py`) |
| Who touched it, from tracking: acceleration-peak kicks (Hoernig et al. 2015); a switch only stands after the new team has sole control (possession zones, Vidal-Codina et al. 2022) | spellF1 0.500-0.600, never above the submitted model | acceleration peaks are not near players (ball jitter, airborne balls); proximity cannot tell pressing from winning the ball (`scripts/touch_feasibility.py`) |

## 4. Training data

- **Labelled datasets used: none from people.** No hand labels, no event data, no external data.
- **Training labels** come from a hand-written 3-state hidden Markov model (H, A, D) in
  `possession/model.py`. It has no learned parameters: a player within reach of the ball and
  nearer than any opponent is evidence for his team; a ball off the pitch, walking players or a
  still ball is evidence for D; fixed switching costs turn this into possessions. Its switching
  cost (5) was picked on whole-match plausibility (median possession 13 s), not on the public window.
- An X ("contested") label is added where both teams have a player within reach, neither has had
  sole control within 0.5 s, and the ball is not in fast flight.
- **Volume:** all scored frames of this match (11,185–90,651 and 110,808–190,197) except the
  public window (11,185–18,684), about 151,000 frames.
- **Model:** LightGBM multiclass, learning rate 0.05, 63 leaves, min 200 samples per leaf,
  fixed seed, deterministic mode. Cross-validation is 5-fold grouped by label run, so no run is
  split across train and validation.

## 5. Possession rules implemented

**Who has the ball.** A team has the ball while one of its players is in reach of it and nearer
than any opponent, and keeps it until the evidence for the other team is strong enough to pay the
decoder's switching cost.

**Loose balls and passes.**
- *Pass in flight:* a change of team during fast ball flight is moved to the first frame the new
  team has a player in reach. If no one reaches it within 2 s, the change is undone.
- *Ball not tracked:* no change of team while the ball is unseen; the last label holds until the
  ball reappears.

**Challenges.** Both teams within reach and neither in control -> X. The scorer fills X with the
previous label, so in practice the team that had the ball keeps it until the challenge is won.

**Dead ball (decision tree).** Each frame takes the first test that fires, most certain first. The
order was fixed from football reasoning before looking at scores.

| Rank | Evidence | Strength (log-odds added to D) |
|---|---|---|
| 1 | before this half's kickoff tag | forced to D |
| 2 | set-piece tag just started, and players stopped or ball off the pitch | 3.0 |
| 3 | ball tracked outside the lines for at least 2 s | 3.0 |
| 4 | (dropped in v3: set-piece tag near and ball still) | – |
| 5 | ball untracked for at least 5 s while players are slow (under 1.6 m/s) | 2.5 |
| 6 | players slow (under 1.3 m/s) and ball moving under 3 m/s, for at least 5 s, no tag needed (fouls, injuries) | 1.5 |

"Ball missing" alone is not a stoppage: the ball goes missing mostly in crowded boxes, aerial
duels and tackles, which are live play, so test 5 also needs slow players. Tests that depend on
player tracking are skipped during tracking gaps, because a missing player reads as "not moving".

**Referee whistle (fouls, injuries, offsides).** This referee's whistle sounds two notes at once,
near 3.8 kHz and 4.25 kHz; crowd noise almost never has both. For each frame we measure how far
the weaker of the two notes stands above the 2.5-6 kHz background (`pipeline/audio.py`). The
fingerprint and the 20 dB threshold were set on the kickoffs, where a whistle is certain (5 of 6
score 28-37 dB; 99% of all frames score under 14.2). Then (`pipeline/whistle.py`):
- a whistle is at least 80 ms above the threshold;
- it is a *stop* whistle only if players were moving before it (mean speed >= 1.5 m/s over 3 s)
  and slow down after it (at least 0.3 m/s lower over the next 1-4 s); kickoff and restart
  whistles come after players have stopped, so they are ignored;
- from a stop whistle until the restart kick (ball >= 8 m/s for 3 frames, looked for from 2 s on,
  at most 25 s), D gets +3.0 log-odds, the same strength as the strong tree tests.

Of 76 whistles during play, 31 are stop whistles; they add 2.5 minutes of D and remove none. The
result is the same at thresholds 16, 20 and 24. Checked without answers: after stop whistles the
players' mean speed falls from 2.0 to 1.3 m/s within 2 s and stays low, while at random live
moments it stays near 2.0.

**Restarts and stoppage clean-up.**
- Two stoppages less than 3 s apart become one (the "play" between them is not real play).
- A stoppage under 2 s with no set-piece tag within 6 s is removed and given back to the team that
  had the ball.
- Finally, no possession or stoppage may be shorter than 1 s; a shorter one joins the label before it.
- After a restart, possession is decided by the normal evidence (who reaches the ball), not by a
  fixed "other team restarts" rule (see the dropped restart tree).

## 6. Missing and noisy tracking

- *Ball not tracked:* uncertainty features (ball visible, frames since last seen, players tracked);
  the decoder holds the last label; the "ball not tracked" rule blocks team changes while unseen;
  only a long absence with slow players counts as a stoppage.
- *Noisy positions and speeds:* ball speed is derived from positions; trailing averages over
  2–3 s (the most important feature group: removing it drops public spellF1 from 0.550 to 0.242).
- *Noisy tags:* used only as soft evidence (section 3).
- *Noisy sound:* the whistle needs both of its notes at once, a minimum length and a slowdown of
  the players, and it only nudges the decoder.
- *Fragments:* Viterbi switching costs, stoppage merge and drop, and the 1 s minimum.

## 7. Reproduction

```
python -m venv .venv
.venv/Scripts/python -m pip install -r requirements-lock.txt     # Python 3.14; LightGBM 4.7.0
# install ffmpeg (tested with 4.2.3); if it is not on PATH, set FFMPEG to its full path
# put the organisers' data bundle in data/ (see data/README.md)
.venv/Scripts/python -m pipeline --out Eire404.csv
.venv/Scripts/python evaluation/validate_submission.py Eire404.csv
```

`python -m pipeline` rebuilds everything from the raw data: parse the tracking, build features,
make the training labels, train LightGBM, extract the sound and score the whistle, decode,
validate. About 1-2 minutes. Two runs from scratch give byte-identical files (SHA-256 above).
Tests: `python -m pytest -q tests` (63 tests, synthetic data only). To compare
versions, `python -m pipeline.run --proba full --variants <name>` builds any variant listed in
`pipeline/run.py`, and `python scripts/compare_candidates.py` prints the comparison table.

## 8. Limitations, and what we would fix with more time

- **Dead ball:** public recall is 0.56 with the whistle. Stoppages without a detected whistle,
  tag or off-pitch ball are still missed. Next: the broadcast video's replays and close-ups, which
  appear when play stops (the broadcast is already aligned to our frames via the whistle: +4011
  frames in the 1st half, +26917 in the 2nd).
- **Duels:** most remaining home/away errors are duels and pressing, where an opponent is nearest
  to the ball for seconds but the team in possession keeps it. Proximity cannot tell pressing from
  winning the ball (section 3). Next: body orientation or contact from the video, or the direction
  of the next pass.
- **The whistle detector is tuned to this referee's whistle** (its two notes). Another match would
  need its fingerprint re-measured on that match's kickoffs.
- **The teacher limits the student.** The classifier can only be as good as the algorithmic labels;
  hand labels on other matches (allowed) would be the biggest single improvement.
- **Thresholds are reasoned, not tuned** (3 s merge, 2 s short, 1 s minimum, soft strengths),
  because the public window cannot tune them. A second labelled match would let us tune them.
- **Our evaluation is weak:** 17 public possessions. Every comparison after the decoder lies
  inside overlapping intervals.

## 9. Tools and LLMs

- Python, NumPy, SciPy, pandas, LightGBM, scikit-learn, pytest; all versions pinned in
  `requirements-lock.txt`. ffmpeg 4.2.3 extracts the sound.
- Methods consulted: referee-whistle detection by band energy and spectral peakiness (e.g. arXiv
  1411.6496); Hoernig et al. 2015 (kick detection from ball acceleration); Vidal-Codina et al. 2022,
  arXiv 2202.00804 (possession zones and duels from tracking data).
- LLM coding assistants helped write and review code: Claude Code (Anthropic) and, for two small
  helper functions written from a plain-text specification only, Gemini (Google web chat). The
  Gemini functions were checked against ours on 3,000 random cases and were identical. No LLM is
  part of the pipeline; it runs fully offline.
