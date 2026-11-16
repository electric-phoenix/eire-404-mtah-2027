# Experiments

Every change of more than 0.005 spellF1 is traced to a specific change (Hard Rule 8). Scores are
from the official `evaluation/run_evaluation.py` on the public window, with 90% bootstrap
intervals (1000 draws). "Full poss" counts possessions over the whole match (no labels needed).

| # | date | id | change | spellF1 [CI] | macroF1 | full poss |
|---|---|---|---|---|---|---|
| 0 | 11-10 | baseline_smooth50 | published floor | 0.326 [0.21, 0.45] | 0.621 | 594 |
| 1 | 11-10 | hmm_v1 | hand-built 3-state HMM, cost_HA 12 | 0.474 [0.27, 0.66] | 0.684 | 190 |
| 2 | 11-10 | hmm_v2 | cost_HA 12 → 5 (picked on full-match plausibility) | 0.473 [0.29, 0.65] | 0.668 | 284 |
| 3 | 11-10 | lgbm_full_classical | LightGBM on pseudo-labels from hmm_v2; argmax | 0.159 [0.12, 0.21] | 0.598 | 871 |
| 4 | 11-10 | lgbm_full_classical_s50 | + ±50 majority vote | 0.420 [0.28, 0.56] | 0.607 | 252 |
| 5 | 11-10 | lgbm_full_viterbi_nomod | 4-state Viterbi, uniform transitions | 0.528 [0.43, 0.61] | 0.624 | 364 |
| 6 | 11-10 | lgbm_full_viterbi | + Smart Tagging transition modifiers | 0.528 [0.43, 0.61] | 0.624 | 369 |
| 7 | 11-10 | **lgbm_full_hybrid** | + rule layer (dead-ball, pass-flight, invisible, contested) | **0.550 [0.42, 0.67]** | 0.654 | 350 |
| 8 | 11-10 | lgbm_full_hybrid_min25 | + 1 s minimum spell | 0.564 [0.43, 0.69] | 0.663 | 291 |
| 9 | 11-13 | **lgbm_full_hybrid_tree** | dead-ball override replaced by the 6-leaf dead-ball decision tree (`pipeline/dead_tree.py`) | 0.564 [0.43, 0.69] | 0.718 | 362 |
| 10 | 11-13 | hybrid_tree-ball_missing_slow | tree without the "ball missing >= 5 s and players slow" leaf | 0.537 [0.41, 0.65] | 0.653 | 352 |
| 11 | 11-13 | hybrid_tree-players_still | tree without the "players and ball still >= 3 s" leaf | 0.564 [0.43, 0.69] | 0.718 | 358 |
| 12 | 11-13 | hybrid_tree-setpiece_ball_still | tree without the "set piece, ball still" leaf | 0.579 [0.44, 0.71] | 0.719 | 358 |
| 13 | 11-13 | **lgbm_full_hybrid_tree_restart** | hybrid_tree + throw-in/corner restart tree (`pipeline/restart_tree.py`) | 0.564 [0.43, 0.69] | 0.718 | 363 |
| 14 | 11-13 | hybrid_tree_restart-literal_corner | corners get the plain "other team restarts" idea (ablation, conflicts with the tags) | 0.564 [0.43, 0.69] | 0.718 | 364 |
| 15 | 11-13 | **lgbm_full_hybrid_tree_crowd** | hybrid_tree + crowd rule (`pipeline/cluster_tree.py`): ball untracked in a pack of >= 6 players -> team with more players in the pack | 0.564 [0.43, 0.69] | 0.718 | 377 |
| 16 | 11-14 | **lgbm_full_hybrid_stoppage** | reworked leaves + merge close stoppages + drop short + tag-midpoint/player-speed edges (`pipeline/stoppage_tree.py`) | 0.526 [0.38, 0.66] | 0.679 | 341 |
| 17 | 11-14 | hybrid_stoppage_soft | dead-ball leaves as soft log-odds on D in Viterbi instead of a hard override | 0.526 [0.38, 0.66] | 0.679 | 342 |
| 18 | 11-14 | hybrid_stoppage-midpoint | row 16 without the tag-midpoint end | 0.568 [0.43, 0.69] | 0.685 | 349 |
| 19 | 11-14 | hybrid_stoppage-edges | row 16 without the player-speed / out-of-bounds edges | 0.526 [0.38, 0.66] | 0.699 | 338 |
| 20 | 11-14 | **lgbm_full_hybrid_tree_tidy** | hybrid_tree + merge stoppages < 3 s apart + drop untagged stoppages < 2 s (nothing else) | 0.579 [0.44, 0.71] | 0.715 | 336 |
| 21 | 11-14 | hybrid_tree_tidy-merge | row 20 without the merge step | 0.564 [0.43, 0.69] | 0.718 | 356 |
| 22 | 11-14 | hybrid_tree_tidy-short | row 20 without the drop-short step | 0.579 [0.44, 0.71] | 0.715 | 340 |
| 23 | 11-14 | hybrid_tree_tidy-min25 | tidy + 1 s minimum spell | 0.579 [0.44, 0.71] | 0.716 | 285 |
| 24 | 11-14 | hybrid_stoppage-midpoint-edges | reworked tree (hard) + merge + drop short | 0.579 [0.44, 0.71] | 0.719 | 334 |
| 25 | 11-14 | hybrid_stoppage_soft-midpoint-edges | reworked tree as SOFT evidence + merge + drop short | 0.579 [0.44, 0.71] | 0.719 | 334 |
| 26 | 11-14 | hybrid_stoppage_soft-midpoint-edges-rework | as row 25 with the original tree leaves | 0.579 [0.44, 0.71] | 0.719 | 334 |
| 27 | 11-14 | **hybrid_stoppage_soft-midpoint-edges-min25** | row 25 + 1 s minimum spell | 0.584 [0.43, 0.71] | 0.721 | 290 |
| 28 | 11-16 | **v3 + whistle** (`...-min25-whistle`) | referee-whistle evidence for D from the video's sound (`pipeline/audio.py`, `pipeline/whistle.py`) | 0.600 [0.48, 0.71] | **0.793** | 289 |
| 29 | 11-16 | v3 + whistle16 | looser whistle threshold (16 dB) | 0.600 [0.48, 0.71] | 0.793 | 288 |
| 30 | 11-16 | v3 + whistle24 | stricter whistle threshold (24 dB) | 0.600 [0.48, 0.71] | 0.793 | 289 |
| 31 | 11-16 | v3 + whistle + foulteam | after a stop whistle, nudge the restart (3 s, +1.5) to the team that had the ball before | 0.600 [0.48, 0.71] | 0.793 | 289 |
| 32 | 11-16 | proba `audio` + v3 + whistle | LightGBM retrained with 6 causal sound features (whistle, crowd loudness) | 0.524 [0.40, 0.64] | 0.778 | 284 |
| 33 | 11-16 | proba `audioplacebo` + v3 + whistle | as row 32, sound shifted 20,000 frames (placebo) | 0.589 [0.44, 0.73] | 0.789 | 279 |
| 34 | 11-16 | proba `self` + v3 + whistle | self-training: LightGBM learns from row 28's output instead of the teacher | 0.585 [0.44, 0.71] | 0.796 | 280 |
| 35 | 11-16 | proba `selfaudio` + v3 + whistle | rows 32 + 34 together | 0.556 [0.42, 0.67] | 0.786 | 278 |

Who touched it, from tracking (16 Nov, **negative result, nothing adopted**; `scripts/touch_feasibility.py`).
Methods from Hoernig et al. 2015 (kicks = local peaks of ball acceleration, Savitzky-Golay
smoothed, credited to the nearest player in reach) and Vidal-Codina et al. 2022 (possession zones,
duels keep the previous team):
- Acceleration peaks are not touches here: nearest player at peaks median 2.1 m vs 2.3 m at random
  frames; strongest peaks 5.6 m. Ball jitter and airborne balls projected to the ground dominate.
- The model is not late: its 176 switches sit at the new team's control onset (median 0 to -3
  frames). The 12 public-answer switches come 6-16 frames after the onset (control confirmed).
- Public team errors (diagnosis only) are mostly duels: ball tracked, an opponent nearest for
  seconds (true team nearest 16-42% of those frames).
- Confirmed-control rule (revert switches without sole control; or move switches to confirmed
  control), 12 settings: spellF1 0.500-0.600, macroF1 0.772-0.793, never above v3 + whistle
  (0.600 / 0.793). Proximity cannot tell pressing from winning the ball.

Kick sounds (16 Nov, idea 7, **negative result, nothing adopted**; `scripts/kick_feasibility.py`).
Goal: hear ball touches to time possession changes. Tested without answers against tracking-derived
touches (sharp ball-velocity changes) and with time-shifted placebos throughout:
- Band-energy onsets (60-300 / 300-1500 / 1500-5000 Hz), tactical sound: no lag shows a rise
  beyond noise around 662 touches.
- 2 ms click detector (energy vs 250 ms background): per-touch AUC 0.40-0.48 tactical.
- Broadcast sound (pitch-side microphones), aligned to tactical frames with the referee whistle
  (+4011 / +26917 frames per half): click AUC 0.57-0.63, but a learned detector on held-out 5-min
  blocks scores 0.473 (placebo 0.490). Tactical learned detector 0.599 vs placebo 0.561.
- Moving the best model's team switches to the strongest click within 0.5 s: macroF1 0.791-0.795,
  spellF1 0.600 (tactical) / 0.600 (broadcast), no better than the placebos (0.795 / 0.610 and
  0.792 / 0.600). Conclusion: the microphones do not resolve single kicks over crowd noise.

Medium ideas (rows 31-35), all tested against row 28 (v3 + whistle), **none adopted**:
- Foul restart team (row 31): changes 28 frames in 3 places over the whole match, nothing on the
  public window. The ball is tracked at restarts, so the model already gets the team right there
  (same reason the restart tree was inert). Rejected: no effect.
- Sound features in the classifier (row 32): worse (spellF1 0.600 -> 0.524, macroF1 0.793 -> 0.778),
  and the time-shifted placebo (row 33) scored BETTER than the real sound. No sound feature reached
  the top 15 by gain. The teacher's labels carry no whistle information, so the classifier has
  nothing to learn from the sound; the whistle works as a rule, not as a feature. Rejected.
- Self-training (row 34): macroF1 +0.003 and D recall 0.56 -> 0.59, but spellF1 -0.015 and more
  fragments (stoppages 106 -> 112, median possession 10.2 -> 9.4 s). Rejected on structure.
- Both (row 35): worse on every column. Rejected.
Efficiency: the sound step was streamlined (unused single-pitch measure removed, Fourier transforms
on all cores, only the 2.5-6 kHz bins converted): 23 s from the video including ffmpeg 4.2.3
(11 s when the WAV is cached), down from ~95 s. Row 28 still regenerates byte-identically
(SHA-256 fea816bf...), also after re-extracting the WAV.

Referee whistle (rows 28-30). The tactical video's sound is frame-aligned (frame i = second i/25).
This referee's whistle is two notes at once (~3.7-3.9 and ~4.2-4.3 kHz), found on the kickoffs
(5 of 6 score 28-37 dB; 99% of frames score under 14.2). `whistle_db` = the weaker of the two
notes. 76 whistles in play, 31 "stop" whistles (players moving before, slower after). Frames from a
stop whistle to the restart kick get +3.0 log-odds on D (same strength as the strong tree leaves).
Results: public D recall 0.37 -> 0.56 at precision 0.95, macroF1 0.721 -> 0.793; whole-match
structure unchanged (289 possessions, median 10.2 s, 106 stoppages); 2.5 min newly D, none removed;
newly-D frames have median player speed 1.12 m/s (v3 D 0.81, live 1.91). Identical at thresholds
16/20/24. **Placebo** (same scores shifted by +3000, -7000, +20000 frames): D recall stays 0.37 and
macroF1 0.721-0.726, so the gain comes from the real whistle timings. The +3000 placebo scored
spellF1 0.615 by chance, another reminder that public spellF1 cannot rank these. Label-free
evidence: after stop whistles where v3 says live, player speed falls 2.0 -> 1.3 m/s within 2 s;
at random live moments it stays ~2.0. Regenerates byte-identically.

Teammate 1 round (rows 23-27; table from `scripts/compare_candidates.py`, bootstrap 1000):
every public spellF1 interval overlaps almost completely, so the public window cannot rank these.
On the full match, row 27 has the most plausible structure: 103 stoppages, none under 3 s apart,
median possession 10.5 s (v2: 6.4 s), dead-ball precision kept at 0.95. It regenerates
byte-identically (SHA-256 9b1f1f94...) and validates. Rows 25 and 26 are identical: the leaf
rework makes no difference once the evidence is soft. **Recommended freeze candidate, pending the
Lead's decision.** Full write-up: `reports/teammate1_handover.md`.

Tidy (rows 20-22): stoppages 138 -> 111, under 2 s 16 -> 3, pairs under 3 s apart 22 -> 0, median
possession 6.4 -> 7.0 s, full-match possessions 362 -> 336. Public D precision 0.94 -> 0.91, recall
unchanged at 0.37. The merge step does the work (row 21 returns to 0.564). spellF1 +0.015 is inside the
noise band; the structural gain (fewer fragments) is the reason to prefer it.

Stoppage clean-up (rows 16-19): stoppages 138 -> 112, under 2 s 16 -> 5, pairs under 3 s apart 22 -> 2,
so the fragmentation targets were met, but the median possession barely moved (6.4 -> 6.6 s; soft
variant 8.4 s with 103 stoppages). Public D precision fell 0.94 -> 0.75 and recall 0.37 -> 0.32. The
tag-midpoint end is the cause: removing it restores spellF1 to 0.568 (row 18). **Not adopted.**

Crowd rule (row 15): changes 222 frames on the full match, none in the public window, so the public
score cannot move. Only 372 of 85,831 ball-untracked frames qualify as a crowd. The check that
matters: on frames where the ball IS tracked and sits inside such a pack, the team with more players
in the pack is the decoder's team on only 41.9% of 782 frames, below a coin flip. Numerical
superiority in a pack does not predict possession (the ball carrier is usually outnumbered).
**Rejected; kept as an ablation.** Full-match possessions rise 362 -> 377.

Restart tree (rows 13-14): changes 105 frames on the whole match (0.07% of alive frames) and nothing
on the public window, which has no throw-in or corner. The ball is tracked at the restart in 44 of 48
stoppages, so there is almost no blind stretch to fill. Corner team vs Smart Tagging team (n=12):
baseline_smooth50 5, hybrid 10, hybrid_tree 9, hybrid_tree_restart 9, literal idea 8. The literal
idea disagrees with the tag in 8 of 8 corners where the last controller is clear. See `scripts/restart_report.py`.

Dead-ball tree (rows 9-12), public window: D recall 0.23 -> 0.37, D precision 0.92 -> 0.94. The
spellF1 changes are inside the noise band; the macroF1 gain (+0.064) comes almost entirely from the
`ball_missing_slow` leaf (row 10 returns to 0.653). Full-match median possession falls from 9.1 s to
6.4 s because more stoppages split possessions: **not yet decided whether to freeze**. The leaf
order was fixed in advance (see the module docstring), not chosen from these numbers.

Rule ablations, feature-group ablations and the frozen choice are in `reports/model.md`.

## hmm_v2 switching-cost sweep (full match, no labels)

| cost_HA | possessions | mean s | median s | spells under 2 s |
|---|---|---|---|---|
| 3 | 386 | 13.1 | 7.8 | 68 |
| 4 | 327 | 15.4 | 10.4 | 40 |
| **5** | 284 | 17.7 | 13.0 | 21 |
| 6 | 270 | 18.6 | 13.6 | 17 |
| 8 | 236 | 21.2 | 16.1 | 4 |
| 12 | 190 | 26.3 | 19.8 | 0 |
