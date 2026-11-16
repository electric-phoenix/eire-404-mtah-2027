# Techniques used: what we have, what works, what to build on

For the team, to agree on one method and extend it. Deadline: 17 Nov 2026.

## Recommendation

**Finalise on v3 + whistle** (`hybrid_stoppage_soft-midpoint-edges-min25-whistle`, frozen 16 Nov). It is frozen,
reproducible (SHA-256 `fea816bf…`) and already in `Eire404.csv`. Build on top of it, and only
replace the frozen file if a change wins on the whole-match checks in the last section.

## How the method fits together

```
tracking + Smart Tagging
 1  features (42, causal)                    pipeline/features.py
 2  teacher labels (hand-written HMM)        possession/model.py, pipeline/pseudo_labels.py
 3  LightGBM per-frame probabilities         pipeline/train.py
 4  dead-ball tree -> soft evidence for D    pipeline/dead_tree.py, pipeline/stoppage_tree.py
 5  Viterbi decoder (4 states H/A/D/X)       pipeline/decode.py
 6  rule layer                               pipeline/decode.py
 7  stoppage clean-up                        pipeline/stoppage_tree.py
 8  1 s minimum spell                        pipeline/decode.py (enforce_min_spell)
 -> Eire404.csv                               python -m pipeline
```

Each step's output is the next step's input, so a new idea slots in at one step without touching
the others.

## Techniques in the submitted model

| # | Technique | What it does, simply | Evidence it helps | Code |
|---|---|---|---|---|
| 1 | Causal features | 42 numbers per frame from the past and present only: who is near the ball, speeds, ball visible, latest tag, trailing averages | trailing averages are vital: removing them drops public spellF1 0.550 -> 0.242 | `pipeline/features.py` |
| 2 | Teacher labels | a rule-based model labels the whole match (no humans), so we have something to train on | the rules forbid hand labels on this match | `possession/model.py` |
| 3 | LightGBM | learns from the teacher's labels, gives P(H), P(A), P(D), P(X) per frame | needed by step 5; public window excluded from training | `pipeline/train.py` |
| 4 | Dead-ball tree, soft | six ranked tests for "ball is dead"; each adds a bonus to D instead of forcing it | public D recall 0.23 -> 0.37; soft gives fewer stoppages than hard (103 vs 112) | `pipeline/dead_tree.py`, `stoppage_tree.soft_bonus` |
| 5 | Viterbi | picks the best whole sequence; every label change costs a penalty, so flicker disappears | **biggest gain**: 0.16 -> 0.53 public spellF1, 871 -> 364 possessions | `pipeline/decode.py` |
| 5b | Tag transition modifiers | while a tag is active, switching to the tagged state is cheaper | no measurable effect; kept because harmless | `decode.smart_tagging_modifier` |
| 6a | Pass flight | a team change mid-pass waits until the new team reaches the ball | public possessions 29 -> 23 | `decode.py` |
| 6b | Ball invisible | no team change while the ball is not tracked | +0.026, the most useful rule | `decode.py` |
| 6c | Contested -> X | both teams in reach, nobody in control -> no answer | no effect on spellF1 (scorer fills X) | `decode.py` |
| 7a | Merge stoppages | two stoppages < 3 s apart become one | pairs < 3 s apart 22 -> 0; this step does the work of the clean-up | `stoppage_tree.merge_close` |
| 7b | Drop short stoppages | a stoppage < 2 s with no set-piece tag within 6 s is removed | stoppages < 2 s 16 -> 3 | `stoppage_tree.drop_short` |
| 8 | 1 s minimum | anything shorter than 1 s joins the label before it | median possession 8.7 -> 10.5 s | `decode.enforce_min_spell` |

## Techniques tried and dropped

| Technique | Why dropped | Code (kept for ablation) |
|---|---|---|
| Restart tree: after throw-in/corner, the team that didn't last touch it restarts | changes 0.07% of frames: the ball is usually tracked at restarts. For corners it contradicts the tag team | `pipeline/restart_tree.py` |
| Crowd rule: ball lost in a pack -> team with more players there | wrong more often than right (41.9%) | `pipeline/cluster_tree.py` |
| End tagged stoppages at the tag's midpoint | D precision 0.94 -> 0.75 | `stoppage_tree.refine_edges` (`-midpoint`) |
| Move stoppage edges by player speed | no gain | `stoppage_tree.refine_edges` (`-edges`) |
| Hard dead-ball override | same score, worse structure than soft | `hybrid_stoppage` |
| "Set piece + ball still" dead-ball test | weak evidence; removed in the rework | `dead_tree.py` (`-rework` restores it) |
| Foul restart team: after a whistle, the team that had the ball restarts | changes 28 frames: the model already gets restarts right | `whistle.restart_bonus` (`-foulteam`) |
| Sound features inside LightGBM | worse, and the placebo beat the real sound: the teacher's labels hold no whistle signal to learn | `train.py --audio` |
| Self-training on our own output | more fragments (112 stoppages, median 9.4 s), spellF1 -0.015 | `train.py --labels` |
| Kick sounds: hear touches to time possession changes | not detectable: learned detector AUC 0.47 (broadcast) / 0.60 vs placebo 0.56 (tactical); snapping switches to clicks no better than placebo | `scripts/kick_feasibility.py` |
| Who touched it from tracking: acceleration-peak kicks; confirmed-control switches | acceleration peaks are not near players (tracking jitter, airborne balls); confirmed control 0.500-0.600 spellF1, never above best | `scripts/touch_feasibility.py` |
| Plain argmax / majority vote | far too fragmented (0.16 / 0.42) | `run.py` `classical*` |

## Added 16 Nov: the referee whistle (best so far, not yet frozen)

`-whistle` add-on (`pipeline/audio.py`, `pipeline/whistle.py`): this referee's whistle is two
notes at once (~3.8 and ~4.25 kHz). A "stop" whistle (players moving before, slower after) nudges
the frames up to the restart kick towards dead ball. Public D recall 0.37 -> 0.56 at precision
0.95, macroF1 0.721 -> 0.793, spellF1 0.584 -> 0.600; whole-match structure unchanged; the gain
disappears when the sound is shifted in time (placebo). Run `python -m pipeline.audio` first.

## Where to build next, best value first

| Idea | Step | Why | Effort |
|---|---|---|---|
| Better dead-ball detection for fouls and injuries (ball visible, no tag) | 4 | dead-ball recall is 0.37, the weakest point | medium |
| "Who touched it" from the TRACKING (not sound): ball changes direction near a player | 1 | late team changes; sound cannot do it (see dropped) | medium |
| Retrain LightGBM on v3's own output (second round of teacher labels) | 2-3 | the student is capped by the teacher | low, but risky to judge without labels |
| Tune switching cost / soft bonuses | 4-5 | set by reasoning, not tuned | low, but nothing reliable to tune on |

## How to test a change before it replaces v3

1. Add it as a new variant (or an option) in `pipeline/run.py`; never edit the frozen path in place.
2. Build it: `python -m pipeline.run --proba full --variants <name>`.
3. Add it to `CANDIDATES` in `scripts/compare_candidates.py` and run that script.
4. Prefer it only if, **over the whole match**, it has fewer tiny or almost-touching stoppages, a
   plausible possession count (about 290–335) and median length, and it does not drop public
   dead-ball precision. The public spellF1 is a sanity check only: one possession moves it 0.06.
5. Run `python -m pytest -q tests`; add a test with synthetic data for the new step.
6. To freeze: change `FROZEN["variant"]` in `pipeline/__main__.py`, build twice and compare
   SHA-256, validate, copy to `Eire404.csv`, update `Eire404.md`.
