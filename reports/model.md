# Model: frozen configuration

**Frozen (16 Nov 2026):** `full` features + `hybrid_stoppage_soft-midpoint-edges-min25-whistle`
(v3 + referee whistle: LightGBM -> dead-ball tree and stop-whistle evidence as soft log-odds on D ->
4-state Viterbi with Smart Tagging modifiers -> rule layer -> merge stoppages <3 s apart, drop
untagged stoppages <2 s -> 1 s minimum spell). Public spellF1 **0.600** [90% CI 0.48, 0.71],
macroF1 0.793. Regenerate with `python -m pipeline` (needs ffmpeg); two runs from scratch produce
byte-identical files (SHA-256 `fea816bf…`). Whistle method: `pipeline/audio.py`,
`pipeline/whistle.py`, reports/experiments.md rows 28-30.

**Previously frozen (v3, 15-16 Nov):** `hybrid_stoppage_soft-midpoint-edges-min25`, public spellF1
0.584 [0.43, 0.71], macroF1 0.721 (SHA-256 `9b1f1f94…`).

**Previously frozen (Model v1):** `hybrid`, public spellF1 0.550 [0.42, 0.67], macroF1 0.654
(SHA-256 `217a35e2…`). The sections below describe v1 and its ablations; the v3 additions and
the comparison table are in reports/teammate1_handover.md and reports/experiments.md.

## Architecture

```
ATD txt + XML ─┐
homography ────┼─> frame-aligned dataset (data/processed/frames.parquet, 9.37M rows)
Smart Tagging ─┘        └─> 42 causal frame features (frame_features.parquet)
offline HMM teacher ──> pseudo-labels H/A/D (+X contested) ──> LightGBM (spell-grouped CV)
                                                             └─> P(H,A,D,X) per frame
Viterbi: base priors × smart_tagging_modifier(event) per frame ──> rule layer ──> submission.csv
```

## Verified data facts (some differ from the original brief)

| fact | evidence |
|---|---|
| Metres = normalised × (105.3, 68.0) | the 1st-half kickoff ball lands at (53.1, 32.7), 1.4 m from the centre spot |
| **The homography maps to image pixels, not decimetres.** `HOMOGRAPHY_SCALE_TO_METERS = 0.1` fails the landmark check | the scaled kickoff projects to (97.4, 45.9), 46 m off; pitch corners project to a pixel trapezoid inside 1920×1080 |
| Player speed column is m/s, despite the XML saying kph | ratio to position-derived speed 1.00, r = 0.95 |
| Ball speed column is unusable | r = −0.03 with movement; ball speed is derived from positions |
| Slots 1–24 are H (GK 24), slots 25–46 are A (GK 46) | ATD XML |

## Feature importance (gain, top 10)

`speed_all_50`, `margin_50`, `ctrl_H_75`, `speed_all`, `cx_H_att`, `cx_A_att`, `tag_code`,
`since_ctrl_A`, `margin`, `ball_oob`. Trailing player speed (a stoppage cue) and trailing
distance margin to the ball (a control cue) dominate.

## Feature-group ablation (each group dropped and retrained, hybrid decoder)

| dropped | public spellF1 | Δ vs full |
|---|---|---|
| none (full) | 0.550 | — |
| temporal | 0.242 | **−0.308** |
| uncertainty | 0.436 | −0.114 |
| kinematics | 0.488 | −0.062 |
| team | 0.522 | −0.028 |
| tagging | 0.579 | +0.029 |

Only temporal is clearly outside the noise. Tagging features are kept: the change is inside the
0.04 noise band, and Hard Rule 5 requires using Smart Tagging. The decoder uses the tags in
either case.

## Decoder comparison and rule ablation

| variant | public spellF1 | macroF1 | public / full-match possessions |
|---|---|---|---|
| classical (argmax) | 0.159 | 0.598 | 101 / 871 |
| classical + ±50 majority vote | 0.420 | 0.607 | 23 / 252 |
| Viterbi, uniform transitions | 0.528 | 0.624 | 30 / 364 |
| Viterbi + Smart Tagging modifiers | 0.528 | 0.624 | 30 / 369 |
| **hybrid (Viterbi + 4 rules)** | **0.550** | 0.654 | 23 / 350 |
| hybrid without dead-ball override | 0.554 | 0.631 | 22 / 341 |
| hybrid without pass-flight | 0.548 | 0.647 | 29 / 365 |
| hybrid without invisible fallback | 0.524 | 0.654 | 25 / 361 |
| hybrid without contested→X | 0.550 | 0.655 | 23 / 350 |
| hybrid + 1 s minimum spell | 0.564 | 0.663 | 22 / 291 |

What this shows:
- Viterbi provides the jump (0.16 → 0.53).
- The transition modifiers change the public window by nothing measurable.
- Of the rules, the invisible fallback helps most (+0.026). Pass-flight halves spurious public
  possessions (29 → 23).
- Contested→X cannot change spellF1, because the scorer forward-fills `X`. It costs about 0.001 macroF1.
- The minimum spell length is 1 frame because the metric enforces none (brief Step 3). The 1 s
  variant is +0.014, inside noise, but has the more plausible full-match median (12.2 s vs 9.1 s).
  **Member A should decide.**

## Failure cases (public window, frozen model)

- **Dead ball is the weak class:** recall 0.23. 838 true-D frames are predicted A, 296 predicted H.
  The causes: pseudo-label D comes from a crude teacher, and stoppages without a set-piece tag
  or an off-pitch ball (fouls, injuries) leave little signal.
- **11 of 20 GT transitions are missed within 1 s:** 6 H↔A (all with the ball visible) and
  5 dead↔alive.
- **Over-segmentation on the full match:** 350 possessions with a median of 9 s, against a
  public-window mean of 14.2 s.
- **Home share is under-predicted:** 0.51 against 0.57 on the public window.

## Limitations of the evaluation

There are 17 public possessions. One possession moves spellF1 by about 0.06, and every comparison
above except "classical vs decoded" and the temporal ablation sits inside overlapping bootstrap
intervals. Choosing between variants on this window carries the risk EVALUATION.md demonstrates.
