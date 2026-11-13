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
