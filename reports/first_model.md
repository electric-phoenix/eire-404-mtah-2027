# First model: frame-level LightGBM (brief Step 2)

**Public spellF1 0.159 [90% CI 0.12, 0.21], macroF1 0.598.** It does not beat the published floor
on its own (smooth50: 0.326 on this window). That is the expected result for an undecoded frame
classifier. It becomes the input to the Step 3 decoder, which brings it to 0.550.

## Set-up

- **Labels:** algorithmic pseudo-labels only (Hard Rule 1). The teacher is the offline hand-built HMM
  (`possession/model.py`, H/A/D), plus `X` on contested frames (`pipeline/pseudo_labels.py`).
  Training uses the 151,357 graded frames outside the public window: H 76,860 · A 41,189 · D 32,494 · X 814.
- **Features:** 42 causal features in 5 groups (`pipeline/features.py`). A pytest checks that no
  feature at frame t depends on frames after t.
- **Model:** LightGBM multiclass with `num_class = 4`, 49 rounds (median early-stopping point across folds).
- **Cross-validation:** 5-fold `GroupKFold`, grouped by pseudo-label spell, so no spell is split
  across train and validation. Log-loss 0.485 ± 0.136; accuracy against the pseudo-labels 0.80
  (folds range 0.69–0.86).

## Public-window confusion (rows = GT, columns = prediction)

| | H | A | D | X |
|---|---|---|---|---|
| **H** | 3069 | 341 | 0 | 44 |
| **A** | 633 | 1922 | 22 | 0 |
| **D** | 782 | 432 | 255 | 0 |

Per-class F1: H 0.77 · A 0.73 · **D 0.29** (precision 0.92, recall 0.17).

## Where spells break

- **101 predicted possessions against 17 true ones**, mean 2.9 s against 14.2 s. Per-frame
  argmax flickers.
- **85 spurious flips:** 38% happen in fast ball flight, 33% with the ball invisible, 7% contested.
- **D is rarely predicted.** Most true dead-ball frames come out as H or A, so stoppages don't
  split spells and dead→alive restarts are missed (1 of 4 detected within 1 s).
