# Where the baselines lose spell continuity

Public window (frames 11185–18684): 20 GT transitions, 17 possessions. Produced by
`python pipeline/spell_report.py evaluation/results/baseline_*.csv`. A transition counts as
*detected* if the prediction has a transition to the same label within ±1 s. A *spurious*
transition is a predicted flip with no GT transition within ±1 s. Abstentions are filled exactly as
the scorer fills them.

## Summary

| baseline | public spellF1 | GT transitions detected (of 20) | H↔A | alive→D | D→alive | spurious flips |
|---|---|---|---|---|---|---|
| raw | 0.038 | 11 | 8/12 | 1/4 | 2/4 | **113** |
| smooth25 | 0.287 | 6 | 5/12 | 1/4 | 0/4 | 31 |
| smooth50 | 0.326 | 2 | 2/12 | 0/4 | 0/4 | 24 |
| smooth125 | 0.424 | 3 | 3/12 | 0/4 | 0/4 | 12 |

Causes flagged at the spurious flips:

| baseline | ball invisible | ball in fast flight (>12 m/s, previous 1 s) | contested (both teams ≤2 m) |
|---|---|---|---|
| raw | 4% | **55%** | 11% |
| smooth25 | 23% | 35% | 13% |
| smooth50 | 38% | 38% | 17% |
| smooth125 | **42%** | 8% | 0% |

## Findings

1. **Pass flight is the main failure of the raw nearest-player rule.** 55% of its 113 spurious
   flips happen while the ball is in fast flight. The ball passes near an opponent mid-pass and the
   label flips. Smoothing hides this, but it does not solve it.
2. **Smoothing trades false flips for lag.** As the window widens, spurious flips fall
   (113 → 12), but true transitions land later than 1 s (11 → 3 detected). spellF1 averages IoU
   from 0.3 to 0.7, and being 1 s late already costs about 0.35. The baselines therefore cap out
   low. No single window fixes both problems, which is why a state model is needed.
3. **Dead-ball transitions are essentially never caught.** alive→D and D→alive are detected 0–2
   times out of 8 by the smoothed baselines. The baseline's only dead-ball signal is "ball not
   tracked", and the ball is **visible** around most real stoppages (mean visibility 0.78–0.86 in
   ±1 s). It needs Smart Tagging set pieces, off-pitch ball, and players stopping.
4. **Ball invisibility hurts through D, not through H↔A.** True H↔A hand-overs happen with the ball
   visible (mean 0.98 visibility in ±1 s). Invisibility mostly sits *inside* possessions. The
   baseline writes `D` when the ball is lost, which splits one possession into two spells. With
   wider windows, that becomes the dominant cause of spurious flips (42% for smooth125). The right
   behaviour is to hold the current state, or emit `X`, which the scorer forward-fills.
5. **Tackles and duels (contested ball) cause a minority of flips** (0–17%).

## What this implies for our design

- A Viterbi decoder with switching costs targets finding 2. It doesn't flip on short evidence and
  it doesn't lag the way a centred window does.
- Pass-flight logic targets finding 1: a fast ball with no player in reach gives no team evidence.
- A dead-ball model from Smart Tagging, off-pitch ball and player speed targets finding 3.
- An invisibility fallback that holds the state, never writing `D`, targets finding 4.
