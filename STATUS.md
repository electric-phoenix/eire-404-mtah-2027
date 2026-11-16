# STATUS — 2026-11-10 (Day 1 of 7, deadline 17 Nov 2026)

**Frozen 16 Nov 2026: `hybrid_stoppage_soft-midpoint-edges-min25-whistle` (v3 + whistle), public
spellF1 0.600 [90% CI 0.48, 0.71], macroF1 0.793.** v3 plus the referee whistle from the tactical
video's sound as soft dead-ball evidence. Public dead-ball recall 0.37 -> 0.56 at precision 0.95;
whole match 289 possessions, median 10.2 s, 106 stoppages. Placebo (time-shifted whistle) loses
the gain. `Eire404.csv` = `submission.csv`, validated, regenerates byte-identically from scratch
(SHA-256 `fea816bf…`). Needs ffmpeg (set FFMPEG if not on PATH). Replaces v3 (0.584, `9b1f1f94…`).
Dropped on 16 Nov (all logged in reports/experiments.md): foul restart, sound features in the
model, self-training, kick sounds, tracking-based touch / confirmed control.

## Run

```
python -m venv .venv && .venv/Scripts/python -m pip install -r requirements-lock.txt
.venv/Scripts/python -m pipeline                 # raw data -> submission.csv, ~1-2 min, needs ffmpeg
./scripts/make_dataset.sh                        # dataset only -> data/processed/frames.parquet
.venv/Scripts/python scripts/board.py            # public board + full-match possession counts
.venv/Scripts/python -m pytest -q tests          # 20 tests: fixtures + no-leakage
```

## On disk

All assets are present and complete. The homography is **not** truncated (199,319 rows, all valid).
Hardware: Ryzen 7 3700X (16 threads), 32 GB RAM, RTX 2060 Super, Windows 11, Python 3.14.

## Deliverables checklist

- [x] `data/processed/frames.parquet`: 9.37M rows, one per (frame, player_id), with all flags
- [x] `pipeline/`: one command from raw data to `submission.csv` (`python -m pipeline`)
- [x] `reports/baseline_failure.md`
- [x] `reports/experiments.md`
- [x] `reports/model.md`: architecture, importance, ablations, failure cases
- [x] `reports/first_model.md`
- [x] `submission.csv`, validated. **Awaiting A's review** (Hard Rule 7).
- [x] pytest: builder fixture test and leakage test (20 pass)

## Brief vs reality (decisions A should check)

1. **The homography maps to image pixels.** The brief's decimetre hypothesis fails its own landmark
   check (46 m off). Metres come from normalised × (105.3, 68.0) instead (1.4 m from the centre spot).
2. **The speed column is m/s**, although the XML says kph. The ball speed column is unusable;
   ball speed is derived from positions.
3. **`io/` was renamed to `mtah_io/`.** Python's built-in `io` module makes `import io.atd` impossible.
4. **No minimum spell length.** The metric enforces none, so it is 1 frame. A 1 s minimum scores
   +0.014 (inside noise) with more plausible spell lengths. **A to decide.**
5. **Training labels are algorithmic pseudo-labels.** Hard Rule 1 forbids training on the public
   GT. Spell-level CV runs on those pseudo-labels; the public window is used for scoring only.
6. **Organiser T&C:** the team chose to proceed despite the FAQ clause on cloud LLMs. A should be
   aware.

## Next (Day 2–4, Member C)

1. **Dead ball** (recall 0.23): a better teacher for D (foul/injury stoppages with no tag), and a
   dwell prior for D.
2. **Over-segmentation** (350 full-match possessions, median 9 s): tune dwell priors on
   full-match plausibility.
3. Team inference for throw-in restarts (the tag says `Team = N/A`).
4. Bootstrap every comparison. Most current gaps are inside the noise band.

## Needs from humans

- [ ] A: review `pipeline/decode.py` and `pipeline/train.py`, run the validator, approve or reject the freeze.
- [ ] A: decide on the 1 s minimum spell.
- [ ] Team name (sets the CSV and document filenames) and a new **public** repo without `data/`.
