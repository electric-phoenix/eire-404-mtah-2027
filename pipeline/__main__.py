"""One command: raw data -> validated submission.csv (frozen configuration).

    python -m pipeline [--out submission.csv]

1. parse ATD + build frame-aligned dataset (data/processed/*.parquet)
2. pseudo-label (offline HMM teacher) and train LightGBM with spell-level CV
3. sound: extract the tactical video's audio (ffmpeg) and score the referee's whistle per frame
4. dead-ball tree + referee whistle as soft evidence -> 4-state Viterbi -> rule layer ->
   stoppage clean-up (merge stoppages <3 s apart, drop untagged ones <2 s) -> 1 s minimum spell
5. validate with the official checks

ffmpeg must be installed (tested with 4.2.3). If it is not on PATH, set FFMPEG to its full path.
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# Frozen 16 Nov 2026: v3 + referee whistle (reports/experiments.md row 28).
# History: "hybrid" (v1, to 15 Nov), "hybrid_stoppage_soft-midpoint-edges-min25" (v3, 15-16 Nov).
FROZEN = {"features": "full", "variant": "hybrid_stoppage_soft-midpoint-edges-min25-whistle"}


def step(*args):
    print(">>", " ".join(args), flush=True)
    subprocess.run([sys.executable, *args], check=True, cwd=ROOT)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "submission.csv"))
    out = os.path.abspath(ap.parse_args().out)
    step("-m", "possession.atd", "--rebuild")
    step("-m", "pipeline.build_dataset")
    step("-m", "pipeline.train", "--tag", FROZEN["features"])
    step("-m", "pipeline.audio")
    step("-m", "pipeline.run", "--proba", FROZEN["features"], "--variants", FROZEN["variant"],
         "--out", out)
    step(os.path.join("evaluation", "validate_submission.py"), out)


if __name__ == "__main__":
    main()
