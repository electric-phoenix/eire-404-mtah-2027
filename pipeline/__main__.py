"""One command: raw data -> validated submission.csv (frozen configuration).

    python -m pipeline [--out submission.csv]

1. parse ATD + build frame-aligned dataset (data/processed/*.parquet)
2. pseudo-label (offline HMM teacher) and train LightGBM with spell-level CV
3. 4-state Viterbi with Smart Tagging modifiers + rule layer
4. validate with the official checks
"""
import argparse
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FROZEN = {"features": "full", "variant": "hybrid"}


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
    step("-m", "pipeline.run", "--proba", FROZEN["features"], "--variants", FROZEN["variant"],
         "--out", out)
    step(os.path.join("evaluation", "validate_submission.py"), out)


if __name__ == "__main__":
    main()
