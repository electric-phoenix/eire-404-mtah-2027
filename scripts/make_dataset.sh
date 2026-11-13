#!/usr/bin/env sh
# Raw data -> data/processed/frames.parquet (+ frame_features.parquet). One command.
set -e
cd "$(dirname "$0")/.."
PY=${PY:-.venv/Scripts/python}
[ -x "$PY" ] || PY=.venv/bin/python
"$PY" -m possession.atd --rebuild
"$PY" -m pipeline.build_dataset
