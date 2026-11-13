"""Frame-level LightGBM multiclass classifier (brief Step 2).

    python -m pipeline.train [--drop-group tagging]

Trains on algorithmic pseudo-labels (pipeline/pseudo_labels.py) over the graded frames OUTSIDE the
public window. Cross-validation is spell-level: GroupKFold with each pseudo-label run (a maximal
run of one label) as a group, so no spell is split across train/validation.
Writes data/processed/proba_<tag>.npy: (N, 4) P(H), P(A), P(D), P(X) for every frame.
"""
import argparse
import json
import os
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from pipeline import features as FE, pseudo_labels as PL  # noqa: E402

PROC = os.path.join(ROOT, "data", "processed")
PUBLIC = (11185, 18684)
SEED = 0
PARAMS = {
    "objective": "multiclass", "num_class": 4, "learning_rate": 0.05, "num_leaves": 63,
    "min_data_in_leaf": 200, "feature_fraction": 0.8, "bagging_fraction": 0.8, "bagging_freq": 1,
    "lambda_l2": 1.0, "seed": SEED, "deterministic": True, "num_threads": 8, "verbose": -1,
}
CATEGORICAL = ["tag_code", "sp_type", "sp_team"]


def runs(labels):
    change = np.r_[True, labels[1:] != labels[:-1]]
    return np.cumsum(change)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drop-group", action="append", default=[])
    ap.add_argument("--tag", default="full")
    args = ap.parse_args()

    df = pd.read_parquet(os.path.join(PROC, "frame_features.parquet"))
    feats = [c for g, cols in FE.GROUPS.items() if g not in args.drop_group for c in cols]
    y_str, graded = PL.build()
    y = pd.Series(y_str).map({c: i for i, c in enumerate(PL.CLASSES)}).to_numpy()
    groups = runs(y_str)

    idx = np.arange(len(df))
    train = graded & ~((idx >= PUBLIC[0]) & (idx <= PUBLIC[1]))
    X = df.loc[train, feats]
    yt, gt = y[train], groups[train]
    cats = [c for c in CATEGORICAL if c in feats]

    best_iters, fold_logloss, fold_acc = [], [], []
    for k, (tr, va) in enumerate(GroupKFold(n_splits=5).split(X, yt, gt)):
        dtr = lgb.Dataset(X.iloc[tr], yt[tr], categorical_feature=cats)
        dva = lgb.Dataset(X.iloc[va], yt[va], categorical_feature=cats, reference=dtr)
        m = lgb.train(PARAMS, dtr, 2000, valid_sets=[dva],
                      callbacks=[lgb.early_stopping(50, verbose=False)])
        p = m.predict(X.iloc[va], num_iteration=m.best_iteration)
        best_iters.append(m.best_iteration)
        fold_logloss.append(float(m.best_score["valid_0"]["multi_logloss"]))
        fold_acc.append(float((p.argmax(1) == yt[va]).mean()))
        print("fold {}  iters {:4d}  logloss {:.4f}  acc vs pseudo-labels {:.4f}".format(
            k, m.best_iteration, fold_logloss[-1], fold_acc[-1]))

    n_iter = int(np.median(best_iters))
    final = lgb.train(PARAMS, lgb.Dataset(X, yt, categorical_feature=cats), n_iter)
    proba = final.predict(df[feats], num_iteration=n_iter)
    np.save(os.path.join(PROC, "proba_{}.npy".format(args.tag)), proba.astype(np.float32))
    imp = pd.Series(final.feature_importance("gain"), index=feats).sort_values(ascending=False)
    summary = {"tag": args.tag, "dropped": args.drop_group, "n_train": int(train.sum()),
               "label_counts": {c: int((yt == i).sum()) for i, c in enumerate(PL.CLASSES)},
               "cv_logloss": fold_logloss, "cv_acc": fold_acc, "n_iter": n_iter,
               "importance_gain_top15": {k: float(v) for k, v in imp.head(15).items()}}
    with open(os.path.join(PROC, "train_{}.json".format(args.tag)), "w") as fh:
        json.dump(summary, fh, indent=1)
    print(json.dumps({k: summary[k] for k in ("n_train", "label_counts", "n_iter")}))
    print("CV logloss {:.4f} ± {:.4f}   acc {:.4f}".format(np.mean(fold_logloss), np.std(fold_logloss),
                                                            np.mean(fold_acc)))
    print("top features by gain:", ", ".join(imp.head(10).index))


if __name__ == "__main__":
    main()
