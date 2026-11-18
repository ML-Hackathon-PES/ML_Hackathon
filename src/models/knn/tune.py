"""Bayesian hyperparameter tuning for K-Nearest Neighbours.

Uses scikit-optimize BayesSearchCV with trip-grouped 5-fold CV.
Fits imputation + standard scaling inside each fold (no leakage).

Usage:
    python src/models/knn/tune.py
    python src/models/knn/tune.py --config config/knn.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.data_loader import load_config, load_cv_splits, load_dataset

try:
    from skopt import BayesSearchCV
    from skopt.space import Categorical, Integer, Real
    _SKOPT = True
except ImportError:
    _SKOPT = False


def build_param_space() -> dict:
    return {
        "knn__n_neighbors": Integer(3, 31),
        "knn__weights":     Categorical(["uniform", "distance"]),
        "knn__metric":      Categorical(["euclidean", "manhattan", "minkowski"]),
        "knn__p":           Integer(1, 3),
    }


def run_tuning(config_path: str | Path) -> dict:
    config     = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    data   = load_dataset(config, ROOT)
    X_train, y_train = data["X_train"], data["y_train"]
    groups = data["groups_train"]
    splits = load_cv_splits(config, ROOT, groups)

    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("knn",     KNeighborsClassifier()),
    ])

    n_calls = config["tuning"].get("n_calls", 32)

    if not _SKOPT:
        print("skopt not installed; using default params.")
        best_params = {"knn__n_neighbors": 7, "knn__weights": "distance",
                       "knn__metric": "euclidean", "knn__p": 2}
    else:
        # KNN on 863K rows is slow; use a stratified subsample (20%) for tuning
        rng  = np.random.default_rng(config.get("random_state", 42))
        n    = len(X_train)
        samp = max(10_000, int(n * 0.20))
        idx  = rng.choice(n, size=samp, replace=False)
        X_s, y_s, g_s = X_train[idx], y_train[idx], groups[idx]

        # Rebuild splits for the subsample
        from models.data_loader import load_cv_splits as _lcv
        import pandas as pd
        fold_df   = pd.read_csv(ROOT / config["cv_folds"])
        sub_splits = []
        for fold_num in sorted(fold_df["fold"].unique()):
            fold_rows  = fold_df[fold_df["fold"] == fold_num]
            val_trips  = set(fold_rows.loc[fold_rows["cv_role"] == "validation", "trip_id"])
            train_trips = set(fold_rows.loc[fold_rows["cv_role"] == "training",  "trip_id"])
            tr_i = np.where(np.isin(g_s, list(train_trips)))[0]
            vl_i = np.where(np.isin(g_s, list(val_trips)))[0]
            if len(tr_i) == 0 or len(vl_i) == 0:
                continue
            sub_splits.append((tr_i, vl_i))

        print(f"Bayesian KNN tuning on {samp:,} rows, {n_calls} calls ...")
        t0 = time.perf_counter()

        search = BayesSearchCV(
            pipe,
            build_param_space(),
            n_iter=n_calls,
            cv=sub_splits,
            scoring="f1",
            n_jobs=1,
            random_state=config.get("random_state", 42),
            refit=False,
            verbose=1,
        )
        search.fit(X_s, y_s)
        elapsed = time.perf_counter() - t0
        print(f"KNN tuning done in {elapsed:.1f}s.  Best F1={search.best_score_:.4f}")
        best_params = dict(search.best_params_)

    path = output_dir / "best_hyperparams.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"Best hyperparams -> {path}")
    print(best_params)
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/knn.json")
    args = parser.parse_args()
    run_tuning(ROOT / args.config)


if __name__ == "__main__":
    main()
