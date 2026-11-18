"""Bayesian hyperparameter tuning for Logistic Regression.

Uses scikit-optimize (skopt) BayesSearchCV with a trip-grouped 5-fold CV.
Fits imputation + standard scaling inside each fold (no leakage).

Usage:
    python src/models/logistic_regression/tune.py
    python src/models/logistic_regression/tune.py --config config/logistic_regression.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
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
        "lr__C":            Real(1e-4, 1e2, prior="log-uniform"),
        "lr__max_iter":     Integer(300, 2000),
        "lr__class_weight": Categorical(["balanced", None]),
        "lr__l1_ratio":     Real(0.0, 1.0),
    }


def run_tuning(config_path: str | Path) -> dict:
    config = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    data   = load_dataset(config, ROOT)
    X_train, y_train = data["X_train"], data["y_train"]
    groups = data["groups_train"]
    splits = load_cv_splits(config, ROOT, groups)

    # Pipeline: impute NaN → scale → LR with saga+elasticnet (supports l1/l2/elasticnet)
    pipe = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr",      LogisticRegression(
            solver="saga",
            l1_ratio=0.5,
            random_state=config.get("random_state", 42),
            max_iter=2000,
        )),
    ])

    n_calls = config["tuning"].get("n_calls", 40)

    if not _SKOPT:
        print("skopt not installed; falling back to default params.")
        best_params = {"lr__C": 1.0, "lr__class_weight": "balanced",
                       "lr__l1_ratio": 0.5, "lr__max_iter": 1000}
    else:
        print(f"Starting Bayesian search ({n_calls} calls) ...")
        t0 = time.perf_counter()

        search = BayesSearchCV(
            pipe,
            build_param_space(),
            n_iter=n_calls,
            cv=splits,
            scoring="f1",
            n_jobs=1,
            random_state=config.get("random_state", 42),
            refit=False,
            verbose=1,
        )
        search.fit(X_train, y_train)
        elapsed = time.perf_counter() - t0
        print(f"Tuning done in {elapsed:.1f}s.  Best F1={search.best_score_:.4f}")
        best_params = dict(search.best_params_)

    best_params_path = output_dir / "best_hyperparams.json"
    with open(best_params_path, "w", encoding="utf-8") as f:
        json.dump(best_params, f, indent=2, default=str)
    print(f"Best hyperparams -> {best_params_path}")
    print(best_params)
    return best_params


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/logistic_regression.json")
    args = parser.parse_args()
    run_tuning(ROOT / args.config)


if __name__ == "__main__":
    main()
