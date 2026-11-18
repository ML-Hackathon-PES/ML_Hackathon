"""Train K-Nearest Neighbours for PathSnap map-matching.

Workflow:
  1. Load best hyperparams from tune.py output (or use defaults).
  2. 5-fold trip-grouped CV; record per-fold metrics + Top-1 accuracy.
  3. Fit final model on all 80 training trips.
  4. Save model, fold metrics, overall metrics, confusion matrix,
     classification report, training time.

Usage:
    python src/models/knn/train.py
    python src/models/knn/train.py --config config/knn.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.data_loader import load_config, load_cv_splits, load_dataset
from models.evaluation import (
    aggregate_fold_metrics,
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_fold_metrics,
    save_overall_metrics,
    top1_candidate_accuracy,
)


def build_pipeline(params: dict) -> Pipeline:
    n_neighbors = int(params.get("knn__n_neighbors", 7))
    weights     = params.get("knn__weights", "distance")
    metric      = params.get("knn__metric", "euclidean")
    p           = int(params.get("knn__p", 2))
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("knn", KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            metric=metric,
            p=p,
        )),
    ])


def run_training(config_path: str | Path) -> None:
    config     = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    hp_path = output_dir / "best_hyperparams.json"
    if hp_path.exists():
        with open(hp_path, encoding="utf-8") as f:
            best_params = json.load(f)
        print(f"Loaded hyperparams from {hp_path}")
    else:
        best_params = {}
        print("No tuning output found; using default hyperparams.")

    data       = load_dataset(config, ROOT)
    X_train    = data["X_train"]
    y_train    = data["y_train"]
    groups     = data["groups_train"]
    df_train   = data["df_train"]
    splits     = load_cv_splits(config, ROOT, groups)
    feat_names = data["feature_names"]

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n=== 5-Fold Cross-Validation (KNN) ===")
    fold_metrics = []
    fold_top1    = []
    cv_t0        = time.perf_counter()

    for fold_idx, (tr_idx, vl_idx) in enumerate(splits, 1):
        print(f"\n--- Fold {fold_idx} ---")
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

        pipe = build_pipeline(best_params)

        t_fit = time.perf_counter()
        pipe.fit(X_tr, y_tr)
        fit_secs = time.perf_counter() - t_fit

        t_inf = time.perf_counter()
        y_pred = pipe.predict(X_vl)
        y_prob = pipe.predict_proba(X_vl)[:, 1]
        inf_secs = time.perf_counter() - t_inf

        m = compute_metrics(y_vl, y_pred, y_prob)
        m["fold"]             = fold_idx
        m["fit_time_seconds"] = fit_secs
        m["inf_time_seconds"] = inf_secs

        df_vl = df_train.iloc[vl_idx].reset_index(drop=True)
        t1 = top1_candidate_accuracy(df_vl, y_pred, y_prob)
        m["top1_candidate_accuracy"] = t1
        fold_top1.append(t1)

        fold_metrics.append(m)
        print(f"  F1={m['f1']:.4f}  MacroF1={m['macro_f1']:.4f}"
              f"  AUC={m.get('roc_auc', float('nan')):.4f}"
              f"  LogLoss={m.get('log_loss', float('nan')):.4f}"
              f"  Top-1={t1:.4f}  fit={fit_secs:.1f}s  inf={inf_secs:.1f}s")

    cv_elapsed = time.perf_counter() - cv_t0
    print(f"\nCV done in {cv_elapsed:.1f}s")

    overall_cv = aggregate_fold_metrics(fold_metrics)
    overall_cv["cv_time_seconds"] = cv_elapsed
    overall_cv["top1_mean"] = float(np.mean(fold_top1))
    overall_cv["top1_std"]  = float(np.std(fold_top1))

    save_fold_metrics(fold_metrics, output_dir, prefix="knn_")
    save_overall_metrics(overall_cv, output_dir, prefix="knn_cv_")

    # ── Final model on all 80 training trips ──────────────────────────────────
    print("\n=== Fitting final KNN model on all 80 training trips ===")
    final_t0   = time.perf_counter()
    final_pipe = build_pipeline(best_params)
    final_pipe.fit(X_train, y_train)
    final_elapsed = time.perf_counter() - final_t0
    print(f"Final fit done in {final_elapsed:.1f}s")

    y_pred_all = final_pipe.predict(X_train)
    y_prob_all = final_pipe.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_pred_all, y_prob_all)
    train_metrics["fit_time_seconds"] = final_elapsed
    train_metrics["feature_count"]    = len(feat_names)
    train_metrics["training_rows"]    = int(X_train.shape[0])

    save_overall_metrics(train_metrics, output_dir, prefix="knn_train_")
    save_confusion_matrix(y_train, y_pred_all, output_dir, prefix="knn_train_")
    save_classification_report(y_train, y_pred_all, output_dir, prefix="knn_train_")

    model_path = output_dir / "knn_model.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"Model saved -> {model_path}")

    feat_path = output_dir / "knn_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feat_names, f, indent=2)

    timing = {
        "cv_time_seconds":   cv_elapsed,
        "final_fit_seconds": final_elapsed,
        "total_seconds":     cv_elapsed + final_elapsed,
    }
    with open(output_dir / "knn_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print("\n=== KNN Training Complete ===")
    print(f"  CV F1 (mean±std): {overall_cv['f1_mean']:.4f} ± {overall_cv['f1_std']:.4f}")
    print(f"  CV Top-1 acc:     {overall_cv['top1_mean']:.4f} ± {overall_cv['top1_std']:.4f}")
    print(f"  Outputs in:       {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/knn.json")
    args = parser.parse_args()
    run_training(ROOT / args.config)


if __name__ == "__main__":
    main()
