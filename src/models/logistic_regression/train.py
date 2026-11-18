"""Train Logistic Regression for PathSnap map-matching.

Workflow:
  1. Load best hyperparams from tune.py output (or use defaults if not found).
  2. Run 5-fold trip-grouped cross-validation; record per-fold metrics.
  3. Fit a final model on all 80 training trips.
  4. Save model, fold metrics, overall metrics, confusion matrix,
     classification report, training time.

Usage:
    python src/models/logistic_regression/train.py
    python src/models/logistic_regression/train.py --config config/logistic_regression.json
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
from sklearn.linear_model import LogisticRegression
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


def build_pipeline(params: dict, random_state: int) -> Pipeline:
    C            = float(params.get("lr__C", 1.0))
    max_iter     = int(params.get("lr__max_iter", 1000))
    class_weight = params.get("lr__class_weight", "balanced")
    l1_ratio     = float(params.get("lr__l1_ratio", 0.5))
    if class_weight == "None" or class_weight is None:
        class_weight = None
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler",  StandardScaler()),
        ("lr", LogisticRegression(
            C=C,
            solver="saga",
            l1_ratio=l1_ratio,
            class_weight=class_weight,
            max_iter=max_iter,
            random_state=random_state,
        )),
    ])


def run_training(config_path: str | Path) -> None:
    config     = load_config(config_path)
    output_dir = ROOT / config["output_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)
    rng        = config.get("random_state", 42)

    # Load best hyperparams if tuning was run
    hp_path = output_dir / "best_hyperparams.json"
    if hp_path.exists():
        with open(hp_path, encoding="utf-8") as f:
            best_params = json.load(f)
        print(f"Loaded hyperparams from {hp_path}")
    else:
        best_params = {}
        print("No tuning output found; using default hyperparams.")

    data        = load_dataset(config, ROOT)
    X_train     = data["X_train"]
    y_train     = data["y_train"]
    groups      = data["groups_train"]
    df_train    = data["df_train"]
    splits      = load_cv_splits(config, ROOT, groups)
    feat_names  = data["feature_names"]

    # ── Cross-validation ──────────────────────────────────────────────────────
    print("\n=== 5-Fold Cross-Validation ===")
    fold_metrics = []
    fold_top1    = []
    cv_t0        = time.perf_counter()

    for fold_idx, (tr_idx, vl_idx) in enumerate(splits, 1):
        print(f"\n--- Fold {fold_idx} ---")
        X_tr, y_tr = X_train[tr_idx], y_train[tr_idx]
        X_vl, y_vl = X_train[vl_idx], y_train[vl_idx]

        pipe = build_pipeline(best_params, rng)
        pipe.fit(X_tr, y_tr)

        y_pred = pipe.predict(X_vl)
        y_prob = pipe.predict_proba(X_vl)[:, 1]

        m = compute_metrics(y_vl, y_pred, y_prob)
        m["fold"] = fold_idx

        df_vl = df_train.iloc[vl_idx].reset_index(drop=True)
        t1 = top1_candidate_accuracy(df_vl, y_pred, y_prob)
        m["top1_candidate_accuracy"] = t1
        fold_top1.append(t1)

        fold_metrics.append(m)
        print(f"  F1={m['f1']:.4f}  MacroF1={m['macro_f1']:.4f}"
              f"  AUC={m.get('roc_auc', float('nan')):.4f}"
              f"  LogLoss={m.get('log_loss', float('nan')):.4f}"
              f"  Top-1={t1:.4f}")

    cv_elapsed = time.perf_counter() - cv_t0
    print(f"\nCV done in {cv_elapsed:.1f}s")

    overall_cv = aggregate_fold_metrics(fold_metrics)
    overall_cv["cv_time_seconds"] = cv_elapsed
    overall_cv["top1_mean"] = float(np.mean(fold_top1))
    overall_cv["top1_std"]  = float(np.std(fold_top1))

    save_fold_metrics(fold_metrics, output_dir, prefix="lr_")
    save_overall_metrics(overall_cv, output_dir, prefix="lr_cv_")

    # ── Final model on all 80 training trips ──────────────────────────────────
    print("\n=== Fitting final model on all 80 training trips ===")
    final_t0   = time.perf_counter()
    final_pipe = build_pipeline(best_params, rng)
    final_pipe.fit(X_train, y_train)
    final_elapsed = time.perf_counter() - final_t0
    print(f"Final fit done in {final_elapsed:.1f}s")

    # In-sample evaluation on full training set
    y_pred_all = final_pipe.predict(X_train)
    y_prob_all = final_pipe.predict_proba(X_train)[:, 1]
    train_metrics = compute_metrics(y_train, y_pred_all, y_prob_all)
    train_metrics["fit_time_seconds"]   = final_elapsed
    train_metrics["feature_count"]      = len(feat_names)
    train_metrics["training_rows"]      = int(X_train.shape[0])

    save_overall_metrics(train_metrics, output_dir, prefix="lr_train_")
    save_confusion_matrix(y_train, y_pred_all, output_dir, prefix="lr_train_")
    save_classification_report(y_train, y_pred_all, output_dir, prefix="lr_train_")

    # Save model
    model_path = output_dir / "lr_model.joblib"
    joblib.dump(final_pipe, model_path)
    print(f"Model saved -> {model_path}")

    # Save feature list
    feat_path = output_dir / "lr_features.json"
    with open(feat_path, "w", encoding="utf-8") as f:
        json.dump(feat_names, f, indent=2)

    # Training time summary
    timing = {
        "cv_time_seconds":    cv_elapsed,
        "final_fit_seconds":  final_elapsed,
        "total_seconds":      cv_elapsed + final_elapsed,
    }
    with open(output_dir / "lr_timing.json", "w", encoding="utf-8") as f:
        json.dump(timing, f, indent=2)

    print("\n=== Logistic Regression Training Complete ===")
    print(f"  CV F1 (mean±std): {overall_cv['f1_mean']:.4f} ± {overall_cv['f1_std']:.4f}")
    print(f"  CV Top-1 acc:     {overall_cv['top1_mean']:.4f} ± {overall_cv['top1_std']:.4f}")
    print(f"  Outputs in:       {output_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/logistic_regression.json")
    args = parser.parse_args()
    run_training(ROOT / args.config)


if __name__ == "__main__":
    main()
