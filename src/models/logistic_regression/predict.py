"""Run the trained Logistic Regression model on student_test data.

Loads the saved Pipeline, runs inference, saves predictions and metrics.
Student_test data is NEVER used for training or tuning.

Usage:
    python src/models/logistic_regression/predict.py
    python src/models/logistic_regression/predict.py --config config/logistic_regression.json
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import joblib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

from models.data_loader import load_config, load_dataset
from models.evaluation import (
    compute_metrics,
    save_classification_report,
    save_confusion_matrix,
    save_overall_metrics,
    save_predictions,
    top1_candidate_accuracy,
)


def run_predict(config_path: str | Path) -> None:
    config     = load_config(config_path)
    output_dir = ROOT / config["output_dir"]

    model_path = output_dir / "lr_model.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"Trained model not found at {model_path}. Run train.py first.")

    print(f"Loading model from {model_path}")
    pipe = joblib.load(model_path)

    data        = load_dataset(config, ROOT)
    X_student   = data["X_student"]
    y_student   = data["y_student"]
    df_student  = data["df_student"]
    feat_names  = data["feature_names"]

    print(f"Running inference on {X_student.shape[0]:,} student_test rows ...")
    t0     = time.perf_counter()
    y_pred = pipe.predict(X_student)
    y_prob = pipe.predict_proba(X_student)[:, 1]
    elapsed = time.perf_counter() - t0
    print(f"Inference done in {elapsed:.3f}s  ({elapsed/len(X_student)*1000:.4f} ms/row)")

    metrics = compute_metrics(y_student, y_pred, y_prob)
    t1 = top1_candidate_accuracy(df_student, y_pred, y_prob)
    metrics["top1_candidate_accuracy"] = t1
    metrics["inference_time_seconds"]  = elapsed
    metrics["rows_evaluated"]          = int(len(X_student))

    save_overall_metrics(metrics, output_dir, prefix="lr_student_")
    save_confusion_matrix(y_student, y_pred, output_dir, prefix="lr_student_")
    save_classification_report(y_student, y_pred, output_dir, prefix="lr_student_")
    save_predictions(df_student, y_pred, y_prob, output_dir, prefix="lr_student_")

    print(f"\n=== Student-test results ===")
    print(f"  F1       : {metrics['f1']:.4f}")
    print(f"  ROC-AUC  : {metrics.get('roc_auc', float('nan')):.4f}")
    print(f"  Top-1 acc: {t1:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/logistic_regression.json")
    args = parser.parse_args()
    run_predict(ROOT / args.config)


if __name__ == "__main__":
    main()
