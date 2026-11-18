"""Shared evaluation utilities for PathSnap map-matching models.

Provides:
  - per-fold metric computation
  - overall metric aggregation
  - Top-1 candidate accuracy (ground-truth arc matched = positive label,
    prediction on the highest-scored candidate per observation)
  - confusion matrix and classification report saving
  - predictions CSV saving
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    precision_score,
    recall_score,
    roc_auc_score,
)

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    _MPL = True
except ImportError:
    _MPL = False


def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray | None = None) -> dict:
    """Return a flat dict of classification metrics."""
    metrics = {
        "accuracy":    float(accuracy_score(y_true, y_pred)),
        "precision":   float(precision_score(y_true, y_pred, zero_division=0)),
        "recall":      float(recall_score(y_true, y_pred, zero_division=0)),
        "f1":          float(f1_score(y_true, y_pred, zero_division=0)),
        "macro_f1":    float(f1_score(y_true, y_pred, average="macro", zero_division=0)),
        "precision_macro": float(precision_score(y_true, y_pred, average="macro", zero_division=0)),
        "recall_macro":    float(recall_score(y_true, y_pred, average="macro", zero_division=0)),
    }
    if y_prob is not None:
        try:
            metrics["roc_auc"] = float(roc_auc_score(y_true, y_prob))
        except ValueError:
            metrics["roc_auc"] = float("nan")
        try:
            metrics["avg_precision"] = float(average_precision_score(y_true, y_prob))
        except ValueError:
            metrics["avg_precision"] = float("nan")
        try:
            metrics["log_loss"] = float(log_loss(y_true, y_prob))
        except ValueError:
            metrics["log_loss"] = float("nan")
    return metrics


def top1_candidate_accuracy(df: pd.DataFrame,
                             y_pred: np.ndarray,
                             y_prob: np.ndarray | None = None) -> float:
    """Top-1 candidate accuracy.

    For each (trip_id, observation_index) group that has at least one labeled
    positive candidate, pick the candidate with the highest predicted
    probability (or predicted label=1 if no probabilities).  Return the
    fraction of observations where the top-scored candidate has label=1.

    Only observations where any candidate has label=1 are counted.
    """
    work = df[["trip_id", "observation_index", "label", "candidate_rank"]].copy()
    work["y_pred"] = y_pred
    work["y_prob"] = y_prob if y_prob is not None else y_pred.astype(float)

    # Only consider observations that have a positive candidate
    has_pos = work.groupby(["trip_id", "observation_index"])["label"].max()
    has_pos = has_pos[has_pos == 1].index
    work = work.set_index(["trip_id", "observation_index"])
    work = work[work.index.isin(has_pos)].reset_index()

    if work.empty:
        return float("nan")

    # Pick highest-probability candidate per observation
    top = (work.sort_values("y_prob", ascending=False)
               .groupby(["trip_id", "observation_index"])
               .first()
               .reset_index())

    return float((top["y_pred"] == 1).mean())


def save_fold_metrics(fold_metrics: list[dict], output_dir: Path,
                      prefix: str = "") -> None:
    path = output_dir / f"{prefix}fold_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(fold_metrics, f, indent=2)
    print(f"Fold metrics -> {path}")


def save_overall_metrics(metrics: dict, output_dir: Path,
                         prefix: str = "") -> None:
    path = output_dir / f"{prefix}overall_metrics.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    print(f"Overall metrics -> {path}")


def save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          output_dir: Path, prefix: str = "") -> None:
    cm = confusion_matrix(y_true, y_pred)
    cm_dict = {"matrix": cm.tolist(),
               "labels": [0, 1],
               "tn": int(cm[0, 0]), "fp": int(cm[0, 1]),
               "fn": int(cm[1, 0]), "tp": int(cm[1, 1])}
    path_json = output_dir / f"{prefix}confusion_matrix.json"
    with open(path_json, "w", encoding="utf-8") as f:
        json.dump(cm_dict, f, indent=2)

    if _MPL:
        fig, ax = plt.subplots(figsize=(4, 3))
        im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
        ax.figure.colorbar(im, ax=ax)
        ax.set(xticks=[0, 1], yticks=[0, 1],
               xticklabels=["Pred 0", "Pred 1"],
               yticklabels=["True 0", "True 1"],
               title=f"{prefix.rstrip('_')} confusion matrix")
        for i in range(2):
            for j in range(2):
                ax.text(j, i, str(cm[i, j]),
                        ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        fig.tight_layout()
        path_png = output_dir / f"{prefix}confusion_matrix.png"
        fig.savefig(path_png, dpi=120)
        plt.close(fig)
        print(f"Confusion matrix -> {path_json}, {path_png}")
    else:
        print(f"Confusion matrix -> {path_json}")


def save_classification_report(y_true: np.ndarray, y_pred: np.ndarray,
                                output_dir: Path, prefix: str = "") -> None:
    report = classification_report(y_true, y_pred,
                                   target_names=["negative", "positive"],
                                   digits=4)
    path = output_dir / f"{prefix}classification_report.txt"
    path.write_text(report, encoding="utf-8")
    print(f"Classification report -> {path}")
    print(report)


def save_predictions(df: pd.DataFrame, y_pred: np.ndarray,
                     y_prob: np.ndarray | None,
                     output_dir: Path, prefix: str = "") -> None:
    out = df[["partition", "trip_id", "observation_index", "arc_id",
              "label"]].copy()
    out["predicted_label"] = y_pred
    if y_prob is not None:
        out["predicted_prob"] = y_prob
    path = output_dir / f"{prefix}predictions.csv"
    out.to_csv(path, index=False)
    print(f"Predictions -> {path}  ({len(out):,} rows)")


def aggregate_fold_metrics(fold_metrics: list[dict]) -> dict:
    """Mean ± std across folds for every numeric metric."""
    keys = [k for k in fold_metrics[0] if k != "fold"]
    result = {}
    for k in keys:
        vals = [m[k] for m in fold_metrics if isinstance(m.get(k), (int, float))]
        if vals:
            result[f"{k}_mean"] = float(np.mean(vals))
            result[f"{k}_std"]  = float(np.std(vals))
    return result
