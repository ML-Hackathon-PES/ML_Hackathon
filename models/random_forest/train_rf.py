"""PathSnap - Model 4: Random Forest candidate classifier.

Reads the team's common tables, runs group-aware CV on the shared folds
(80 training trips), refits on all 80 trips, evaluates on the 10 student-test
trips, and saves metrics, plots, probabilities and the trained model.

Usage:
    python train_rf.py --features candidate_feature_table.csv \
        --labels candidate_labels.csv --folds cross_validation_folds.csv \
        --out rf_results
"""
import argparse
import json
import platform
import time
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import sklearn

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (ConfusionMatrixDisplay, accuracy_score, confusion_matrix,
                             f1_score, log_loss, precision_score, recall_score)
from sklearn.pipeline import Pipeline

SEED = 42
KEYS = ["partition", "trip_id", "observation_index", "arc_id"]
# Common team features. Identifiers and absolute timestamp are excluded on purpose.
FEATURES = [
    "gps_to_segment_distance_meters", "gps_to_segment_start_meters",
    "gps_to_segment_end_meters", "segment_length_meters", "candidate_rank",
    "candidate_count", "gps_speed_mps", "gps_movement_direction_degrees",
    "previous_gps_speed_mps", "previous_movement_direction_degrees",
    "previous_delta_time_seconds", "road_segment_direction_degrees",
    "gps_road_direction_difference_degrees", "delta_time_seconds", "speed_suspicious",
]
RF_PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=10,
                 max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=-1)


def make_model():
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fitted on training rows only
        ("rf", RandomForestClassifier(**RF_PARAMS)),
    ])


def load_data(features_path, labels_path):
    feats = pd.read_csv(features_path, dtype={"trip_id": str})
    labs = pd.read_csv(labels_path, dtype={"trip_id": str},
                       usecols=KEYS + ["label"])
    n_before = len(labs)
    labs = labs.dropna(subset=["label"])  # unaligned observations are unlabeled
    df = feats.merge(labs, on=KEYS, how="inner")
    df["label"] = df["label"].astype(int)
    print(f"label rows: {n_before}, labeled: {len(labs)}, after merge: {len(df)}")
    return df.reset_index(drop=True)


def top1_accuracy(df, prob):
    d = pd.DataFrame({"trip_id": df["trip_id"].values,
                      "obs": df["observation_index"].values,
                      "label": df["label"].values, "p": prob})
    best = d.loc[d.groupby(["trip_id", "obs"])["p"].idxmax()]
    return float(best["label"].mean())


def candidate_miss_rate(df):
    g = df.groupby(["trip_id", "observation_index"])["label"].max()
    return float((g == 0).mean())


def evaluate(df, prob):
    y = df["label"].values
    pred = (prob >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_positive": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(prob, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "top1_accuracy": top1_accuracy(df, prob),
        "candidate_miss_rate": candidate_miss_rate(df),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", required=True)
    ap.add_argument("--labels", required=True)
    ap.add_argument("--folds", required=True)
    ap.add_argument("--out", default="rf_results")
    args = ap.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.features, args.labels)
    train = df[df["partition"] == "train"].reset_index(drop=True)
    test = df[df["partition"] == "student_test"].reset_index(drop=True)
    print(f"train rows {len(train)} ({train.trip_id.nunique()} trips), "
          f"student_test rows {len(test)} ({test.trip_id.nunique()} trips)")
    print("positive rate (train):", round(train["label"].mean(), 4))

    # ---- group-aware CV on the team's shared folds (split by trip) ----
    folds = pd.read_csv(args.folds, dtype={"trip_id": str})
    cv_rows = []
    for f, g in folds.groupby("fold"):
        tr_trips = set(g.loc[g.cv_role == "training", "trip_id"])
        va_trips = set(g.loc[g.cv_role == "validation", "trip_id"])
        tr = train[train.trip_id.isin(tr_trips)]
        va = train[train.trip_id.isin(va_trips)]
        m = make_model()
        t0 = time.time()
        m.fit(tr[FEATURES], tr["label"])
        fit_s = time.time() - t0
        prob = m.predict_proba(va[FEATURES])[:, 1]
        res = evaluate(va, prob)
        res.update(fold=int(f), train_trips=len(tr_trips), val_trips=len(va_trips),
                   fit_seconds=fit_s)
        cv_rows.append(res)
        print(f"fold {f}: macroF1={res['macro_f1']:.4f} top1={res['top1_accuracy']:.4f} "
              f"logloss={res['log_loss']:.4f}")
    cv = pd.DataFrame(cv_rows)
    cv.to_csv(out / "cv_fold_metrics.csv", index=False)

    # ---- final fit on all 80 training trips ----
    model = make_model()
    t0 = time.time()
    model.fit(train[FEATURES], train["label"])
    train_seconds = time.time() - t0

    train_prob = model.predict_proba(train[FEATURES])[:, 1]
    train_metrics = evaluate(train, train_prob)

    t0 = time.time()
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    infer_seconds = time.time() - t0
    test_metrics = evaluate(test, test_prob)
    test_metrics["inference_seconds_total"] = infer_seconds
    test_metrics["inference_ms_per_1000_rows"] = 1000 * infer_seconds / len(test) * 1000

    summary = {
        "model": "RandomForestClassifier",
        "seed": SEED,
        "hyperparameters": RF_PARAMS,
        "features": FEATURES,
        "cv_mean": cv.drop(columns=["fold"]).mean().to_dict(),
        "cv_std": cv.drop(columns=["fold"]).std().to_dict(),
        "train_fit_seconds": train_seconds,
        "train_set_metrics": train_metrics,
        "student_test_metrics": test_metrics,
        "route_continuity": "N/A for classifier alone (needs HMM/Viterbi route)",
        "versions": {"python": platform.python_version(), "sklearn": sklearn.__version__,
                     "pandas": pd.__version__, "numpy": np.__version__},
    }
    (out / "metrics.json").write_text(json.dumps(summary, indent=2, default=float))
    (out / "model_config.json").write_text(json.dumps(
        {"seed": SEED, "hyperparameters": RF_PARAMS, "features": FEATURES}, indent=2))

    # probabilities for the HMM/Viterbi teammate
    probs = test[KEYS + ["label"]].copy()
    probs["rf_probability"] = test_prob
    probs.to_csv(out / "rf_student_test_probabilities.csv", index=False)

    joblib.dump(model, out / "rf_model.joblib", compress=3)

    # ---- plots (actual results only) ----
    cm = confusion_matrix(test["label"], (test_prob >= 0.5).astype(int), labels=[0, 1])
    ConfusionMatrixDisplay(cm, display_labels=["not true arc", "true arc"]).plot(
        cmap="Blues", values_format="d")
    plt.title("Random Forest - confusion matrix (student-test)")
    plt.tight_layout(); plt.savefig(out / "confusion_matrix.png", dpi=150); plt.close()

    plt.figure(figsize=(6, 4))
    plt.bar(cv["fold"].astype(str), cv["macro_f1"], color="#4c78a8")
    plt.axhline(cv["macro_f1"].mean(), color="k", ls="--", label="mean")
    plt.xlabel("CV fold"); plt.ylabel("Macro F1"); plt.title("Random Forest - CV macro F1")
    plt.legend(); plt.tight_layout(); plt.savefig(out / "cv_macro_f1.png", dpi=150); plt.close()

    imp = pd.Series(model.named_steps["rf"].feature_importances_, index=FEATURES).sort_values()
    plt.figure(figsize=(7, 5)); imp.plot.barh(color="#59a14f")
    plt.title("Random Forest - feature importance"); plt.tight_layout()
    plt.savefig(out / "feature_importance.png", dpi=150); plt.close()

    print("\nCV macro F1: %.4f +/- %.4f" % (cv.macro_f1.mean(), cv.macro_f1.std()))
    print("Train  macroF1 %.4f top1 %.4f" % (train_metrics["macro_f1"], train_metrics["top1_accuracy"]))
    print("Test   macroF1 %.4f top1 %.4f logloss %.4f" % (
        test_metrics["macro_f1"], test_metrics["top1_accuracy"], test_metrics["log_loss"]))
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
