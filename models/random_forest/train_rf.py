"""PathSnap - Model 4: Random Forest candidate classifier.

Reads the team's common tables, tunes hyperparameters with group-aware CV on the
shared folds (80 training trips), refits the best configuration on all 80 trips,
evaluates once on the 10 student-test trips, and saves metrics, plots,
probabilities and the trained model.

Only labeled candidates are used for fitting and metrics. Every candidate gets a
probability, including those at GPS points with no aligned reference route, so
the HMM/Viterbi step has emissions for the whole trip.

Usage (from the repository root, with the tables in PathSnap_Validated_Dataset/):
    python models/random_forest/train_rf.py

Add --no-tune to train only the baseline configuration, or --fast for a much
quicker run with fewer trees, each built from a quarter of the training rows.
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
REPO_ROOT = Path(__file__).resolve().parents[2]
OBSERVATION_KEYS = ["partition", "trip_id", "observation_index"]
KEYS = OBSERVATION_KEYS + ["arc_id"]
# Common team features. Identifiers and absolute timestamp are excluded on purpose.
FEATURES = [
    "gps_to_segment_distance_meters", "gps_to_segment_start_meters",
    "gps_to_segment_end_meters", "segment_length_meters", "candidate_rank",
    "candidate_count", "gps_speed_mps", "gps_movement_direction_degrees",
    "previous_gps_speed_mps", "previous_movement_direction_degrees",
    "previous_delta_time_seconds", "road_segment_direction_degrees",
    "gps_road_direction_difference_degrees", "delta_time_seconds", "speed_suspicious",
]
BASE_PARAMS = dict(n_estimators=100, max_depth=20, min_samples_leaf=10,
                   max_features="sqrt", class_weight=None, random_state=SEED, n_jobs=-1)
# Each option changes one setting from the baseline. Log loss picks the winner
# because the probabilities become HMM emission scores.
TUNING_OPTIONS = [
    {},
    {"max_depth": 12},
    {"max_depth": None},
    {"min_samples_leaf": 5},
    {"min_samples_leaf": 25},
    {"class_weight": "balanced_subsample"},
]
FAST_OPTION = {"n_estimators": 40, "max_samples": 0.25}


def make_model(params):
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),  # fitted on training rows only
        ("rf", RandomForestClassifier(**params)),
    ])


def load_data(features_path, labels_path):
    """Return every candidate; `label` is NaN where no reference route aligned."""
    feats = pd.read_csv(features_path, usecols=KEYS + FEATURES,
                        dtype={"trip_id": str, **{c: "float32" for c in FEATURES}})
    labs = pd.read_csv(labels_path, dtype={"trip_id": str}, usecols=KEYS + ["label"])
    if len(feats) != len(labs):
        raise SystemExit(f"Feature table has {len(feats)} rows but label table has "
                         f"{len(labs)}. Regenerate both from the same candidate table.")
    df = feats.merge(labs, on=KEYS, how="left", validate="one_to_one", indicator=True)
    if (df["_merge"] != "both").any():
        raise SystemExit("Feature and label tables contain different candidates. "
                         "Regenerate both from the same candidate table.")
    df = df.drop(columns="_merge")
    print(f"candidates: {len(df)}, labeled: {int(df['label'].notna().sum())}")
    return df


def load_candidate_summary(path):
    """Per-trip GPS observation counts from candidate generation."""
    summary = pd.read_csv(path, usecols=["partition", "trip_id", "gps_observations"],
                          dtype={"trip_id": str})
    return summary[summary.partition.isin(["train", "student_test"])]


def load_folds(path, train_trips):
    """Read the shared folds and confirm they split the training trips correctly."""
    folds = pd.read_csv(path, dtype={"trip_id": str})
    validation_counts = folds[folds.cv_role == "validation"].trip_id.value_counts()
    if set(folds.trip_id) != train_trips or set(validation_counts.index) != train_trips \
            or (validation_counts != 1).any():
        raise SystemExit("Folds must place each training trip in validation exactly once.")
    for fold, group in folds.groupby("fold"):
        roles = group.groupby("trip_id").cv_role.nunique()
        if len(group) != len(train_trips) or (roles != 1).any():
            raise SystemExit(f"Fold {fold} does not list each trip in one role.")
    return folds


def require_complete_trips(df, summary):
    """Stop on partial tables, such as an interrupted candidate generation run."""
    labeled = df[df["label"].notna()]
    expected = set(map(tuple, summary[["partition", "trip_id"]].values))
    found = set(map(tuple, labeled[["partition", "trip_id"]].drop_duplicates().values))
    missing = sorted(expected - found)
    if missing:
        raise SystemExit(f"{len(missing)} of {len(expected)} trips have no labeled "
                         f"candidates, first: {missing[0]}. Finish the data pipeline first.")


def top1_accuracy(df, prob):
    d = pd.DataFrame({"trip_id": df["trip_id"].values,
                      "obs": df["observation_index"].values,
                      "label": df["label"].values, "p": prob})
    best = d.loc[d.groupby(["trip_id", "obs"])["p"].idxmax()]
    return float(best["label"].mean())


def true_arc_not_in_candidates_rate(df):
    """Share of aligned GPS points whose reference arc is not a candidate."""
    g = df.groupby(["trip_id", "observation_index"])["label"].max()
    return float((g == 0).mean())


def coverage_metrics(candidates, summary):
    """Coverage over all cleaned GPS points of the evaluated trips."""
    total = int(summary["gps_observations"].sum())
    with_candidates = len(candidates[OBSERVATION_KEYS].drop_duplicates())
    labeled = candidates.loc[candidates["label"].notna(), OBSERVATION_KEYS]
    return {
        "gps_observations": total,
        # Same definition as candidate_generation_summary.csv.
        "candidate_miss_rate": 1 - with_candidates / total,
        "labeled_observation_coverage": len(labeled.drop_duplicates()) / total,
    }


def evaluate(candidates, prob, summary):
    """Score labeled candidates and report coverage for the same trips."""
    mask = candidates["label"].notna().values
    df, p = candidates[mask], prob[mask]
    y = df["label"].astype(int).values
    pred = (p >= 0.5).astype(int)
    return {
        "accuracy": accuracy_score(y, pred),
        "precision": precision_score(y, pred, zero_division=0),
        "recall": recall_score(y, pred, zero_division=0),
        "f1_positive": f1_score(y, pred, zero_division=0),
        "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
        "log_loss": log_loss(y, np.clip(p, 1e-15, 1 - 1e-15), labels=[0, 1]),
        "top1_accuracy": top1_accuracy(df, p),
        "true_arc_not_in_candidates_rate": true_arc_not_in_candidates_rate(df),
        **coverage_metrics(candidates, summary),
    }


def fit_model(rows, params):
    labeled = rows[rows["label"].notna()]
    model = make_model(params)
    t0 = time.time()
    model.fit(labeled[FEATURES], labeled["label"].astype(int))
    return model, time.time() - t0


def cross_validate(train, train_summary, folds, params):
    """Return per-fold metrics and an out-of-fold probability for each candidate."""
    rows = []
    oof = np.full(len(train), np.nan)
    for fold, group in folds.groupby("fold"):
        tr_trips = set(group.loc[group.cv_role == "training", "trip_id"])
        va_trips = set(group.loc[group.cv_role == "validation", "trip_id"])
        va_mask = train.trip_id.isin(va_trips).values
        va = train[va_mask]
        model, fit_seconds = fit_model(train[train.trip_id.isin(tr_trips)], params)
        prob = model.predict_proba(va[FEATURES])[:, 1]
        oof[va_mask] = prob
        res = evaluate(va, prob, train_summary[train_summary.trip_id.isin(va_trips)])
        res.update(fold=int(fold), train_trips=len(tr_trips), val_trips=len(va_trips),
                   fit_seconds=fit_seconds)
        rows.append(res)
    return pd.DataFrame(rows), oof


def tune(train, train_summary, folds, options):
    """Pick the option with the lowest mean validation log loss."""
    best = None
    results = []
    for number, option in enumerate(options, start=1):
        params = {**BASE_PARAMS, **option}
        cv, oof = cross_validate(train, train_summary, folds, params)
        score = cv["log_loss"].mean()
        results.append({"option": json.dumps(option), "mean_log_loss": score,
                        "std_log_loss": cv["log_loss"].std(),
                        "mean_macro_f1": cv["macro_f1"].mean(),
                        "mean_top1_accuracy": cv["top1_accuracy"].mean(),
                        "mean_fit_seconds": cv["fit_seconds"].mean()})
        print(f"option {number}/{len(options)} {option or 'baseline'}: "
              f"logloss={score:.4f} macroF1={cv['macro_f1'].mean():.4f} "
              f"top1={cv['top1_accuracy'].mean():.4f}")
        if best is None or score < best[0]:
            best = (score, params, cv, oof)
    table = pd.DataFrame(results).sort_values("mean_log_loss")
    return best[1], best[2], best[3], table


def probability_table(rows, prob):
    table = rows[KEYS + ["label"]].copy()
    table["label"] = table["label"].astype("Int64")
    table["rf_probability"] = prob
    return table


def save_plots(out, test, test_prob, cv, model):
    labeled = test["label"].notna().values
    cm = confusion_matrix(test.loc[labeled, "label"].astype(int),
                          (test_prob[labeled] >= 0.5).astype(int), labels=[0, 1])
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


def main():
    outputs = REPO_ROOT / "outputs"
    data = REPO_ROOT / "PathSnap_Validated_Dataset"
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--features", type=Path, default=data / "candidate_feature_table.csv")
    ap.add_argument("--labels", type=Path, default=data / "candidate_labels.csv")
    ap.add_argument("--candidate-summary", type=Path,
                    default=data / "candidate_generation_summary.csv")
    ap.add_argument("--folds", type=Path, default=outputs / "cross_validation_folds.csv")
    ap.add_argument("--out", type=Path, default=outputs / "random_forest")
    speed = ap.add_mutually_exclusive_group()
    speed.add_argument("--no-tune", action="store_true", help="train only the baseline")
    speed.add_argument("--fast", action="store_true", help="train one smaller, quicker forest")
    ap.add_argument("--reuse-cv", action="store_true",
                    help="with --no-tune or --fast, keep the CV results already in --out")
    args = ap.parse_args()
    if args.reuse_cv and not (args.no_tune or args.fast):
        raise SystemExit("--reuse-cv needs --no-tune or --fast.")
    for path in (args.features, args.labels, args.candidate_summary, args.folds):
        if not path.is_file():
            raise SystemExit(f"Missing input {path}.")
    out = args.out
    out.mkdir(parents=True, exist_ok=True)

    df = load_data(args.features, args.labels)
    summary = load_candidate_summary(args.candidate_summary)
    require_complete_trips(df, summary)
    train = df[df["partition"] == "train"].reset_index(drop=True)
    test = df[df["partition"] == "student_test"].reset_index(drop=True)
    train_summary = summary[summary.partition == "train"]
    test_summary = summary[summary.partition == "student_test"]
    folds = load_folds(args.folds, set(train_summary.trip_id))
    print(f"train rows {len(train)} ({train.trip_id.nunique()} trips), "
          f"student_test rows {len(test)} ({test.trip_id.nunique()} trips)")
    print("positive rate (train):", round(train["label"].mean(), 4))

    # ---- tuning and group-aware CV on the shared folds (split by trip) ----
    if args.fast:
        options = [FAST_OPTION]
    else:
        options = TUNING_OPTIONS[:1] if args.no_tune else TUNING_OPTIONS
    if args.reuse_cv:
        for name in ("cv_fold_metrics.csv", "rf_train_oof_probabilities.csv"):
            if not (out / name).is_file():
                raise SystemExit(f"Cannot reuse CV: {out / name} is missing.")
        params = {**BASE_PARAMS, **options[0]}
        cv = pd.read_csv(out / "cv_fold_metrics.csv")
        print("reusing saved CV results")
    else:
        params, cv, oof, tuning = tune(train, train_summary, folds, options)
        cv.to_csv(out / "cv_fold_metrics.csv", index=False)
        tuning.to_csv(out / "tuning_results.csv", index=False)
        probability_table(train, oof).to_csv(out / "rf_train_oof_probabilities.csv",
                                             index=False)
    print("selected:", params)

    # ---- final fit on all 80 training trips ----
    model, train_seconds = fit_model(train, params)
    train_prob = model.predict_proba(train[FEATURES])[:, 1]
    train_metrics = evaluate(train, train_prob, train_summary)

    # ---- single evaluation on student-test, after all choices are fixed ----
    t0 = time.time()
    test_prob = model.predict_proba(test[FEATURES])[:, 1]
    infer_seconds = time.time() - t0
    test_metrics = evaluate(test, test_prob, test_summary)
    test_metrics["inference_seconds_total"] = infer_seconds
    test_metrics["inference_ms_per_1000_rows"] = 1000 * infer_seconds / len(test) * 1000

    report = {
        "model": "RandomForestClassifier",
        "seed": SEED,
        "hyperparameters": params,
        "tuning_metric": "mean CV log loss",
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
    (out / "metrics.json").write_text(json.dumps(report, indent=2, default=float))
    (out / "model_config.json").write_text(json.dumps(
        {"seed": SEED, "hyperparameters": params, "features": FEATURES}, indent=2))

    # probabilities for the HMM/Viterbi step
    probability_table(test, test_prob).to_csv(out / "rf_student_test_probabilities.csv",
                                              index=False)
    joblib.dump(model, out / "rf_model.joblib", compress=3)
    save_plots(out, test, test_prob, cv, model)

    print("\nCV macro F1: %.4f +/- %.4f" % (cv.macro_f1.mean(), cv.macro_f1.std()))
    print("Train  macroF1 %.4f top1 %.4f" % (train_metrics["macro_f1"], train_metrics["top1_accuracy"]))
    print("Test   macroF1 %.4f top1 %.4f logloss %.4f" % (
        test_metrics["macro_f1"], test_metrics["top1_accuracy"], test_metrics["log_loss"]))
    print("Saved to", out.resolve())


if __name__ == "__main__":
    main()
