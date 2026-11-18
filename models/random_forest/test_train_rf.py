"""End-to-end check of train_rf.py on small synthetic tables.

Needs no project data and writes only to a temporary folder.

Usage (from the repository root):
    python models/random_forest/test_train_rf.py
"""
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
SCRIPT = HERE / "train_rf.py"
FOLDS = HERE.parents[1] / "outputs" / "cross_validation_folds.csv"
sys.path.insert(0, str(HERE))
from train_rf import FEATURES  # noqa: E402

OBSERVATIONS_PER_TRIP = 30
CANDIDATES_PER_OBSERVATION = 3


def build_tables():
    """Each 10 GPS points hold 8 aligned, 1 unaligned and 1 without candidates."""
    rng = np.random.default_rng(0)
    train_ids = sorted(pd.read_csv(FOLDS, dtype={"trip_id": str}).trip_id.unique())
    test_ids = [f"9{i:07d}" for i in range(10)]
    gps, feats, labels = [], [], []
    for partition, trip_ids in (("train", train_ids), ("student_test", test_ids)):
        for trip in trip_ids:
            for obs in range(OBSERVATIONS_PER_TRIP):
                gps.append((partition, trip, obs))
                if obs % 10 == 9:
                    continue
                aligned = obs % 10 != 8
                has_true_arc = obs % 10 != 7
                true_rank = int(rng.integers(0, CANDIDATES_PER_OBSERVATION))
                for rank in range(CANDIDATES_PER_OBSERVATION):
                    positive = aligned and has_true_arc and rank == true_rank
                    row = {name: rng.normal() for name in FEATURES}
                    row["gps_to_segment_distance_meters"] = rng.normal(5 if positive else 40, 8)
                    if obs == 0:
                        row["gps_speed_mps"] = np.nan
                    key = {"partition": partition, "trip_id": trip,
                           "observation_index": obs, "arc_id": rank}
                    feats.append({**key, **row})
                    labels.append({**key, "label": int(positive) if aligned else ""})
    summary = (pd.DataFrame(gps, columns=["partition", "trip_id", "observation_index"])
               .groupby(["partition", "trip_id"]).size()
               .reset_index(name="gps_observations"))
    return summary, pd.DataFrame(feats), pd.DataFrame(labels), train_ids


def run_script(work, features, labels, *extra):
    return subprocess.run(
        [sys.executable, str(SCRIPT), "--features", str(work / features),
         "--labels", str(work / labels),
         "--candidate-summary", str(work / "candidate_summary.csv"),
         "--folds", str(FOLDS), "--out", str(work / "out"), *extra],
        capture_output=True, text=True)


def check_successful_run(work, feats):
    result = run_script(work, "features.csv", "labels.csv")
    assert result.returncode == 0, result.stdout + result.stderr
    out = work / "out"

    metrics = json.loads((out / "metrics.json").read_text())["student_test_metrics"]
    assert metrics["gps_observations"] == 300
    assert abs(metrics["candidate_miss_rate"] - 0.1) < 1e-9
    assert abs(metrics["labeled_observation_coverage"] - 0.8) < 1e-9
    assert abs(metrics["true_arc_not_in_candidates_rate"] - 1 / 8) < 1e-9

    test_probs = pd.read_csv(out / "rf_student_test_probabilities.csv", dtype={"trip_id": str})
    assert len(test_probs) == (feats.partition == "student_test").sum()
    assert test_probs.rf_probability.notna().all()
    assert test_probs.label.isna().sum() == 10 * 3 * CANDIDATES_PER_OBSERVATION

    oof = pd.read_csv(out / "rf_train_oof_probabilities.csv", dtype={"trip_id": str})
    assert len(oof) == (feats.partition == "train").sum()
    assert oof.rf_probability.notna().all()
    assert oof.trip_id.str.len().eq(8).all()

    assert len(pd.read_csv(out / "tuning_results.csv")) == 6
    assert len(pd.read_csv(out / "cv_fold_metrics.csv")) == 5
    for name in ("rf_model.joblib", "confusion_matrix.png", "cv_macro_f1.png",
                 "feature_importance.png", "model_config.json"):
        assert (out / name).is_file(), name


def check_fast_run(work):
    result = run_script(work, "features.csv", "labels.csv", "--fast")
    assert result.returncode == 0, result.stdout + result.stderr
    params = json.loads((work / "out" / "metrics.json").read_text())["hyperparameters"]
    assert params["n_estimators"] == 40 and params["max_samples"] == 0.25


def check_incomplete_tables_stop(work, feats, labels, train_ids):
    keep = feats.trip_id != train_ids[0]
    feats[keep].to_csv(work / "partial_features.csv", index=False)
    labels[keep].to_csv(work / "partial_labels.csv", index=False)

    result = run_script(work, "partial_features.csv", "partial_labels.csv", "--no-tune")
    assert result.returncode != 0 and "Finish the data pipeline" in result.stderr, result.stderr

    result = run_script(work, "partial_features.csv", "labels.csv", "--no-tune")
    assert result.returncode != 0 and "Regenerate both" in result.stderr, result.stderr


def main():
    summary, feats, labels, train_ids = build_tables()
    with tempfile.TemporaryDirectory() as folder:
        work = Path(folder)
        summary.to_csv(work / "candidate_summary.csv", index=False)
        feats.to_csv(work / "features.csv", index=False)
        labels.to_csv(work / "labels.csv", index=False)
        check_successful_run(work, feats)
        check_fast_run(work)
        check_incomplete_tables_stop(work, feats, labels, train_ids)
    print("ALL CHECKS PASSED")


if __name__ == "__main__":
    main()
