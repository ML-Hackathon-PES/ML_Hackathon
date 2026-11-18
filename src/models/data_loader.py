"""Shared dataset loader for PathSnap map-matching models.

Joins candidate_feature_table.csv with candidate_labels.csv on the composite
key (partition, trip_id, observation_index, arc_id).  Drops unlabeled rows and
splits into train / student_test.  Returns numpy arrays ready for sklearn
Pipelines; identifiers are returned separately for grouping and diagnostics.

No preprocessing (scaling, imputation) is applied here.  Each model's Pipeline
must fit those transformers inside cross-validation folds only.
"""

from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd


# Columns that identify rows but must never be model inputs
_ID_COLS = ["partition", "trip_id", "observation_index", "arc_id",
            "from_node_id", "to_node_id"]


def load_config(config_path: str | Path) -> dict:
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def load_dataset(config: dict, repo_root: str | Path) -> dict:
    """Load, join, and partition the candidate-level dataset.

    Parameters
    ----------
    config      : dict loaded from a model config JSON.
    repo_root   : repository root (all relative paths in config resolve from here).

    Returns
    -------
    dict with keys:
        X_train, y_train, groups_train  — training arrays + trip_id groups
        X_student, y_student, groups_student
        feature_names                   — list[str] of feature column names
        df_train, df_student            — full DataFrames (for diagnostics)
    """
    t0 = time.perf_counter()
    root = Path(repo_root)

    feat_path   = root / config["feature_table"]
    label_path  = root / config["labels_table"]
    features    = config["model_features"]

    print(f"Loading feature table  : {feat_path}")
    feat_df = pd.read_csv(feat_path, low_memory=False)

    print(f"Loading labels table   : {label_path}")
    label_df = pd.read_csv(label_path, low_memory=False)

    # Keep only the columns needed from labels
    label_cols = ["partition", "trip_id", "observation_index", "arc_id", "label"]
    label_df = label_df[label_cols].copy()

    # Join on composite key
    df = feat_df.merge(label_df,
                       on=["partition", "trip_id", "observation_index", "arc_id"],
                       how="inner")

    # Drop unlabeled rows (empty label string — not forced-negative)
    if config.get("drop_unlabeled", True):
        before = len(df)
        df = df[df["label"].notna() & (df["label"].astype(str).str.strip() != "")]
        print(f"Dropped {before - len(df):,} unlabeled rows; {len(df):,} remain.")

    # Cast label to int
    df["label"] = df["label"].astype(int)

    # Split partitions
    train_part   = config.get("train_partition", "train")
    student_part = config.get("student_test_partition", "student_test")

    df_train   = df[df["partition"] == train_part].reset_index(drop=True)
    df_student = df[df["partition"] == student_part].reset_index(drop=True)

    # Validate feature columns exist
    missing_feats = [f for f in features if f not in df.columns]
    if missing_feats:
        raise ValueError(f"Features missing from dataset: {missing_feats}")

    X_train   = df_train[features].values.astype(np.float32)
    y_train   = df_train["label"].values.astype(np.int8)
    groups_train = df_train["trip_id"].values

    X_student   = df_student[features].values.astype(np.float32)
    y_student   = df_student["label"].values.astype(np.int8)
    groups_student = df_student["trip_id"].values

    elapsed = time.perf_counter() - t0
    print(f"Dataset loaded in {elapsed:.1f}s")
    print(f"  Train   : {X_train.shape[0]:,} rows, {X_train.shape[1]} features")
    print(f"  Student : {X_student.shape[0]:,} rows")
    pos_rate = y_train.mean() * 100
    print(f"  Train positive rate: {pos_rate:.1f}%")

    return {
        "X_train":        X_train,
        "y_train":        y_train,
        "groups_train":   groups_train,
        "X_student":      X_student,
        "y_student":      y_student,
        "groups_student": groups_student,
        "feature_names":  features,
        "df_train":       df_train,
        "df_student":     df_student,
    }


def load_cv_splits(config: dict, repo_root: str | Path,
                   groups_train: np.ndarray) -> list[tuple[np.ndarray, np.ndarray]]:
    """Build sklearn-compatible (train_idx, val_idx) pairs from the
    pre-computed cross_validation_folds.csv.  Groups are trip-level.

    Returns list of (train_indices, val_indices) for each fold.
    """
    root      = Path(repo_root)
    fold_path = root / config["cv_folds"]
    fold_df   = pd.read_csv(fold_path)

    splits = []
    for fold_num in sorted(fold_df["fold"].unique()):
        fold_rows = fold_df[fold_df["fold"] == fold_num]
        val_trips  = set(fold_rows.loc[fold_rows["cv_role"] == "validation", "trip_id"])
        train_trips = set(fold_rows.loc[fold_rows["cv_role"] == "training",  "trip_id"])

        train_idx = np.where(np.isin(groups_train, list(train_trips)))[0]
        val_idx   = np.where(np.isin(groups_train, list(val_trips)))[0]

        # Safety check: no trip overlap
        overlap = train_trips & val_trips
        if overlap:
            raise RuntimeError(f"Fold {fold_num} has trip overlap: {overlap}")

        splits.append((train_idx, val_idx))

    print(f"CV splits built: {len(splits)} folds")
    for i, (tr, vl) in enumerate(splits, 1):
        print(f"  Fold {i}: {len(tr):,} train rows, {len(vl):,} val rows")

    return splits
