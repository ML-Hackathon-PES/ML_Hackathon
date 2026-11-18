"""Validate candidate datasets and trip-level partition separation."""

from __future__ import annotations

import csv
import math
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as source:
        return list(csv.DictReader(source))


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_validation(repository_root: Path) -> None:
    output_dir = repository_root / "outputs"
    candidate_rows = read_csv(output_dir / "candidate_table.csv")
    candidate_features = read_csv(output_dir / "candidate_feature_table.csv")
    generation_summary = read_csv(output_dir / "candidate_generation_summary.csv")
    splits = read_csv(output_dir / "dataset_splits.csv")
    folds = read_csv(output_dir / "cross_validation_folds.csv")

    candidate_keys = set()
    duplicate_candidate_rows = 0
    invalid_candidate_arc_ids = 0
    invalid_candidate_node_refs = 0
    missing_feature_rows = 0

    dataset_root = repository_root / "ML_Hackathon" / "Student folder" / "Student folder"
    arc_counts = {}
    node_counts = {}
    for partition in ("train", "student_test"):
        for trip_dir in (dataset_root / partition).iterdir():
            if not trip_dir.is_dir():
                continue
            trip_id = trip_dir.name
            with (trip_dir / f"{trip_id}.arcs").open(encoding="utf-8-sig") as source:
                arc_counts[(partition, trip_id)] = sum(1 for _ in source)
            with (trip_dir / f"{trip_id}.nodes").open(encoding="utf-8-sig") as source:
                node_counts[(trip_dir.name)] = sum(1 for _ in source)

    for row in candidate_rows:
        key = (
            row["partition"], row["trip_id"], int(row["observation_index"]),
            int(row["arc_id"]),
        )
        if key in candidate_keys:
            duplicate_candidate_rows += 1
        candidate_keys.add(key)
        arc_id = int(row["arc_id"])
        if not 0 <= arc_id < arc_counts[(row["partition"], row["trip_id"])]:
            invalid_candidate_arc_ids += 1
        from_node = int(row["from_node_id"])
        to_node = int(row["to_node_id"])
        if (
            not 0 <= from_node < node_counts[row["trip_id"]]
            or not 0 <= to_node < node_counts[row["trip_id"]]
        ):
            invalid_candidate_node_refs += 1

    numeric_feature_names = [
        "gps_to_segment_distance_meters", "gps_to_segment_start_meters",
        "gps_to_segment_end_meters", "segment_length_meters",
        "candidate_rank", "candidate_count", "gps_speed_mps",
        "gps_movement_direction_degrees", "previous_gps_speed_mps",
        "previous_movement_direction_degrees", "previous_delta_time_seconds",
        "road_segment_direction_degrees", "gps_road_direction_difference_degrees",
        "timestamp_seconds", "delta_time_seconds",
    ]
    missing_feature_values = {name: 0 for name in numeric_feature_names}
    for row in candidate_features:
        for name in numeric_feature_names:
            try:
                value = float(row[name])
            except (KeyError, TypeError, ValueError):
                missing_feature_values[name] += 1
                missing_feature_rows += 1
                continue
            if not math.isfinite(value):
                missing_feature_values[name] += 1
                missing_feature_rows += 1

    train_trips = {row["trip_id"] for row in splits if row["partition"] == "train"}
    student_trips = {row["trip_id"] for row in splits if row["partition"] == "student_test"}
    partition_overlap = train_trips & student_trips

    leakage_errors = []
    fold_training_by_fold = {}
    fold_validation_by_fold = {}
    for row in folds:
        fold = int(row["fold"])
        fold_training_by_fold.setdefault(fold, set())
        fold_validation_by_fold.setdefault(fold, set())
        if row["cv_role"] == "training":
            fold_training_by_fold[fold].add(row["trip_id"])
        else:
            fold_validation_by_fold[fold].add(row["trip_id"])
    for fold in sorted(fold_training_by_fold):
        overlap = fold_training_by_fold[fold] & fold_validation_by_fold[fold]
        if overlap:
            leakage_errors.append(f"Fold {fold} overlaps: {sorted(overlap)}")

    label_distribution = {"positive": 0, "negative": 0, "unlabeled": 0}
    labels_path = output_dir / "candidate_labels.csv"
    if labels_path.is_file():
        labels = read_csv(labels_path)
        for row in labels:
            if row["label"] == "1":
                label_distribution["positive"] += 1
            elif row["label"] == "0":
                label_distribution["negative"] += 1
            else:
                label_distribution["unlabeled"] += 1
    else:
        labels = []

    gps_observations = sum(int(row["gps_observations"]) for row in generation_summary)
    without_candidates = sum(int(row["observations_without_candidates"]) for row in generation_summary)
    candidate_pairs = sum(int(row["candidate_pairs"]) for row in generation_summary)
    trips_without_candidates = sum(
        int(row["observations_without_candidates"]) == int(row["gps_observations"])
        for row in generation_summary
    )

    summary_rows = [
        {"check": "candidate_rows", "value": len(candidate_rows), "status": "PASS"},
        {"check": "candidate_pairs", "value": candidate_pairs, "status": "PASS"},
        {"check": "duplicate_candidate_rows", "value": duplicate_candidate_rows,
         "status": "PASS" if duplicate_candidate_rows == 0 else "FAIL"},
        {"check": "missing_feature_row_count", "value": missing_feature_rows,
         "status": "PASS" if missing_feature_rows == 0 else "WARNING"},
        {"check": "invalid_candidate_arc_ids", "value": invalid_candidate_arc_ids,
         "status": "PASS" if invalid_candidate_arc_ids == 0 else "FAIL"},
        {"check": "invalid_candidate_node_references", "value": invalid_candidate_node_refs,
         "status": "PASS" if invalid_candidate_node_refs == 0 else "FAIL"},
        {"check": "gps_observations", "value": gps_observations, "status": "PASS"},
        {"check": "observations_without_candidates", "value": without_candidates,
         "status": "PASS" if without_candidates == 0 else "WARNING"},
        {"check": "candidate_miss_rate", "value":
         without_candidates / gps_observations if gps_observations else 1.0,
         "status": "PASS" if without_candidates == 0 else "WARNING"},
        {"check": "trips_with_no_valid_candidates", "value": trips_without_candidates,
         "status": "PASS" if trips_without_candidates == 0 else "WARNING"},
        {"check": "positive_labels", "value": label_distribution["positive"], "status": "PASS"},
        {"check": "negative_labels", "value": label_distribution["negative"], "status": "PASS"},
        {"check": "unlabeled_candidate_rows", "value": label_distribution["unlabeled"], "status": "WARNING"},
        {"check": "train_trip_count", "value": len(train_trips),
         "status": "PASS" if len(train_trips) == 80 else "FAIL"},
        {"check": "student_test_trip_count", "value": len(student_trips),
         "status": "PASS" if len(student_trips) == 10 else "FAIL"},
        {"check": "train_student_partition_overlap", "value": len(partition_overlap),
         "status": "PASS" if not partition_overlap else "FAIL"},
        {"check": "cv_trip_overlap_errors", "value": len(leakage_errors),
         "status": "PASS" if not leakage_errors else "FAIL"},
    ]
    write_csv(
        output_dir / "dataset_validation_summary.csv",
        ["check", "value", "status"],
        summary_rows,
    )

    missing_rows = [
        {"feature": name, "missing_or_nonfinite_values": count}
        for name, count in missing_feature_values.items()
    ]
    write_csv(
        output_dir / "candidate_feature_missing_values.csv",
        ["feature", "missing_or_nonfinite_values"],
        missing_rows,
    )

    report = [
        "# Dataset Validation Report",
        "",
        "## Candidate table",
        "",
        f"- Candidate rows: {len(candidate_rows):,}",
        f"- Candidate pairs: {candidate_pairs:,}",
        f"- Duplicate candidate rows: {duplicate_candidate_rows:,}",
        f"- Invalid candidate arc IDs: {invalid_candidate_arc_ids:,}",
        f"- Invalid candidate node references: {invalid_candidate_node_refs:,}",
        f"- GPS observations without candidates: {without_candidates:,}/{gps_observations:,}",
        f"- Trips with no valid candidates: {trips_without_candidates:,}",
        "",
        "## Label distribution",
        "",
        f"- Positive labels: {label_distribution['positive']:,}",
        f"- Negative labels: {label_distribution['negative']:,}",
        f"- Unlabeled candidate rows: {label_distribution['unlabeled']:,}",
        "",
        "## Partition and leakage checks",
        "",
        f"- Train trips: {len(train_trips)}",
        f"- Student-test trips: {len(student_trips)}",
        f"- Train/student-test overlap: {len(partition_overlap)}",
        f"- Cross-validation leakage errors: {len(leakage_errors)}",
    ]
    if leakage_errors:
        report.extend(["", "### Leakage details"] + [f"- {error}" for error in leakage_errors])
    report.extend([
        "",
        "## Missing and nonfinite feature values",
        "",
        "| Feature | Missing or nonfinite |",
        "|---|---:|",
    ])
    report.extend(
        f"| {row['feature']} | {row['missing_or_nonfinite_values']:,} |"
        for row in missing_rows
    )
    report.extend([
        "",
        "## Limitations",
        "",
        "- Candidate rows are observation-candidate pairs, not model-ready labels.",
        "- Missing features are not imputed automatically.",
        "- Reference labels are spatially aligned and only retained where unambiguous.",
        "- Evaluator-testing data is excluded because it is not available locally.",
    ])
    (output_dir / "dataset_validation_report.md").write_text(
        "\n".join(report) + "\n",
        encoding="utf-8",
    )
    print(f"Dataset validation written to {output_dir}")


def main() -> None:
    import argparse

    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--repository-root",
        type=Path,
        default=repository_root,
    )
    arguments = parser.parse_args()
    run_validation(arguments.repository_root.resolve())


if __name__ == "__main__":
    main()
