"""PATHSNAP dataset preparation for dataset exploration deliverables."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Iterator


EARTH_RADIUS_METERS = 6_371_008.8


def haversine_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(hav)))


def valid_coordinate(longitude: float, latitude: float) -> bool:
    return (
        math.isfinite(longitude)
        and math.isfinite(latitude)
        and -180 <= longitude <= 180
        and -90 <= latitude <= 90
    )


def read_track(path: Path) -> Iterator[tuple[float, float, float]]:
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            tokens = line.split()
            if len(tokens) != 3:
                continue
            try:
                longitude, latitude, timestamp = (float(token) for token in tokens)
            except ValueError:
                continue
            if (
                valid_coordinate(longitude, latitude)
                and math.isfinite(timestamp)
            ):
                yield longitude, latitude, timestamp


def clean_track(
    path: Path,
    maximum_speed_mps: float = 150.0,
) -> tuple[list[dict[str, float | int]], dict[str, int]]:
    cleaned: list[dict[str, float | int]] = []
    statistics = {
        "input_observations": 0,
        "missing_or_invalid_coordinates_or_timestamp": 0,
        "non_increasing_timestamp": 0,
        "duplicate_coordinate_timestamp": 0,
        "speed_outlier": 0,
        "output_observations": 0,
        "speed_suspicious_observations": 0,
    }

    previous_key: tuple[float, float, float] | None = None
    cumulative_distance = 0.0
    previous_accepted: tuple[float, float, float] | None = None

    with path.open(encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            statistics["input_observations"] += 1
            tokens = line.split()
            if len(tokens) != 3:
                statistics["missing_or_invalid_coordinates_or_timestamp"] += 1
                continue
            try:
                longitude, latitude, timestamp = (float(token) for token in tokens)
            except ValueError:
                statistics["missing_or_invalid_coordinates_or_timestamp"] += 1
                continue
            if not valid_coordinate(longitude, latitude) or not math.isfinite(timestamp):
                statistics["missing_or_invalid_coordinates_or_timestamp"] += 1
                continue
            key = (longitude, latitude, timestamp)
            if previous_key is not None and key == previous_key:
                statistics["duplicate_coordinate_timestamp"] += 1
                continue
            if previous_accepted is not None and timestamp <= previous_accepted[2]:
                statistics["non_increasing_timestamp"] += 1
                continue

            delta_time = 0.0
            step_distance = 0.0
            speed = 0.0
            if previous_accepted is not None:
                delta_time = timestamp - previous_accepted[2]
                step_distance = haversine_meters(
                    (previous_accepted[0], previous_accepted[1]),
                    (longitude, latitude),
                )
                speed = step_distance / delta_time if delta_time > 0 else 0.0
            speed_suspicious = int(speed > maximum_speed_mps)
            if speed_suspicious:
                statistics["speed_suspicious_observations"] += 1
                statistics["speed_outlier"] += 1

            cumulative_distance += step_distance
            cleaned.append({
                "longitude": longitude,
                "latitude": latitude,
                "timestamp_seconds": timestamp,
                "delta_time_seconds": delta_time,
                "delta_longitude_degrees": (
                    longitude - previous_accepted[0] if previous_accepted else 0.0
                ),
                "delta_latitude_degrees": (
                    latitude - previous_accepted[1] if previous_accepted else 0.0
                ),
                "step_distance_meters": step_distance,
                "speed_meters_per_second": speed,
                "cumulative_distance_meters": cumulative_distance,
                "speed_suspicious": speed_suspicious,
            })
            previous_accepted = (longitude, latitude, timestamp)
            previous_key = key

    statistics["output_observations"] = len(cleaned)
    return cleaned, statistics


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def find_trip_dirs(dataset_root: Path) -> list[tuple[str, Path]]:
    trips: list[tuple[str, Path]] = []
    for partition in sorted(path for path in dataset_root.iterdir() if path.is_dir()):
        name = partition.name.lower()
        if "evaluator" in name or name in {"final_test", "final-testing"}:
            continue
        trips.extend(
            (partition.name, trip)
            for trip in sorted(path for path in partition.iterdir() if path.is_dir())
        )
    return trips


def create_partition_splits(dataset_root: Path) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    trip_ids = sorted(trip.name for partition, trip in find_trip_dirs(dataset_root) if partition == "train")
    fold_count = 5

    partition_rows: list[dict[str, object]] = []
    cv_rows: list[dict[str, object]] = []
    for partition, trip in find_trip_dirs(dataset_root):
        if partition == "train":
            role = "cross_validation_pool"
        elif partition == "student_test":
            role = "development_evaluation"
        else:
            role = "unclassified"
        partition_rows.append({
            "partition": partition,
            "trip_id": trip.name,
            "role": role,
            "used_for_feature_selection": role == "cross_validation_pool",
            "used_for_model_evaluation": role == "development_evaluation",
            "used_for_evaluator_testing": False,
        })

    for fold in range(fold_count):
        validation_for_fold = set(
            trip_ids[index] for index in range(fold, len(trip_ids), fold_count)
        )
        for trip_id in trip_ids:
            cv_rows.append({
                "fold": fold + 1,
                "trip_id": trip_id,
                "cv_role": "validation" if trip_id in validation_for_fold else "training",
            })
    return partition_rows, cv_rows


def run_preprocessing(dataset_root: Path, output_dir: Path) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(dataset_root)

    cleaned_sample_rows: list[dict[str, object]] = []
    cleaned_feature_rows: list[dict[str, object]] = []
    cleaning_log: list[dict[str, object]] = []
    for partition, trip_dir in find_trip_dirs(dataset_root):
        track_path = trip_dir / f"{trip_dir.name}.track"
        if not track_path.is_file():
            continue
        cleaned, counts = clean_track(track_path)
        cleaning_log.append({"partition": partition, "trip_id": trip_dir.name, **counts})

        for observation_index, row in enumerate(cleaned):
            common = {"partition": partition, "trip_id": trip_dir.name, "observation_index": observation_index}
            cleaned_feature_rows.append({**common, **row})
            if observation_index < 25:
                cleaned_sample_rows.append({**common, **row})

    write_csv(
        output_dir / "cleaned_gps_samples.csv",
        [
            "partition", "trip_id", "observation_index", "longitude", "latitude",
            "timestamp_seconds", "delta_time_seconds", "delta_longitude_degrees",
            "delta_latitude_degrees", "step_distance_meters",
            "speed_meters_per_second", "cumulative_distance_meters", "speed_suspicious",
        ],
        cleaned_sample_rows,
    )
    write_csv(
        output_dir / "gps_feature_table.csv",
        [
            "partition", "trip_id", "observation_index", "longitude", "latitude",
            "timestamp_seconds", "delta_time_seconds", "delta_longitude_degrees",
            "delta_latitude_degrees", "step_distance_meters",
            "speed_meters_per_second", "cumulative_distance_meters", "speed_suspicious",
        ],
        cleaned_feature_rows,
    )
    write_csv(
        output_dir / "gps_cleaning_log.csv",
        [
            "partition", "trip_id", "input_observations",
            "missing_or_invalid_coordinates_or_timestamp", "non_increasing_timestamp",
            "duplicate_coordinate_timestamp", "speed_outlier",
            "speed_suspicious_observations", "output_observations",
        ],
        cleaning_log,
    )

    sensitivity_rows: list[dict[str, object]] = []
    for threshold in (50.0, 100.0, 150.0, 200.0, 300.0):
        suspicious_rows = [
            row for row in cleaned_feature_rows
            if float(row["speed_meters_per_second"]) > threshold
        ]
        suspicious_trip_count = len({
            (row["partition"], row["trip_id"]) for row in suspicious_rows
        })
        sensitivity_rows.append({
            "speed_threshold_mps": threshold,
            "suspicious_observation_count": len(suspicious_rows),
            "suspicious_trip_count": suspicious_trip_count,
        })
    write_csv(
        output_dir / "speed_outlier_sensitivity.csv",
        ["speed_threshold_mps", "suspicious_observation_count", "suspicious_trip_count"],
        sensitivity_rows,
    )
    speed_audit_rows = [
        {
            "partition": row["partition"],
            "trip_id": row["trip_id"],
            "observation_index": row["observation_index"],
            "timestamp_seconds": row["timestamp_seconds"],
            "longitude": row["longitude"],
            "latitude": row["latitude"],
            "speed_meters_per_second": row["speed_meters_per_second"],
            "threshold_mps": 150.0,
            "disposition": "preserved_and_flagged",
        }
        for row in cleaned_feature_rows
        if int(row["speed_suspicious"]) == 1
    ]
    write_csv(
        output_dir / "speed_outlier_audit.csv",
        [
            "partition", "trip_id", "observation_index", "timestamp_seconds",
            "longitude", "latitude", "speed_meters_per_second",
            "threshold_mps", "disposition",
        ],
        speed_audit_rows,
    )

    partition_rows, cv_rows = create_partition_splits(dataset_root)
    write_csv(
        output_dir / "dataset_splits.csv",
        [
            "partition", "trip_id", "role", "used_for_feature_selection",
            "used_for_model_evaluation", "used_for_evaluator_testing",
        ],
        partition_rows,
    )
    write_csv(
        output_dir / "cross_validation_folds.csv",
        ["fold", "trip_id", "cv_role"],
        cv_rows,
    )

    partition_config = {
        "train_partition": "train",
        "student_test_partition": "student_test",
        "evaluator_test_partition": "not_available_locally",
        "training_pool_from_train": "all 80 trips",
        "holdout": "none",
        "cross_validation_folds": 5,
        "cross_validation_pool": "80 train trips",
        "student_test_usage": "development-stage evaluation only",
        "evaluator_test_usage": "final evaluation only; never use labels or trips for tuning",
    }
    feature_config = {
        "maximum_speed_mps": 150.0,
        "coordinate_order": ["longitude", "latitude"],
        "cleaning": [
            "valid finite longitude and latitude",
            "valid finite timestamp",
            "strictly increasing timestamps",
            "remove exact duplicate observations",
            "remove observations with speed above 150 m/s",
        ],
        "features": [
            "longitude",
            "latitude",
            "timestamp_seconds",
            "delta_time_seconds",
            "delta_longitude_degrees",
            "delta_latitude_degrees",
            "step_distance_meters",
            "speed_meters_per_second",
            "cumulative_distance_meters",
            "speed_suspicious",
        ],
        "distance_method": "Haversine great-circle distance",
        "candidate_generation": "not performed",
        "route_reconstruction": "not performed",
        "model_training": "not performed",
    }
    config_dir = output_dir.parent / "config"
    config_dir.mkdir(parents=True, exist_ok=True)
    (config_dir / "dataset_partitions.json").write_text(
        json.dumps(partition_config, indent=2) + "\n",
        encoding="utf-8",
    )
    (config_dir / "feature_preparation.json").write_text(
        json.dumps(feature_config, indent=2) + "\n",
        encoding="utf-8",
    )

    print(f"Prepared {len(cleaned_feature_rows):,} cleaned feature rows from {len(cleaning_log)} trips.")
    print(f"Saved preprocessing artifacts in {output_dir}")


def main() -> None:
    import argparse

    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "ML_Hackathon" / "Student folder" / "Student folder",
    )
    parser.add_argument("--output-dir", type=Path, default=repository_root / "outputs")
    arguments = parser.parse_args()
    run_preprocessing(arguments.dataset_root.resolve(), arguments.output_dir.resolve())


if __name__ == "__main__":
    main()
