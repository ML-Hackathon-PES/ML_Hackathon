"""Create candidate-level ML features without candidate-label or model logic."""

from __future__ import annotations

import csv
import math
from pathlib import Path


EARTH_RADIUS_METERS = 6_371_008.8


def bearing_degrees(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angle_difference(first: float, second: float) -> float:
    if not math.isfinite(first) or not math.isfinite(second):
        return math.nan
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def load_nodes(path: Path) -> list[tuple[float, float] | None]:
    nodes = []
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            tokens = line.split()
            if len(tokens) != 2:
                nodes.append(None)
                continue
            try:
                lon, lat = map(float, tokens)
            except ValueError:
                nodes.append(None)
                continue
            if math.isfinite(lon) and math.isfinite(lat) and -180 <= lon <= 180 and -90 <= lat <= 90:
                nodes.append((lon, lat))
            else:
                nodes.append(None)
    return nodes


def run_feature_engineering(
    dataset_root: Path,
    gps_feature_table: Path,
    candidate_table: Path,
    output_path: Path,
) -> None:
    gps_rows = {}
    with gps_feature_table.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            gps_rows[(row["partition"], row["trip_id"], int(row["observation_index"]))] = row

    previous_rows = {}
    with gps_feature_table.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["partition"], row["trip_id"])
            previous_rows.setdefault(key, {})[int(row["observation_index"])] = row

    node_cache = {}
    fieldnames = [
        "partition", "trip_id", "observation_index", "arc_id", "from_node_id",
        "to_node_id", "gps_to_segment_distance_meters",
        "gps_to_segment_start_meters", "gps_to_segment_end_meters",
        "segment_length_meters", "candidate_rank", "candidate_count",
        "gps_speed_mps", "gps_movement_direction_degrees",
        "previous_gps_speed_mps", "previous_movement_direction_degrees",
        "previous_delta_time_seconds", "road_segment_direction_degrees",
        "gps_road_direction_difference_degrees", "timestamp_seconds",
        "delta_time_seconds", "speed_suspicious",
    ]

    with output_path.open("w", newline="", encoding="utf-8") as destination:
        writer = csv.DictWriter(destination, fieldnames=fieldnames)
        writer.writeheader()
        with candidate_table.open(encoding="utf-8-sig", newline="") as source:
            for candidate in csv.DictReader(source):
                partition = candidate["partition"]
                trip_id = candidate["trip_id"]
                observation_index = int(candidate["observation_index"])
                gps_key = (partition, trip_id, observation_index)
                gps = gps_rows.get(gps_key)
                if gps is None:
                    continue

                trip_key = (partition, trip_id)
                trip_rows = previous_rows[trip_key]
                gps_movement_direction = math.nan
                if observation_index > 0:
                    previous = trip_rows.get(observation_index - 1)
                    if previous is not None:
                        delta_lon = float(gps["delta_longitude_degrees"])
                        delta_lat = float(gps["delta_latitude_degrees"])
                        if abs(delta_lon) > 0 or abs(delta_lat) > 0:
                            gps_movement_direction = bearing_degrees(
                                (float(previous["longitude"]), float(previous["latitude"])),
                                (float(gps["longitude"]), float(gps["latitude"])),
                            )

                previous_movement_direction = math.nan
                previous_speed = math.nan
                previous_delta_time = math.nan
                if observation_index > 1:
                    previous = trip_rows.get(observation_index - 1)
                    before_previous = trip_rows.get(observation_index - 2)
                    if previous is not None:
                        previous_speed = float(previous["speed_meters_per_second"])
                        previous_delta_time = float(previous["delta_time_seconds"])
                    if previous is not None and before_previous is not None:
                        previous_movement_direction = bearing_degrees(
                            (float(before_previous["longitude"]), float(before_previous["latitude"])),
                            (float(previous["longitude"]), float(previous["latitude"])),
                        )

                if trip_key not in node_cache:
                    node_cache[trip_key] = load_nodes(
                        dataset_root / partition / trip_id / f"{trip_id}.nodes"
                    )
                nodes = node_cache[trip_key]
                from_node = int(candidate["from_node_id"])
                to_node = int(candidate["to_node_id"])
                road_direction = math.nan
                if 0 <= from_node < len(nodes) and 0 <= to_node < len(nodes):
                    first = nodes[from_node]
                    second = nodes[to_node]
                    if first is not None and second is not None:
                        road_direction = bearing_degrees(first, second)

                writer.writerow({
                    "partition": partition,
                    "trip_id": trip_id,
                    "observation_index": observation_index,
                    "arc_id": int(candidate["arc_id"]),
                    "from_node_id": from_node,
                    "to_node_id": to_node,
                    "gps_to_segment_distance_meters": candidate["gps_to_segment_distance_meters"],
                    "gps_to_segment_start_meters": candidate["gps_to_segment_start_meters"],
                    "gps_to_segment_end_meters": candidate["gps_to_segment_end_meters"],
                    "segment_length_meters": candidate["segment_length_meters"],
                    "candidate_rank": int(candidate["candidate_rank"]),
                    "candidate_count": int(candidate["candidate_count"]),
                    "gps_speed_mps": float(gps["speed_meters_per_second"]),
                    "gps_movement_direction_degrees": gps_movement_direction,
                    "previous_gps_speed_mps": previous_speed,
                    "previous_movement_direction_degrees": previous_movement_direction,
                    "previous_delta_time_seconds": previous_delta_time,
                    "road_segment_direction_degrees": road_direction,
                    "gps_road_direction_difference_degrees": angle_difference(
                        gps_movement_direction, road_direction
                    ),
                    "timestamp_seconds": float(gps["timestamp_seconds"]),
                    "delta_time_seconds": float(gps["delta_time_seconds"]),
                    "speed_suspicious": int(gps["speed_suspicious"]),
                })


def main() -> None:
    import argparse

    repository_root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "ML_Hackathon" / "Student folder" / "Student folder",
    )
    parser.add_argument(
        "--gps-feature-table",
        type=Path,
        default=repository_root / "outputs" / "gps_feature_table.csv",
    )
    parser.add_argument(
        "--candidate-table",
        type=Path,
        default=repository_root / "outputs" / "candidate_table.csv",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=repository_root / "outputs" / "candidate_feature_table.csv",
    )
    arguments = parser.parse_args()
    run_feature_engineering(
        arguments.dataset_root.resolve(),
        arguments.gps_feature_table.resolve(),
        arguments.candidate_table.resolve(),
        arguments.output.resolve(),
    )
    print(f"Candidate feature table written to {arguments.output.resolve()}")


if __name__ == "__main__":
    main()
