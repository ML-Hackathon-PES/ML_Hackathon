"""Generate nearby road-arc candidates for each cleaned GPS observation."""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree


EARTH_RADIUS_METERS = 6_371_008.8


def point_to_segment(
    point_lon: float,
    point_lat: float,
    lon1: float,
    lat1: float,
    lon2: float,
    lat2: float,
) -> tuple[float, float, float]:
    """Return great-circle-safe planar approximation distances in local meters."""
    origin_lat = math.radians((lat1 + lat2 + point_lat) / 3.0)
    cos_lat = max(math.cos(origin_lat), 1e-6)
    px = math.radians(point_lon) * EARTH_RADIUS_METERS * cos_lat
    py = math.radians(point_lat) * EARTH_RADIUS_METERS
    ax = math.radians(lon1) * EARTH_RADIUS_METERS * cos_lat
    ay = math.radians(lat1) * EARTH_RADIUS_METERS
    bx = math.radians(lon2) * EARTH_RADIUS_METERS * cos_lat
    by = math.radians(lat2) * EARTH_RADIUS_METERS

    dx = bx - ax
    dy = by - ay
    squared_length = dx * dx + dy * dy
    if squared_length <= 1e-12:
        projection = 0.0
        closest_x, closest_y = ax, ay
    else:
        projection = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared_length))
        closest_x = ax + projection * dx
        closest_y = ay + projection * dy

    distance = math.hypot(px - closest_x, py - closest_y)
    distance_to_start = math.hypot(px - ax, py - ay)
    distance_to_end = math.hypot(px - bx, py - by)
    return distance, distance_to_start, distance_to_end


def haversine_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    dlon = lon2 - lon1
    dlat = lat2 - lat1
    hav = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(hav)))


def load_nodes(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        coordinates = []
        for line in source:
            tokens = line.split()
            if len(tokens) != 2:
                coordinates.append((np.nan, np.nan))
                continue
            try:
                coordinates.append((float(tokens[0]), float(tokens[1])))
            except ValueError:
                coordinates.append((np.nan, np.nan))
    return np.asarray(coordinates, dtype=np.float64)


def load_arcs(path: Path) -> np.ndarray:
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        arcs = []
        for line in source:
            tokens = line.split()
            if len(tokens) != 2:
                arcs.append((-1, -1))
                continue
            try:
                arcs.append((int(tokens[0]), int(tokens[1])))
            except ValueError:
                arcs.append((-1, -1))
    return np.asarray(arcs, dtype=np.int64)


def local_xy(longitudes: np.ndarray, latitudes: np.ndarray, origin_latitude: float) -> np.ndarray:
    cos_lat = max(math.cos(math.radians(origin_latitude)), 1e-6)
    return np.column_stack((
        np.radians(longitudes) * EARTH_RADIUS_METERS * cos_lat,
        np.radians(latitudes) * EARTH_RADIUS_METERS,
    ))


def generate_for_trip(
    partition: str,
    trip_id: str,
    trip_dir: Path,
    gps_table_path: Path,
    candidate_path,
    search_radius_meters: float,
    maximum_candidates: int,
) -> dict[str, object]:
    nodes_path = trip_dir / f"{trip_id}.nodes"
    arcs_path = trip_dir / f"{trip_id}.arcs"
    nodes = load_nodes(nodes_path)
    arcs = load_arcs(arcs_path)

    valid_arcs = []
    for arc_id, (from_node, to_node) in enumerate(arcs):
        if not (0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)):
            continue
        first = nodes[from_node]
        second = nodes[to_node]
        if (
            np.isfinite(first).all()
            and np.isfinite(second).all()
            and -180 <= first[0] <= 180
            and -180 <= second[0] <= 180
            and -90 <= first[1] <= 90
            and -90 <= second[1] <= 90
        ):
            valid_arcs.append((arc_id, int(from_node), int(to_node)))

    if not valid_arcs:
        return {
            "partition": partition,
            "trip_id": trip_id,
            "gps_observations": 0,
            "candidate_pairs": 0,
            "observations_with_candidates": 0,
            "observations_without_candidates": 0,
            "candidate_miss_rate": 1.0,
            "mean_candidate_count": 0.0,
            "median_candidate_count": 0.0,
            "valid_arc_count": 0,
            "invalid_arc_count": len(arcs),
        }

    with gps_table_path.open(encoding="utf-8-sig", newline="") as gps_source:
        gps_rows = [
            row for row in csv.DictReader(gps_source)
            if row["partition"] == partition and row["trip_id"] == trip_id
        ]

    gps_origin = float(gps_rows[0]["latitude"]) if gps_rows else float(nodes[np.isfinite(nodes).all(axis=1)][0, 1])
    valid_arc_ids = np.asarray([item[0] for item in valid_arcs], dtype=np.int64)
    midpoint_coordinates = np.asarray([
        ((nodes[from_node][0] + nodes[to_node][0]) / 2.0,
         (nodes[from_node][1] + nodes[to_node][1]) / 2.0)
        for _, from_node, to_node in valid_arcs
    ])
    midpoint_xy = local_xy(
        midpoint_coordinates[:, 0],
        midpoint_coordinates[:, 1],
        gps_origin,
    )
    midpoint_tree = cKDTree(midpoint_xy)

    segment_lengths = np.asarray([
        haversine_meters(
            (nodes[from_node][0], nodes[from_node][1]),
            (nodes[to_node][0], nodes[to_node][1]),
        )
        for _, from_node, to_node in valid_arcs
    ])

    gps_observations = 0
    candidate_pairs = 0
    observations_with_candidates = 0
    observations_without_candidates = 0
    candidate_counts = []

    for gps_index, row in enumerate(gps_rows):
        gps_lon = float(row["longitude"])
        gps_lat = float(row["latitude"])
        gps_xy = local_xy(np.asarray([gps_lon]), np.asarray([gps_lat]), gps_origin)[0]
        query_radius = search_radius_meters + float(np.max(segment_lengths)) if len(segment_lengths) else search_radius_meters
        nearby = midpoint_tree.query_ball_point(gps_xy, r=query_radius)

        candidates = []
        for midpoint_index in nearby:
            arc_id, from_node, to_node = valid_arcs[midpoint_index]
            first = nodes[from_node]
            second = nodes[to_node]
            distance, endpoint1, endpoint2 = point_to_segment(
                gps_lon, gps_lat,
                float(first[0]), float(first[1]),
                float(second[0]), float(second[1]),
            )
            if distance <= search_radius_meters:
                candidates.append({
                    "arc_id": int(arc_id),
                    "from_node_id": int(from_node),
                    "to_node_id": int(to_node),
                    "gps_to_segment_distance_meters": distance,
                    "gps_to_segment_start_meters": endpoint1,
                    "gps_to_segment_end_meters": endpoint2,
                    "segment_length_meters": segment_lengths[midpoint_index],
                })

        candidates.sort(key=lambda item: (
            item["gps_to_segment_distance_meters"],
            item["arc_id"],
        ))
        candidates = candidates[:maximum_candidates]
        gps_observations += 1
        candidate_counts.append(len(candidates))
        if candidates:
            observations_with_candidates += 1
        else:
            observations_without_candidates += 1
        for rank, candidate in enumerate(candidates, start=1):
            candidate_pairs += 1
            candidate_path.writerow({
                "partition": partition,
                "trip_id": trip_id,
                "observation_index": int(row["observation_index"]),
                "gps_longitude": gps_lon,
                "gps_latitude": gps_lat,
                "timestamp_seconds": float(row["timestamp_seconds"]),
                "arc_id": candidate["arc_id"],
                "from_node_id": candidate["from_node_id"],
                "to_node_id": candidate["to_node_id"],
                "gps_to_segment_distance_meters": candidate["gps_to_segment_distance_meters"],
                "gps_to_segment_start_meters": candidate["gps_to_segment_start_meters"],
                "gps_to_segment_end_meters": candidate["gps_to_segment_end_meters"],
                "segment_length_meters": candidate["segment_length_meters"],
                "candidate_rank": rank,
                "candidate_count": len(candidates),
            })

    mean_count = float(np.mean(candidate_counts)) if candidate_counts else 0.0
    median_count = float(np.median(candidate_counts)) if candidate_counts else 0.0
    return {
        "partition": partition,
        "trip_id": trip_id,
        "gps_observations": gps_observations,
        "candidate_pairs": candidate_pairs,
        "observations_with_candidates": observations_with_candidates,
        "observations_without_candidates": observations_without_candidates,
        "candidate_miss_rate": observations_without_candidates / gps_observations if gps_observations else 1.0,
        "mean_candidate_count": mean_count,
        "median_candidate_count": median_count,
        "valid_arc_count": len(valid_arcs),
        "invalid_arc_count": len(arcs) - len(valid_arcs),
    }


def run_candidate_generation(
    dataset_root: Path,
    gps_feature_table: Path,
    candidate_output: Path,
    summary_output: Path,
    search_radius_meters: float,
    maximum_candidates: int,
) -> None:
    candidate_output.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "partition", "trip_id", "observation_index", "gps_longitude",
        "gps_latitude", "timestamp_seconds", "arc_id", "from_node_id",
        "to_node_id", "gps_to_segment_distance_meters",
        "gps_to_segment_start_meters", "gps_to_segment_end_meters",
        "segment_length_meters", "candidate_rank", "candidate_count",
    ]
    summaries = []
    with candidate_output.open("w", newline="", encoding="utf-8") as candidate_file:
        writer = csv.DictWriter(candidate_file, fieldnames=fieldnames)
        writer.writeheader()
        trips = sorted(
            (partition.name, trip)
            for partition in dataset_root.iterdir()
            if partition.is_dir() and "evaluator" not in partition.name.lower()
            for trip in partition.iterdir()
            if trip.is_dir()
        )
        for index, (partition, trip_dir) in enumerate(trips, start=1):
            print(
                f"[{index}/{len(trips)}] Generating candidates "
                f"{partition}/{trip_dir.name}"
            )
            summaries.append(generate_for_trip(
                partition,
                trip_dir.name,
                trip_dir,
                gps_feature_table,
                writer,
                search_radius_meters,
                maximum_candidates,
            ))

    summary_fields = [
        "partition", "trip_id", "gps_observations", "candidate_pairs",
        "observations_with_candidates", "observations_without_candidates",
        "candidate_miss_rate", "mean_candidate_count", "median_candidate_count",
        "valid_arc_count", "invalid_arc_count",
    ]
    with summary_output.open("w", newline="", encoding="utf-8") as summary_file:
        writer = csv.DictWriter(summary_file, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)
    candidate_pairs = sum(row["candidate_pairs"] for row in summaries)
    miss_observations = sum(row["observations_without_candidates"] for row in summaries)
    gps_observations = sum(row["gps_observations"] for row in summaries)
    print(
        f"Candidate generation complete: {candidate_pairs:,} pairs, "
        f"{miss_observations:,}/{gps_observations:,} observations without candidates."
    )


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
        "--candidate-output",
        type=Path,
        default=repository_root / "outputs" / "candidate_table.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repository_root / "outputs" / "candidate_generation_summary.csv",
    )
    parser.add_argument("--search-radius-meters", type=float, default=100.0)
    parser.add_argument("--maximum-candidates", type=int, default=5)
    arguments = parser.parse_args()
    run_candidate_generation(
        arguments.dataset_root.resolve(),
        arguments.gps_feature_table.resolve(),
        arguments.candidate_output.resolve(),
        arguments.summary_output.resolve(),
        arguments.search_radius_meters,
        arguments.maximum_candidates,
    )


if __name__ == "__main__":
    main()
