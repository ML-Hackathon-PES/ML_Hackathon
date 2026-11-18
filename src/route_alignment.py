"""Spatially align reference route arcs to cleaned GPS observations."""

from __future__ import annotations

import csv
import math
from pathlib import Path


EARTH_RADIUS_METERS = 6_371_008.8


def point_segment_distance(
    point: tuple[float, float],
    first: tuple[float, float],
    second: tuple[float, float],
) -> float:
    origin_latitude = math.radians((point[1] + first[1] + second[1]) / 3.0)
    cos_latitude = max(math.cos(origin_latitude), 1e-6)
    px = math.radians(point[0]) * EARTH_RADIUS_METERS * cos_latitude
    py = math.radians(point[1]) * EARTH_RADIUS_METERS
    ax = math.radians(first[0]) * EARTH_RADIUS_METERS * cos_latitude
    ay = math.radians(first[1]) * EARTH_RADIUS_METERS
    bx = math.radians(second[0]) * EARTH_RADIUS_METERS * cos_latitude
    by = math.radians(second[1]) * EARTH_RADIUS_METERS
    dx, dy = bx - ax, by - ay
    squared_length = dx * dx + dy * dy
    if squared_length <= 1e-12:
        t = 0.0
    else:
        t = max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared_length))
    closest_x = ax + t * dx
    closest_y = ay + t * dy
    return math.hypot(px - closest_x, py - closest_y)


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
            nodes.append((lon, lat) if math.isfinite(lon) and math.isfinite(lat) else None)
    return nodes


def load_arcs(path: Path) -> list[tuple[int, int] | None]:
    arcs = []
    with path.open(encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            tokens = line.split()
            if len(tokens) != 2:
                arcs.append(None)
                continue
            try:
                from_node, to_node = map(int, tokens)
                arcs.append((from_node, to_node))
            except ValueError:
                arcs.append(None)
    return arcs


def run_alignment(
    dataset_root: Path,
    gps_feature_table: Path,
    candidate_table: Path,
    labels_output: Path,
    summary_output: Path,
    alignment_radius_meters: float,
) -> None:
    gps_rows = {}
    with gps_feature_table.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            gps_rows[(row["partition"], row["trip_id"], int(row["observation_index"]))] = row

    candidate_sets = {}
    with candidate_table.open(encoding="utf-8-sig", newline="") as source:
        for row in csv.DictReader(source):
            key = (row["partition"], row["trip_id"], int(row["observation_index"]))
            candidate_sets.setdefault(key, set()).add(int(row["arc_id"]))

    candidate_rows = []
    with candidate_table.open(encoding="utf-8-sig", newline="") as source:
        candidate_rows = list(csv.DictReader(source))

    summaries = []
    with labels_output.open("w", newline="", encoding="utf-8") as labels_file:
        writer = csv.DictWriter(labels_file, fieldnames=[
            "partition", "trip_id", "observation_index", "arc_id",
            "candidate_rank", "candidate_count", "gps_to_segment_distance_meters",
            "reference_arc_id", "reference_distance_meters", "label",
            "alignment_status",
        ])
        writer.writeheader()

        partitions = sorted(
            partition for partition in dataset_root.iterdir()
            if partition.is_dir()
            and partition.name in {"train", "student_test"}
        )
        for partition in partitions:
            for trip_dir in sorted(trip for trip in partition.iterdir() if trip.is_dir()):
                trip_id = trip_dir.name
                nodes = load_nodes(trip_dir / f"{trip_id}.nodes")
                arcs = load_arcs(trip_dir / f"{trip_id}.arcs")
                route_path = trip_dir / f"{trip_id}.route"
                route_ids = []
                with route_path.open(encoding="utf-8-sig", errors="replace") as source:
                    for line in source:
                        tokens = line.split()
                        if len(tokens) == 1:
                            try:
                                route_ids.append(int(tokens[0]))
                            except ValueError:
                                pass

                trip_candidate_rows = [
                    row for row in candidate_rows
                    if row["partition"] == partition.name and row["trip_id"] == trip_id
                ]
                trip_gps_rows = [
                    row for key, row in gps_rows.items()
                    if key[0] == partition.name and key[1] == trip_id
                ]
                trip_gps_rows.sort(key=lambda row: int(row["observation_index"]))

                nearest_by_observation = {}
                for gps in trip_gps_rows:
                    observation_index = int(gps["observation_index"])
                    point = (float(gps["longitude"]), float(gps["latitude"]))
                    nearest = None
                    nearest_distance = math.inf
                    for route_arc_id in route_ids:
                        if not 0 <= route_arc_id < len(arcs):
                            continue
                        edge = arcs[route_arc_id]
                        if edge is None:
                            continue
                        from_node, to_node = edge
                        if not (0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)):
                            continue
                        first = nodes[from_node]
                        second = nodes[to_node]
                        if first is None or second is None:
                            continue
                        distance = point_segment_distance(point, first, second)
                        if distance < nearest_distance:
                            nearest_distance = distance
                            nearest = route_arc_id
                    if nearest is not None and nearest_distance <= alignment_radius_meters:
                        nearest_by_observation[observation_index] = (nearest, nearest_distance)

                reference_arc_ids = {
                    route_arc_id for route_arc_id in route_ids
                    if 0 <= route_arc_id < len(arcs)
                }
                matched_reference_arcs = set()
                defensible_observations = 0
                positive_labels = 0
                negative_labels = 0
                unlabeled_observations = 0

                for candidate in trip_candidate_rows:
                    observation_index = int(candidate["observation_index"])
                    arc_id = int(candidate["arc_id"])
                    alignment = nearest_by_observation.get(observation_index)
                    if alignment is None:
                        unlabeled_observations += 1
                        reference_arc_id = ""
                        reference_distance = ""
                        label = ""
                        alignment_status = "unmatched_reference_route"
                    else:
                        reference_arc_id, reference_distance = alignment
                        matched_reference_arcs.add(reference_arc_id)
                        if arc_id == reference_arc_id:
                            label = 1
                            positive_labels += 1
                            alignment_status = "defensible_positive_or_negative"
                        else:
                            label = 0
                            negative_labels += 1
                            alignment_status = "defensible_positive_or_negative"
                        defensible_observations += 1

                    writer.writerow({
                        "partition": partition.name,
                        "trip_id": trip_id,
                        "observation_index": observation_index,
                        "arc_id": arc_id,
                        "candidate_rank": candidate["candidate_rank"],
                        "candidate_count": candidate["candidate_count"],
                        "gps_to_segment_distance_meters": candidate["gps_to_segment_distance_meters"],
                        "reference_arc_id": reference_arc_id,
                        "reference_distance_meters": reference_distance,
                        "label": label,
                        "alignment_status": alignment_status,
                    })

                candidate_reference_misses = sum(
                    1 for route_arc_id in reference_arc_ids
                    if route_arc_id not in matched_reference_arcs
                )
                summaries.append({
                    "partition": partition.name,
                    "trip_id": trip_id,
                    "route_entries": len(route_ids),
                    "unique_reference_arc_ids": len(reference_arc_ids),
                    "gps_observations": len(trip_gps_rows),
                    "defensively_aligned_observations": len({
                        observation_index for observation_index in nearest_by_observation
                    }),
                    "positive_candidate_labels": positive_labels,
                    "negative_candidate_labels": negative_labels,
                    "unlabeled_candidate_rows": unlabeled_observations,
                    "reference_arc_ids_not_matched_to_a_gps": candidate_reference_misses,
                    "candidate_miss_rate": (
                        sum(
                            1 for gps in trip_gps_rows
                            if int(gps["observation_index"]) not in nearest_by_observation
                        ) / len(trip_gps_rows)
                        if trip_gps_rows else 1.0
                    ),
                })

    summary_fields = [
        "partition", "trip_id", "route_entries", "unique_reference_arc_ids",
        "gps_observations", "defensively_aligned_observations",
        "positive_candidate_labels", "negative_candidate_labels",
        "unlabeled_candidate_rows", "reference_arc_ids_not_matched_to_a_gps",
        "candidate_miss_rate",
    ]
    with summary_output.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=summary_fields)
        writer.writeheader()
        writer.writerows(summaries)


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
        "--labels-output",
        type=Path,
        default=repository_root / "outputs" / "candidate_labels.csv",
    )
    parser.add_argument(
        "--summary-output",
        type=Path,
        default=repository_root / "outputs" / "route_alignment_summary.csv",
    )
    parser.add_argument("--alignment-radius-meters", type=float, default=100.0)
    arguments = parser.parse_args()
    run_alignment(
        arguments.dataset_root.resolve(),
        arguments.gps_feature_table.resolve(),
        arguments.candidate_table.resolve(),
        arguments.labels_output.resolve(),
        arguments.summary_output.resolve(),
        arguments.alignment_radius_meters,
    )
    print(f"Candidate labels written to {arguments.labels_output.resolve()}")


if __name__ == "__main__":
    main()
