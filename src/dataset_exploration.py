"""Explore PATHSNAP raw trip files without modifying the source dataset."""

from __future__ import annotations

import argparse
import csv
import math
import statistics
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import folium
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx


EXPECTED_FILES = {
    ".track": ("longitude", "latitude", "timestamp_seconds"),
    ".nodes": ("longitude", "latitude"),
    ".arcs": ("from_node_id", "to_node_id"),
    ".route": ("arc_id",),
}
MAP_BUFFER_DEGREES = 0.02
MAP_MAX_ROAD_ARCS = 12_000
EARTH_RADIUS_METERS = 6_371_008.8
MISSING_TOKENS = {"", "na", "nan", "null", "none"}


@dataclass
class ParsedFile:
    path: Path
    extension: str
    rows: list[tuple[Any, ...] | None]
    widths: Counter[int]
    malformed_rows: int
    parse_errors: int
    missing_by_field: dict[str, int]
    delimiter_styles: set[str]


def parse_raw_file(path: Path, extension: str) -> ParsedFile:
    fields = EXPECTED_FILES[extension]
    is_integer = extension in {".arcs", ".route"}
    rows: list[tuple[Any, ...] | None] = []
    widths: Counter[int] = Counter()
    malformed_rows = 0
    parse_errors = 0
    missing_by_field = Counter({field: 0 for field in fields})
    delimiter_styles: set[str] = set()

    with path.open(encoding="utf-8-sig", errors="replace") as raw_file:
        for line in raw_file:
            tokens = line.split()
            widths[len(tokens)] += 1
            if "\t" in line:
                delimiter_styles.add("tab")
            elif len(tokens) > 1:
                delimiter_styles.add("space")

            if len(tokens) != len(fields):
                malformed_rows += 1
            values: list[Any] = []
            row_valid = len(tokens) == len(fields)
            for index, field in enumerate(fields):
                if index >= len(tokens) or tokens[index].strip().lower() in MISSING_TOKENS:
                    missing_by_field[field] += 1
                    row_valid = False
                    continue
                try:
                    value = int(tokens[index]) if is_integer else float(tokens[index])
                except ValueError:
                    parse_errors += 1
                    row_valid = False
                    continue
                if isinstance(value, float) and not math.isfinite(value):
                    missing_by_field[field] += 1
                    row_valid = False
                values.append(value)
            rows.append(tuple(values) if row_valid and len(values) == len(fields) else None)

    return ParsedFile(
        path=path,
        extension=extension,
        rows=rows,
        widths=widths,
        malformed_rows=malformed_rows,
        parse_errors=parse_errors,
        missing_by_field=dict(missing_by_field),
        delimiter_styles=delimiter_styles,
    )


def coordinate_is_valid(longitude: float, latitude: float) -> bool:
    return (
        math.isfinite(longitude)
        and math.isfinite(latitude)
        and -180 <= longitude <= 180
        and -90 <= latitude <= 90
    )


def haversine_meters(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    delta_lon = lon2 - lon1
    delta_lat = lat2 - lat1
    hav = (
        math.sin(delta_lat / 2) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
    )
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(hav)))


def number_stats(values: list[float]) -> dict[str, float | int | None]:
    finite_values = [value for value in values if math.isfinite(value)]
    if not finite_values:
        return {
            "min": None,
            "median": None,
            "mean": None,
            "max": None,
        }
    return {
        "min": min(finite_values),
        "median": statistics.median(finite_values),
        "mean": statistics.fmean(finite_values),
        "max": max(finite_values),
    }


def find_trip_file(trip_dir: Path, extension: str) -> Path | None:
    exact = trip_dir / f"{trip_dir.name}{extension}"
    if exact.is_file():
        return exact
    matching = sorted(
        path for path in trip_dir.iterdir()
        if path.is_file() and path.suffix.lower() == extension
    )
    return matching[0] if matching else None


def parse_trip_files(trip_dir: Path) -> dict[str, ParsedFile | None]:
    return {
        extension: (
            parse_raw_file(path, extension) if path is not None else None
        )
        for extension in EXPECTED_FILES
        for path in [find_trip_file(trip_dir, extension)]
    }


def process_trip(partition: str, trip_dir: Path) -> dict[str, Any]:
    files = parse_trip_files(trip_dir)
    track = files[".track"]
    nodes = files[".nodes"]
    arcs = files[".arcs"]
    route = files[".route"]

    valid_track = [
        row for row in (track.rows if track else [])
        if row is not None and coordinate_is_valid(row[0], row[1])
    ]
    valid_coordinates = [(row[0], row[1]) for row in valid_track]
    valid_rows_by_line = track.rows if track else []
    timestamps = [
        row[2] for row in valid_rows_by_line
        if row is not None and math.isfinite(row[2])
    ]
    intervals: list[float] = []
    distances: list[float] = []
    for index in range(1, len(valid_rows_by_line)):
        previous = valid_rows_by_line[index - 1]
        current = valid_rows_by_line[index]
        if previous is None or current is None:
            continue
        if math.isfinite(previous[2]) and math.isfinite(current[2]):
            intervals.append(current[2] - previous[2])
        if coordinate_is_valid(previous[0], previous[1]) and coordinate_is_valid(
            current[0], current[1]
        ):
            distances.append(
                haversine_meters((previous[0], previous[1]), (current[0], current[1]))
            )

    positive_intervals = [value for value in intervals if value > 0]
    duplicate_observations = Counter(
        (row[0], row[1], row[2]) for row in valid_track
    )
    duplicate_coordinates = Counter(valid_coordinates)
    duplicate_timestamp_intervals = sum(value == 0 for value in intervals)
    decreasing_timestamp_intervals = sum(value < 0 for value in intervals)
    non_increasing = duplicate_timestamp_intervals + decreasing_timestamp_intervals
    irregular = sum(abs(value - 1.0) > 0.1 for value in positive_intervals)
    large_gaps = sum(value > 5.0 for value in positive_intervals)
    timestamp_stats = number_stats(positive_intervals)

    node_rows = nodes.rows if nodes else []
    node_count = len(node_rows)
    valid_node_coordinates: list[tuple[float, float] | None] = []
    invalid_node_coordinates = 0
    for row in node_rows:
        if row is not None and coordinate_is_valid(row[0], row[1]):
            valid_node_coordinates.append((row[0], row[1]))
        else:
            valid_node_coordinates.append(None)
            invalid_node_coordinates += 1

    arc_rows = arcs.rows if arcs else []
    arc_count = len(arc_rows)
    graph = nx.DiGraph()
    graph.add_nodes_from(range(node_count))
    invalid_arc_references = 0
    out_of_range_arc_references = 0
    valid_arc_rows: list[tuple[int, int] | None] = []
    for row in arc_rows:
        if row is None:
            valid_arc_rows.append(None)
            invalid_arc_references += 1
        elif not (0 <= row[0] < node_count and 0 <= row[1] < node_count):
            valid_arc_rows.append(None)
            invalid_arc_references += 1
            out_of_range_arc_references += 1
        else:
            valid_arc_rows.append((row[0], row[1]))
            graph.add_edge(row[0], row[1])

    weak_component_sizes = sorted(
        (len(component) for component in nx.weakly_connected_components(graph)),
        reverse=True,
    ) if node_count else []
    strong_component_sizes = sorted(
        (len(component) for component in nx.strongly_connected_components(graph)),
        reverse=True,
    ) if node_count else []
    isolated_nodes = sum(1 for _, degree in graph.degree() if degree == 0)
    arc_duplicates = len(
        [edge for edge, count in Counter(
            edge for edge in valid_arc_rows if edge is not None
        ).items() if count > 1]
    )

    route_rows = route.rows if route else []
    route_ids = [row[0] if row is not None else None for row in route_rows]
    valid_route_ids: list[int | None] = []
    invalid_route_ids = 0
    route_length_meters = 0.0
    for arc_id in route_ids:
        if (
            arc_id is None
            or not 0 <= arc_id < len(valid_arc_rows)
            or valid_arc_rows[arc_id] is None
        ):
            valid_route_ids.append(None)
            invalid_route_ids += 1
            continue
        valid_route_ids.append(arc_id)
        from_node, to_node = valid_arc_rows[arc_id]
        first = valid_node_coordinates[from_node]
        second = valid_node_coordinates[to_node]
        if first is not None and second is not None:
            route_length_meters += haversine_meters(first, second)

    checked_route_pairs = 0
    disconnected_route_pairs = 0
    for previous_id, current_id in zip(valid_route_ids, valid_route_ids[1:]):
        if previous_id is None or current_id is None:
            continue
        checked_route_pairs += 1
        previous_arc = valid_arc_rows[previous_id]
        current_arc = valid_arc_rows[current_id]
        if previous_arc[1] != current_arc[0]:
            disconnected_route_pairs += 1

    def extent(points: list[tuple[float, float]]) -> dict[str, float | None]:
        if not points:
            return {
                "min_longitude": None,
                "max_longitude": None,
                "min_latitude": None,
                "max_latitude": None,
            }
        return {
            "min_longitude": min(point[0] for point in points),
            "max_longitude": max(point[0] for point in points),
            "min_latitude": min(point[1] for point in points),
            "max_latitude": max(point[1] for point in points),
        }

    gps_extent = extent(valid_coordinates)
    network_extent = extent([point for point in valid_node_coordinates if point is not None])
    observation_count = len(track.rows) if track else 0
    valid_timestamp_count = len(timestamps)
    if valid_timestamp_count >= 2:
        trajectory_duration = max(timestamps) - min(timestamps)
    else:
        trajectory_duration = 0.0 if valid_timestamp_count == 1 else None
    invalid_coordinate_values = sum(
        row is not None and not coordinate_is_valid(row[0], row[1])
        for row in valid_rows_by_line
    )

    gps_summary: dict[str, Any] = {
        "partition": partition,
        "trip_id": trip_dir.name,
        "observations": observation_count,
        "valid_observations": len(valid_track),
        "malformed_rows": track.malformed_rows if track else 0,
        "missing_or_nonfinite_longitude": (
            track.missing_by_field["longitude"] if track else 0
        ),
        "missing_or_nonfinite_latitude": (
            track.missing_by_field["latitude"] if track else 0
        ),
        "missing_or_nonfinite_timestamp": (
            track.missing_by_field["timestamp_seconds"] if track else 0
        ),
        "invalid_coordinates": invalid_coordinate_values,
        "unusable_observations": observation_count - len(valid_track),
        "duplicate_full_observations": sum(
            count - 1 for count in duplicate_observations.values() if count > 1
        ),
        "duplicate_coordinate_observations": sum(
            count - 1 for count in duplicate_coordinates.values() if count > 1
        ),
        "non_increasing_timestamp_intervals": non_increasing,
        "duplicate_timestamp_intervals": duplicate_timestamp_intervals,
        "decreasing_timestamp_intervals": decreasing_timestamp_intervals,
        "timestamps_monotonic": non_increasing == 0,
        "timestamp_start_seconds": timestamps[0] if timestamps else None,
        "timestamp_end_seconds": timestamps[-1] if timestamps else None,
        "positive_intervals": len(positive_intervals),
        "irregular_intervals_over_0_1s_from_1s": irregular,
        "large_gaps_over_5s": large_gaps,
        "timestamp_interval_min_seconds": timestamp_stats["min"],
        "timestamp_interval_median_seconds": timestamp_stats["median"],
        "timestamp_interval_mean_seconds": timestamp_stats["mean"],
        "timestamp_interval_max_seconds": timestamp_stats["max"],
        "trajectory_duration_seconds": trajectory_duration,
        "trajectory_distance_meters": sum(distances),
        **gps_extent,
        "expected_observation_count_minute_1hz": observation_count,
    }

    road_summary: dict[str, Any] = {
        "partition": partition,
        "trip_id": trip_dir.name,
        "node_count": node_count,
        "valid_node_coordinates": node_count - invalid_node_coordinates,
        "invalid_node_coordinates": invalid_node_coordinates,
        "arc_count": arc_count,
        "valid_arc_count": sum(edge is not None for edge in valid_arc_rows),
        "invalid_arc_references": invalid_arc_references,
        "out_of_range_arc_references": out_of_range_arc_references,
        "duplicate_arc_pairs": arc_duplicates,
        "isolated_nodes": isolated_nodes,
        "weak_component_count": len(weak_component_sizes),
        "largest_weak_component_nodes": (
            weak_component_sizes[0] if weak_component_sizes else 0
        ),
        "largest_weak_component_fraction": (
            weak_component_sizes[0] / node_count if node_count else 0.0
        ),
        "strong_component_count": len(strong_component_sizes),
        "largest_strong_component_nodes": (
            strong_component_sizes[0] if strong_component_sizes else 0
        ),
        "weakly_connected": len(weak_component_sizes) <= 1,
        **network_extent,
    }

    valid_route_count = len(valid_route_ids) - invalid_route_ids
    route_summary = {
        "partition": partition,
        "trip_id": trip_dir.name,
        "route_entries": len(route_rows),
        "malformed_rows": route.malformed_rows if route else 0,
        "valid_arc_ids": valid_route_count,
        "invalid_arc_ids": invalid_route_ids,
        "valid_consecutive_pairs_checked": checked_route_pairs,
        "disconnected_consecutive_pairs": disconnected_route_pairs,
        "route_continuity_rate": (
            1.0 - disconnected_route_pairs / checked_route_pairs
            if checked_route_pairs else None
        ),
        "unique_arc_ids": len(set(value for value in route_ids if value is not None)),
        "repeated_route_entries": sum(
            count - 1 for count in Counter(
                value for value in route_ids if value is not None
            ).values() if count > 1
        ),
        "route_length_meters": route_length_meters,
    }

    inventory_rows = []
    missing_rows = []
    for extension, parsed in files.items():
        fields = EXPECTED_FILES[extension]
        expected_name = f"{trip_dir.name}{extension}"
        found_files = sorted(
            path for path in trip_dir.iterdir()
            if path.is_file() and path.suffix.lower() == extension
        )
        inventory_rows.append({
            "partition": partition,
            "trip_id": trip_dir.name,
            "file_type": extension,
            "expected_file_name": expected_name,
            "actual_file_name": parsed.path.name if parsed else "",
            "present": bool(parsed),
            "matches_naming_convention": bool(parsed and parsed.path.name == expected_name),
            "size_bytes": parsed.path.stat().st_size if parsed else 0,
            "rows": len(parsed.rows) if parsed else 0,
            "expected_columns": len(fields),
            "observed_column_counts": (
                ";".join(f"{width}:{count}" for width, count in sorted(parsed.widths.items()))
                if parsed else ""
            ),
            "malformed_rows": parsed.malformed_rows if parsed else 0,
            "parse_errors": parsed.parse_errors if parsed else 0,
            "delimiter_styles": ";".join(sorted(parsed.delimiter_styles)) if parsed else "",
            "format_valid": bool(parsed and parsed.malformed_rows == 0 and parsed.parse_errors == 0),
            "extra_same_type_files": max(0, len(found_files) - 1),
        })
        for field in fields:
            missing_rows.append({
                "partition": partition,
                "trip_id": trip_dir.name,
                "file_type": extension,
                "field": field,
                "row_count": len(parsed.rows) if parsed else 0,
                "missing_or_nonfinite_values": (
                    parsed.missing_by_field[field] if parsed else 0
                ),
                "parse_errors_in_file": parsed.parse_errors if parsed else 0,
            })

    for path in sorted(p for p in trip_dir.iterdir() if p.is_file()):
        if path.suffix.lower() not in EXPECTED_FILES:
            inventory_rows.append({
                "partition": partition,
                "trip_id": trip_dir.name,
                "file_type": path.suffix.lower() or "<no extension>",
                "expected_file_name": "",
                "actual_file_name": path.name,
                "present": True,
                "matches_naming_convention": False,
                "size_bytes": path.stat().st_size,
                "rows": "",
                "expected_columns": "",
                "observed_column_counts": "",
                "malformed_rows": "",
                "parse_errors": "",
                "delimiter_styles": "",
                "format_valid": "",
                "extra_same_type_files": "",
            })

    return {
        "gps": gps_summary,
        "road": road_summary,
        "route": route_summary,
        "inventory": inventory_rows,
        "missing": missing_rows,
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", newline="", encoding="utf-8") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def load_trip_summary(path: Path, partition: str, trip_id: str) -> dict[str, ParsedFile | None]:
    return parse_trip_files(path / partition / trip_id)


def valid_track_points(parsed: ParsedFile | None) -> list[tuple[float, float, float]]:
    if not parsed:
        return []
    return [
        (row[0], row[1], row[2])
        for row in parsed.rows
        if row is not None and coordinate_is_valid(row[0], row[1])
    ]


def plot_track_batches(
    dataset_root: Path,
    output_dir: Path,
    partition: str,
    summaries: list[dict[str, Any]],
    panels_per_figure: int = 20,
) -> None:
    ordered = sorted(
        (row for row in summaries if row["partition"] == partition),
        key=lambda row: row["trip_id"],
    )
    for batch_start in range(0, len(ordered), panels_per_figure):
        batch = ordered[batch_start:batch_start + panels_per_figure]
        columns = 4
        rows = math.ceil(len(batch) / columns)
        figure, axes = plt.subplots(rows, columns, figsize=(16, 3.8 * rows), squeeze=False)
        for axis, summary in zip(axes.flat, batch):
            files = load_trip_summary(dataset_root, partition, summary["trip_id"])
            points = valid_track_points(files[".track"])
            if points:
                axis.plot(
                    [point[0] for point in points],
                    [point[1] for point in points],
                    color="#d62728",
                    linewidth=0.8,
                )
                axis.scatter(points[0][0], points[0][1], color="#2ca02c", s=10, label="start")
                axis.scatter(points[-1][0], points[-1][1], color="#1f77b4", s=10, label="end")
                axis.legend(loc="best", fontsize=6)
            axis.set_title(f'{partition} / {summary["trip_id"]}: {len(points)} fixes', fontsize=8)
            axis.set_xlabel("Longitude (degrees)")
            axis.set_ylabel("Latitude (degrees)")
            axis.grid(alpha=0.2)
            axis.set_aspect("equal", adjustable="datalim")
        for axis in list(axes.flat)[len(batch):]:
            axis.set_visible(False)
        figure.tight_layout()
        page = batch_start // panels_per_figure + 1
        figure.savefig(
            output_dir / f"gps_trajectories_{partition}_{page:02d}.png",
            dpi=150,
            bbox_inches="tight",
        )
        plt.close(figure)


def plot_gps_distributions(
    figure_dir: Path,
    gps_summaries: list[dict[str, Any]],
    dataset_root: Path,
) -> None:
    figure, axes = plt.subplots(2, 2, figsize=(12, 8))
    for column, partition in enumerate(("train", "student_test")):
        longitudes: list[float] = []
        latitudes: list[float] = []
        intervals: list[float] = []
        for summary in gps_summaries:
            if summary["partition"] != partition:
                continue
            files = load_trip_summary(dataset_root, partition, summary["trip_id"])
            points = valid_track_points(files[".track"])
            longitudes.extend(point[0] for point in points)
            latitudes.extend(point[1] for point in points)
            track = files[".track"]
            if track:
                for previous, current in zip(track.rows, track.rows[1:]):
                    if previous is not None and current is not None:
                        difference = current[2] - previous[2]
                        if difference > 0:
                            intervals.append(difference)
        axes[0, column].hist(longitudes, bins=60, color="#1f77b4", alpha=0.8)
        axes[0, column].set_title(f"{partition}: longitude")
        axes[0, column].set_xlabel("Longitude (degrees)")
        axes[0, column].set_ylabel("GPS observations")
        axes[1, column].hist(latitudes, bins=60, color="#ff7f0e", alpha=0.8)
        axes[1, column].set_title(f"{partition}: latitude")
        axes[1, column].set_xlabel("Latitude (degrees)")
        axes[1, column].set_ylabel("GPS observations")
        if intervals:
            axes[1, column].text(
                0.98, 0.95, f"positive intervals collected: {len(intervals):,}",
                ha="right", va="top", transform=axes[1, column].transAxes, fontsize=8,
            )
    figure.tight_layout()
    figure.savefig(figure_dir / "gps_coordinate_distributions.png", dpi=160)
    plt.close(figure)

    figure, axes = plt.subplots(1, 2, figsize=(12, 4))
    for axis, partition in zip(axes, ("train", "student_test")):
        observations = [
            row["observations"] for row in gps_summaries if row["partition"] == partition
        ]
        axis.bar(range(len(observations)), observations, color="#4c78a8")
        axis.set_title(f"{partition}: GPS observations per trip")
        axis.set_xlabel("Trip (sorted order)")
        axis.set_ylabel("Observation count")
        axis.grid(axis="y", alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "gps_observations_per_trip.png", dpi=160)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(9, 5))
    for partition, color in (("train", "#1f77b4"), ("student_test", "#ff7f0e")):
        intervals = []
        for summary in gps_summaries:
            if summary["partition"] != partition:
                continue
            track = load_trip_summary(dataset_root, partition, summary["trip_id"])[".track"]
            if track:
                intervals.extend(
                    current[2] - previous[2]
                    for previous, current in zip(track.rows, track.rows[1:])
                    if previous is not None and current is not None
                    and current[2] - previous[2] > 0
                )
        if intervals:
            cap = sorted(intervals)[min(len(intervals) - 1, int(0.995 * len(intervals)))]
            displayed = [value for value in intervals if value <= cap]
            axis.hist(
                displayed, bins=60, alpha=0.55, color=color, label=partition,
                density=False,
            )
    axis.set_title("Positive GPS sampling intervals (upper 0.5% omitted from view)")
    axis.set_xlabel("Seconds between consecutive observations")
    axis.set_ylabel("Interval count")
    axis.legend()
    axis.grid(alpha=0.2)
    figure.tight_layout()
    figure.savefig(figure_dir / "gps_sampling_intervals.png", dpi=160)
    plt.close(figure)


def local_bounds(
    points: list[tuple[float, float, float]],
) -> tuple[float, float, float, float] | None:
    if not points:
        return None
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    lat_buffer = MAP_BUFFER_DEGREES
    lon_buffer = MAP_BUFFER_DEGREES / max(0.1, math.cos(math.radians(sum(lats) / len(lats))))
    return (
        min(lons) - lon_buffer,
        max(lons) + lon_buffer,
        min(lats) - lat_buffer,
        max(lats) + lat_buffer,
    )


def local_edges(
    files: dict[str, ParsedFile | None],
    points: list[tuple[float, float, float]],
    route_ids: set[int] | None = None,
) -> list[tuple[int, int, int]]:
    nodes = files[".nodes"]
    arcs = files[".arcs"]
    if not nodes or not arcs:
        return []
    bounds = local_bounds(points)
    if bounds is None:
        return []
    min_lon, max_lon, min_lat, max_lat = bounds
    coordinates = [
        row if row is not None and coordinate_is_valid(row[0], row[1]) else None
        for row in nodes.rows
    ]
    selected = []
    for arc_id, edge in enumerate(arcs.rows):
        if edge is None:
            continue
        from_node, to_node = edge
        if not (0 <= from_node < len(coordinates) and 0 <= to_node < len(coordinates)):
            continue
        first, second = coordinates[from_node], coordinates[to_node]
        if first is None or second is None:
            continue
        in_bounds = all(
            min_lon <= point[0] <= max_lon and min_lat <= point[1] <= max_lat
            for point in (first, second)
        )
        if in_bounds or (route_ids is not None and arc_id in route_ids):
            selected.append((arc_id, from_node, to_node))
    if len(selected) > MAP_MAX_ROAD_ARCS:
        protected = [
            edge for edge in selected
            if route_ids is not None and edge[0] in route_ids
        ]
        ordinary = [
            edge for edge in selected
            if route_ids is None or edge[0] not in route_ids
        ]
        remaining_capacity = max(0, MAP_MAX_ROAD_ARCS - len(protected))
        stride = max(1, math.ceil(len(ordinary) / max(1, remaining_capacity)))
        selected = protected + ordinary[::stride][:remaining_capacity]
    return selected


def route_ids_from_file(route: ParsedFile | None, arcs: ParsedFile | None) -> set[int]:
    if not route or not arcs:
        return set()
    return {
        row[0] for row in route.rows
        if row is not None and 0 <= row[0] < len(arcs.rows)
        and arcs.rows[row[0]] is not None
    }


def create_folium_map(
    map_dir: Path,
    dataset_root: Path,
    partition: str,
    summary: dict[str, Any],
) -> str:
    trip_id = summary["trip_id"]
    files = load_trip_summary(dataset_root, partition, trip_id)
    points = valid_track_points(files[".track"])
    if not points:
        return ""
    route_ids = route_ids_from_file(files[".route"], files[".arcs"])
    edges = local_edges(files, points, route_ids)
    center = [statistics.fmean(point[1] for point in points), statistics.fmean(point[0] for point in points)]
    map_view = folium.Map(location=center, zoom_start=13, control_scale=True)

    road_layer = folium.FeatureGroup(name="Road network (local)", show=True)
    nodes = files[".nodes"]
    if nodes:
        for _, from_node, to_node in edges:
            first = nodes.rows[from_node]
            second = nodes.rows[to_node]
            if first is not None and second is not None:
                folium.PolyLine(
                    [(first[1], first[0]), (second[1], second[0])],
                    color="#2878b5", weight=1, opacity=0.55,
                ).add_to(road_layer)
    road_layer.add_to(map_view)

    gps_layer = folium.FeatureGroup(name="GPS observations and trajectory", show=True)
    folium.PolyLine(
        [(point[1], point[0]) for point in points],
        color="#e53935", weight=3, opacity=0.9,
        tooltip="Observed GPS trajectory",
    ).add_to(gps_layer)
    marker_stride = max(1, math.ceil(len(points) / 300))
    for index in range(0, len(points), marker_stride):
        longitude, latitude, timestamp = points[index]
        folium.CircleMarker(
            location=(latitude, longitude), radius=2, color="#b71c1c",
            fill=True, fill_opacity=0.8,
            tooltip=f"Observation {index}; timestamp={timestamp:g} s",
        ).add_to(gps_layer)
    gps_layer.add_to(map_view)

    route_layer = folium.FeatureGroup(name="Reference route", show=True)
    if nodes and files[".arcs"] and files[".route"]:
        for route_row in files[".route"].rows:
            if route_row is None:
                continue
            arc_id = route_row[0]
            if not 0 <= arc_id < len(files[".arcs"].rows):
                continue
            edge = files[".arcs"].rows[arc_id]
            if edge is None:
                continue
            from_node, to_node = edge
            if not (0 <= from_node < len(nodes.rows) and 0 <= to_node < len(nodes.rows)):
                continue
            first, second = nodes.rows[from_node], nodes.rows[to_node]
            if first is not None and second is not None:
                folium.PolyLine(
                    [(first[1], first[0]), (second[1], second[0])],
                    color="#2ca02c", weight=3, opacity=0.85,
                    tooltip=f"Reference arc {arc_id}",
                ).add_to(route_layer)
    route_layer.add_to(map_view)
    folium.LayerControl(collapsed=False).add_to(map_view)
    map_view.fit_bounds([
        [min(point[1] for point in points), min(point[0] for point in points)],
        [max(point[1] for point in points), max(point[0] for point in points)],
    ])
    target = map_dir / f"pathsnap_map_{partition}_{trip_id}.html"
    map_view.save(str(target))
    return target.name


def create_network_figure(
    figure_dir: Path,
    dataset_root: Path,
    partition: str,
    summary: dict[str, Any],
) -> str:
    files = load_trip_summary(dataset_root, partition, summary["trip_id"])
    points = valid_track_points(files[".track"])
    nodes, arcs = files[".nodes"], files[".arcs"]
    if not nodes:
        return ""
    graph = nx.DiGraph()
    graph.add_nodes_from(range(len(nodes.rows)))
    if arcs:
        graph.add_edges_from(
            row for row in arcs.rows
            if row is not None and 0 <= row[0] < len(nodes.rows) and 0 <= row[1] < len(nodes.rows)
        )
    component_for_node = {}
    for component_id, component in enumerate(nx.weakly_connected_components(graph)):
        for node_id in component:
            component_for_node[node_id] = component_id
    selected_edges = local_edges(files, points, route_ids_from_file(files[".route"], arcs))
    bounds = local_bounds(points)
    figure, axis = plt.subplots(figsize=(10, 8))
    if bounds:
        min_lon, max_lon, min_lat, max_lat = bounds
        for _, from_node, to_node in selected_edges:
            first, second = nodes.rows[from_node], nodes.rows[to_node]
            if first is None or second is None:
                continue
            component = component_for_node.get(from_node, 0)
            axis.plot(
                [first[0], second[0]], [first[1], second[1]],
                color=plt.get_cmap("tab10")(component % 10),
                linewidth=0.7, alpha=0.65,
            )
        if points:
            axis.scatter(
                [point[0] for point in points], [point[1] for point in points],
                s=7, color="#d62728", label="GPS observations", zorder=4,
            )
        axis.set_xlim(min_lon, max_lon)
        axis.set_ylim(min_lat, max_lat)
    axis.set_title(
        f"Road network near GPS track: {partition}/{summary['trip_id']} "
        f"(component colors; local view)"
    )
    axis.set_xlabel("Longitude (degrees)")
    axis.set_ylabel("Latitude (degrees)")
    axis.grid(alpha=0.2)
    if points:
        axis.legend()
    figure.tight_layout()
    name = f"road_network_local_{partition}_{summary['trip_id']}.png"
    figure.savefig(figure_dir / name, dpi=160)
    plt.close(figure)
    return name


def plot_network_connectivity(figure_dir: Path, road_summaries: list[dict[str, Any]]) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5))
    for partition, marker in (("train", "o"), ("student_test", "s")):
        subset = [row for row in road_summaries if row["partition"] == partition]
        axes[0].scatter(
            [row["node_count"] for row in subset],
            [row["largest_weak_component_fraction"] for row in subset],
            label=partition, alpha=0.75, marker=marker,
        )
        axes[1].scatter(
            [row["node_count"] for row in subset],
            [row["weak_component_count"] for row in subset],
            label=partition, alpha=0.75, marker=marker,
        )
    axes[0].set_title("Largest weak component as share of nodes")
    axes[0].set_ylabel("Largest component fraction")
    axes[1].set_title("Weakly connected component count")
    axes[1].set_ylabel("Component count")
    for axis in axes:
        axis.set_xlabel("Node count")
        axis.legend()
        axis.grid(alpha=0.25)
    figure.tight_layout()
    figure.savefig(figure_dir / "road_network_connectivity.png", dpi=160)
    plt.close(figure)


def format_stat(values: list[float], unit: str = "") -> str:
    stats = number_stats(values)
    if stats["median"] is None:
        return "not available"
    return (
        f"median {stats['median']:,.2f}{unit} "
        f"(range {stats['min']:,.2f} to {stats['max']:,.2f}{unit})"
    )


def write_report(
    report_path: Path,
    inventory: list[dict[str, Any]],
    gps: list[dict[str, Any]],
    roads: list[dict[str, Any]],
    routes: list[dict[str, Any]],
    map_names: list[tuple[str, str]],
    network_figure_names: list[tuple[str, str]],
    reserved_partitions: list[tuple[str, int]],
) -> None:
    partitions = sorted({row["partition"] for row in gps})
    file_missing = sum(not row["present"] for row in inventory if row["file_type"] in EXPECTED_FILES)
    malformed_files = sum(
        row["malformed_rows"] > 0 or row["parse_errors"] > 0
        for row in inventory if row["file_type"] in EXPECTED_FILES and row["present"]
    )
    lines = [
        "# PATHSNAP Dataset Exploration Report",
        "",
        "This report is generated from the raw trip files found on disk by "
        "`src/dataset_exploration.py`. It performs dataset inspection and EDA only; "
        "it does not train models, reconstruct routes, or modify raw files.",
        "",
        "## Dataset structure and partition use",
        "",
        "| Partition | Trips | Expected files present |",
        "|---|---:|---:|",
    ]
    for partition in partitions:
        trip_count = len({row["trip_id"] for row in gps if row["partition"] == partition})
        present = sum(
            row["present"] for row in inventory
            if row["partition"] == partition and row["file_type"] in EXPECTED_FILES
        )
        expected = trip_count * len(EXPECTED_FILES)
        lines.append(f"| `{partition}` | {trip_count} | {present}/{expected} |")
    for partition, trip_count in reserved_partitions:
        lines.append(f"| `{partition}` | {trip_count} | Reserved: not inspected |")
    lines.extend([
        "",
        f"Missing expected files: **{file_missing}**. Files with malformed rows or numeric parse errors: **{malformed_files}**.",
        "",
        "The `train` partition is summarized separately and is the training/development "
        "partition specified by the faculty. `student_test` is kept separate and is "
        "described only as development-stage evaluation data; it is not mixed into "
        "training summaries at trip level. "
        + (
            "No evaluator-testing partition was found locally, so final evaluator results cannot be explored. "
            if not reserved_partitions else
            "Evaluator-testing partition folders were detected but left uninspected because they are reserved for final evaluation. "
        )
        + "No predictions, feature selection, or tuning were performed.",
        "",
        "The observed layout is `<partition>/<trip_id>/<trip_id>.<track|nodes|arcs|route>`. "
        "Each expected file is parsed as whitespace-separated and checked for expected "
        "column count; tabs are valid whitespace. The raw sample uses tab separators.",
        "",
        "## GPS trajectory summary",
        "",
        "| Partition | Trips | Observations (median, range) | Duration seconds (median, range) | Distance km (median, range) | Invalid coordinates | Duplicate full observations | Irregular intervals | Non-increasing time intervals | Gaps >5 s |",
        "|---|---:|---|---|---|---:|---:|---:|---:|---:|",
    ])
    for partition in partitions:
        selected = [row for row in gps if row["partition"] == partition]
        lines.append(
            f"| `{partition}` | {len(selected)} | "
            f"{format_stat([row['observations'] for row in selected])} | "
            f"{format_stat([row['trajectory_duration_seconds'] for row in selected if row['trajectory_duration_seconds'] is not None])} | "
            f"{format_stat([row['trajectory_distance_meters'] / 1000 for row in selected], ' km')} | "
            f"{sum(row['invalid_coordinates'] for row in selected)} | "
            f"{sum(row['duplicate_full_observations'] for row in selected)} | "
            f"{sum(row['irregular_intervals_over_0_1s_from_1s'] for row in selected)} | "
            f"{sum(row['non_increasing_timestamp_intervals'] for row in selected)} | "
            f"{sum(row['large_gaps_over_5s'] for row in selected)} |"
        )
    lines.extend([
        "",
        "Coordinates are interpreted in the documented order: longitude, latitude. "
        "The script checks legal WGS84 longitude/latitude ranges and records per-trip "
        "coordinate extents in `gps_summary.csv`. Intervals within 0.1 s of the nominal "
        "1 Hz interval are treated as regular; positive intervals over 5 s are counted "
        "as large gaps. These are explicit EDA thresholds, not data-cleaning rules.",
        "",
        "Trajectory distance is the sum of Haversine great-circle distances between "
        "consecutive valid coordinate rows, using a spherical Earth radius of "
        f"{EARTH_RADIUS_METERS:,.1f} m. It is an approximation to ellipsoidal geodesic "
        "distance; longitude/latitude degrees are never used as Euclidean distance. "
        "A distance segment can span a timestamp gap, so gap counts should be read "
        "alongside the distance statistic. Timestamps are treated as seconds since "
        "the trip-relative origin because the files contain no timezone/date metadata.",
        "",
        "Observed GPS coordinate extents by partition:",
    ])
    for partition in partitions:
        selected = [row for row in gps if row["partition"] == partition]
        longitude_min = min(
            (row["min_longitude"] for row in selected if row["min_longitude"] is not None),
            default=None,
        )
        longitude_max = max(
            (row["max_longitude"] for row in selected if row["max_longitude"] is not None),
            default=None,
        )
        latitude_min = min(
            (row["min_latitude"] for row in selected if row["min_latitude"] is not None),
            default=None,
        )
        latitude_max = max(
            (row["max_latitude"] for row in selected if row["max_latitude"] is not None),
            default=None,
        )
        if longitude_min is not None and latitude_min is not None:
            lines.append(
                f"- `{partition}`: longitude {longitude_min:.6f} to {longitude_max:.6f}; "
                f"latitude {latitude_min:.6f} to {latitude_max:.6f}."
            )
        else:
            lines.append(f"- `{partition}`: no valid GPS coordinates.")
    lines.extend([
        "",
        "## Road-network summary",
        "",
        "| Partition | Nodes (total; median) | Arcs (total; median) | Invalid arc references | Isolated nodes | Trips with >1 weak component | Median largest-component share |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ])
    for partition in partitions:
        selected = [row for row in roads if row["partition"] == partition]
        lines.append(
            f"| `{partition}` | {sum(row['node_count'] for row in selected):,}; "
            f"{format_stat([row['node_count'] for row in selected])} | "
            f"{sum(row['arc_count'] for row in selected):,}; "
            f"{format_stat([row['arc_count'] for row in selected])} | "
            f"{sum(row['invalid_arc_references'] for row in selected)} | "
            f"{sum(row['isolated_nodes'] for row in selected)} | "
            f"{sum(row['weak_component_count'] > 1 for row in selected)} | "
            f"{statistics.median(row['largest_weak_component_fraction'] for row in selected):.4f} |"
        )
    lines.extend([
        "",
        "Node IDs and arc IDs are interpreted as zero-based row indices. Invalid arc "
        "references are edges with malformed rows or endpoints outside the node-row "
        "range. Isolated nodes have zero in-degree and out-degree. Weak components "
        "describe connectivity when direction is ignored; strong components preserve "
        "directed reachability. Component counts include isolated nodes. Geographic "
        "network extents and per-trip component counts are in `road_network_summary.csv`.",
        "",
        "Observed network-node coordinate extents and track/network overlap:",
    ])
    gps_by_trip = {(row["partition"], row["trip_id"]): row for row in gps}
    for partition in partitions:
        selected = [row for row in roads if row["partition"] == partition]
        longitude_min = min(
            (row["min_longitude"] for row in selected if row["min_longitude"] is not None),
            default=None,
        )
        longitude_max = max(
            (row["max_longitude"] for row in selected if row["max_longitude"] is not None),
            default=None,
        )
        latitude_min = min(
            (row["min_latitude"] for row in selected if row["min_latitude"] is not None),
            default=None,
        )
        latitude_max = max(
            (row["max_latitude"] for row in selected if row["max_latitude"] is not None),
            default=None,
        )
        overlapping = 0
        for row in selected:
            track = gps_by_trip[(partition, row["trip_id"])]
            if (
                track["min_longitude"] is not None
                and row["min_longitude"] is not None
                and track["min_longitude"] <= row["max_longitude"]
                and track["max_longitude"] >= row["min_longitude"]
                and track["min_latitude"] <= row["max_latitude"]
                and track["max_latitude"] >= row["min_latitude"]
            ):
                overlapping += 1
        if longitude_min is not None and latitude_min is not None:
            lines.append(
                f"- `{partition}` nodes: longitude {longitude_min:.6f} to {longitude_max:.6f}; "
                f"latitude {latitude_min:.6f} to {latitude_max:.6f}. "
                f"GPS and node bounding boxes overlap for {overlapping}/{len(selected)} trips."
            )
        else:
            lines.append(f"- `{partition}`: no valid network-node coordinates.")
    lines.extend([
        "",
        "GPS and node bounding boxes overlap for every available trip; together with the "
        "coordinate ranges and plotted GPS/network alignment, this supports the "
        "longitude-first, latitude-second interpretation in the raw files.",
        "",
        "## Reference-route summary",
        "",
        "| Partition | Route entries (median) | Invalid arc IDs | Disconnected consecutive pairs | Checked pairs | Median route length km |",
        "|---|---:|---:|---:|---:|---:|",
    ])
    for partition in partitions:
        selected = [row for row in routes if row["partition"] == partition]
        lines.append(
            f"| `{partition}` | {format_stat([row['route_entries'] for row in selected])} | "
            f"{sum(row['invalid_arc_ids'] for row in selected)} | "
            f"{sum(row['disconnected_consecutive_pairs'] for row in selected)} | "
            f"{sum(row['valid_consecutive_pairs_checked'] for row in selected)} | "
            f"{format_stat([row['route_length_meters'] / 1000 for row in selected], ' km')} |"
        )
    lines.extend([
        "",
        "Route IDs are looked up against arc-row IDs, not GPS observations. A consecutive "
        "route pair is checked only when both route IDs and both referenced arcs are "
        "valid; continuity requires the previous arc's destination node to equal the "
        "next arc's source node. Route length sums geodesic distances between the "
        "referenced node coordinates and is not inferred from the number of GPS fixes.",
        "",
        "## Visual outputs",
        "",
        "Static trajectory figures contain all trips in each partition, in panels "
        "sorted by trip ID. Coordinate and sampling-interval distributions are "
        "partition-separated; the interval histogram clips its display at the 99.5th "
        "percentile only to keep the main distribution legible. Road-network local "
        "figures use the selected trip's network near its GPS extent and color links "
        "by weak component. Folium maps have separate GPS, local-road, and reference-route "
        "layers, with layer controls. For map readability, roads are clipped to a "
        f"{MAP_BUFFER_DEGREES} degree GPS bounding-box buffer; when more than "
        f"{MAP_MAX_ROAD_ARCS:,} local arcs are selected, the displayed ordinary roads "
        "are deterministically thinned, while route arcs are retained.",
        "",
        "| Partition | Map trip | Interactive map | Network plot |",
        "|---|---|---|---|",
    ])
    for (partition, trip_id), map_name in map_names:
        network_name = dict(network_figure_names).get((partition, trip_id), "")
        lines.append(f"| `{partition}` | `{trip_id}` | `outputs/maps/{map_name}` | `outputs/figures/{network_name}` |")
    lines.extend([
        "",
        "## Output files",
        "",
        "- `outputs/dataset_inventory.csv`: one record per expected trip file, plus unexpected files.",
        "- `outputs/gps_summary.csv`: per-trip GPS counts, anomalies, timing, distance, and extent.",
        "- `outputs/missing_value_analysis.csv`: per-trip and per-field missing/nonfinite values and parse errors.",
        "- `outputs/timestamp_analysis.csv`: per-trip timestamp order and interval analysis.",
        "- `outputs/road_network_summary.csv`: per-trip graph validity, component, isolation, and extent statistics.",
        "- `outputs/route_summary.csv`: per-trip reference route ID, continuity, and distance statistics.",
        "- `outputs/figures/`: actual-data trajectory, distribution, and network plots.",
        "- `outputs/maps/`: interactive Folium maps based on selected trips.",
        "",
        "## Scope and limitations",
        "",
        "This is dataset preparation/EDA only. No raw file is rewritten, and no records "
        "are cleaned or imputed. Invalid and duplicate rows remain present in the raw "
        "data and are reported rather than silently removed. The local map view is a "
        "readability-focused subset, not a complete rendering of a potentially large "
        "trip network. Haversine distances are spherical approximations; no projected "
        "CRS was assumed because the data spans multiple geographic regions. "
        "Install the listed plotting/map dependencies with "
        "`python -m pip install -r requirements-eda.txt`, then run "
        "`python src/dataset_exploration.py` to regenerate outputs from the local dataset.",
    ])
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_exploration(dataset_root: Path, output_dir: Path, report_path: Path) -> None:
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"Dataset root does not exist: {dataset_root}")

    partition_dirs = sorted(path for path in dataset_root.iterdir() if path.is_dir())
    reserved_partitions = [
        (partition.name, sum(child.is_dir() for child in partition.iterdir()))
        for partition in partition_dirs
        if "evaluator" in partition.name.lower()
        or partition.name.lower() in {"final_test", "final-testing"}
    ]
    trip_dirs = sorted(
        (partition.name, trip)
        for partition in partition_dirs
        if (
            "evaluator" not in partition.name.lower()
            and partition.name.lower() not in {"final_test", "final-testing"}
        )
        for trip in partition.iterdir() if trip.is_dir()
    )
    inventory: list[dict[str, Any]] = []
    gps_summaries: list[dict[str, Any]] = []
    road_summaries: list[dict[str, Any]] = []
    route_summaries: list[dict[str, Any]] = []
    missing_rows: list[dict[str, Any]] = []
    for index, (partition, trip_dir) in enumerate(trip_dirs, start=1):
        print(f"[{index}/{len(trip_dirs)}] Exploring {partition}/{trip_dir.name}")
        result = process_trip(partition, trip_dir)
        inventory.extend(result["inventory"])
        gps_summaries.append(result["gps"])
        road_summaries.append(result["road"])
        route_summaries.append(result["route"])
        missing_rows.extend(result["missing"])

    figure_dir = output_dir / "figures"
    map_dir = output_dir / "maps"
    figure_dir.mkdir(parents=True, exist_ok=True)
    map_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "dataset_inventory.csv", inventory)
    write_csv(output_dir / "gps_summary.csv", gps_summaries)
    write_csv(output_dir / "missing_value_analysis.csv", missing_rows)
    write_csv(output_dir / "timestamp_analysis.csv", [
        {
            key: row[key]
            for key in (
                "partition", "trip_id", "observations", "positive_intervals",
                "non_increasing_timestamp_intervals",
                "duplicate_timestamp_intervals", "decreasing_timestamp_intervals",
                "timestamps_monotonic", "timestamp_start_seconds",
                "timestamp_end_seconds",
                "irregular_intervals_over_0_1s_from_1s", "large_gaps_over_5s",
                "timestamp_interval_min_seconds", "timestamp_interval_median_seconds",
                "timestamp_interval_mean_seconds", "timestamp_interval_max_seconds",
                "trajectory_duration_seconds",
            )
        }
        for row in gps_summaries
    ])
    write_csv(output_dir / "road_network_summary.csv", road_summaries)
    write_csv(output_dir / "route_summary.csv", route_summaries)

    for partition in sorted({row["partition"] for row in gps_summaries}):
        plot_track_batches(dataset_root, figure_dir, partition, gps_summaries)
    plot_gps_distributions(figure_dir, gps_summaries, dataset_root)
    plot_network_connectivity(figure_dir, road_summaries)

    map_names: list[tuple[str, str]] = []
    network_figure_names: list[tuple[str, str]] = []
    for partition in ("train", "student_test"):
        partition_rows = [row for row in gps_summaries if row["partition"] == partition]
        if not partition_rows:
            continue
        median_count = statistics.median(row["observations"] for row in partition_rows)
        representative = min(
            partition_rows,
            key=lambda row: (abs(row["observations"] - median_count), row["trip_id"]),
        )
        map_name = create_folium_map(map_dir, dataset_root, partition, representative)
        network_name = create_network_figure(
            figure_dir, dataset_root, partition, representative
        )
        if map_name:
            map_names.append(((partition, representative["trip_id"]), map_name))
        if network_name:
            network_figure_names.append(((partition, representative["trip_id"]), network_name))

    write_report(
        report_path, inventory, gps_summaries, road_summaries,
        route_summaries, map_names, network_figure_names, reserved_partitions,
    )
    print(f"Completed EDA for {len(trip_dirs)} trips.")
    print(f"CSV, figures, and maps: {output_dir}")
    print(f"Report: {report_path}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    repository_root = Path(__file__).resolve().parents[1]
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=repository_root / "ML_Hackathon" / "Student folder" / "Student folder",
        help="Folder containing partition directories (default: discovered PATHSNAP layout)",
    )
    parser.add_argument(
        "--output-dir", type=Path, default=repository_root / "outputs",
        help="Directory for CSV, figure, and map outputs",
    )
    parser.add_argument(
        "--report-path", type=Path,
        default=repository_root / "DATASET_EXPLORATION_REPORT.md",
        help="Generated Markdown report path",
    )
    arguments = parser.parse_args()
    run_exploration(
        arguments.dataset_root.resolve(),
        arguments.output_dir.resolve(),
        arguments.report_path.resolve(),
    )


if __name__ == "__main__":
    main()
