"""Train an XGBoost arc classifier, decode routes with Viterbi, and map results.

The script expects the cleaned GPS sample and fold CSVs produced by the
PATHSNAP preprocessing workflow. Candidate arcs and training labels are derived
from each trip's .nodes, .arcs, and .route files; no prebuilt candidate tables
are required.

Example:
    python xgb_viterbi_pipeline.py
    python xgb_viterbi_pipeline.py --dataset-root "ML_Hackathon\\Student folder\\Student folder"
"""

from __future__ import annotations

import argparse
import csv
import heapq
import math
import sys
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Sequence


EARTH_RADIUS_METERS = 6_371_008.8
FEATURE_NAMES = (
    "speed_meters_per_second",
    "latitude",
    "longitude",
    "step_distance_meters",
    "bearing_degrees",
    "gps_to_arc_distance_meters",
    "distance_to_arc_start_meters",
    "distance_to_arc_end_meters",
    "arc_length_meters",
    "candidate_rank",
    "bearing_difference_degrees",
    "fraction_along_arc",
    "delta_time_seconds",
)


@dataclass(frozen=True)
class GPSFix:
    partition: str
    trip_id: str
    observation_index: int
    longitude: float
    latitude: float
    timestamp_seconds: float
    delta_time_seconds: float
    step_distance_meters: float
    speed_meters_per_second: float
    bearing_degrees: float


@dataclass(frozen=True)
class ArcCandidate:
    arc_id: int
    from_node: int
    to_node: int
    distance_meters: float
    distance_to_start_meters: float
    distance_to_end_meters: float
    length_meters: float
    fraction_along_arc: float
    bearing_degrees: float
    closest_longitude: float
    closest_latitude: float
    rank: int = 0
    emission_probability: float = 0.0


@dataclass
class TripData:
    partition: str
    trip_id: str
    fixes: list[GPSFix]
    nodes: list[tuple[float, float] | None]
    arcs: list[tuple[int, int] | None]
    route_arc_ids: list[int]
    candidates: list[list[ArcCandidate]]
    labels: list[list[int | None]]


def _require_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    """Import optional ML, spatial-index, and mapping dependencies on demand."""
    missing: list[str] = []
    try:
        import numpy as np
    except ImportError:
        np = None
        missing.append("numpy")
    try:
        from scipy.spatial import cKDTree
    except ImportError:
        cKDTree = None
        missing.append("scipy")
    try:
        from sklearn.metrics import (
            accuracy_score,
            f1_score,
            log_loss,
            precision_score,
            recall_score,
        )
    except ImportError:
        accuracy_score = f1_score = log_loss = None
        precision_score = recall_score = None
        missing.append("scikit-learn")
    try:
        from xgboost import XGBClassifier
    except ImportError:
        XGBClassifier = None
        missing.append("xgboost")
    try:
        import folium
    except ImportError:
        folium = None
        missing.append("folium")
    if missing:
        packages = ", ".join(sorted(set(missing)))
        raise RuntimeError(
            "Missing required Python packages: "
            f"{packages}. Install them with: python -m pip install numpy scipy "
            "scikit-learn xgboost folium"
        )
    return (
        np,
        cKDTree,
        XGBClassifier,
        {
            "accuracy": accuracy_score,
            "precision": precision_score,
            "recall": recall_score,
            "f1": f1_score,
            "log_loss": log_loss,
        },
        folium,
    )


def _finite_float(value: str, field: str, row_number: int) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as error:
        raise ValueError(f"Invalid {field} at CSV row {row_number}: {value!r}") from error
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {field} at CSV row {row_number}: {value!r}")
    return result


def load_gps_samples(path: Path) -> dict[tuple[str, str], list[GPSFix]]:
    """Read, validate, and sort GPS samples into partition/trip groups."""
    if not path.is_file():
        raise FileNotFoundError(f"Cleaned GPS samples CSV not found: {path}")
    required = {
        "partition", "trip_id", "observation_index", "longitude", "latitude",
        "timestamp_seconds", "delta_time_seconds", "step_distance_meters",
        "speed_meters_per_second",
    }
    grouped: dict[tuple[str, str], list[GPSFix]] = defaultdict(list)
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                observation_index = int(row["observation_index"])
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"Invalid observation_index at CSV row {row_number}: "
                    f"{row.get('observation_index')!r}"
                ) from error
            longitude = _finite_float(row["longitude"], "longitude", row_number)
            latitude = _finite_float(row["latitude"], "latitude", row_number)
            if not -180 <= longitude <= 180 or not -90 <= latitude <= 90:
                raise ValueError(f"Out-of-range coordinates at CSV row {row_number}")
            timestamp = _finite_float(row["timestamp_seconds"], "timestamp_seconds", row_number)
            delta_time = _finite_float(row["delta_time_seconds"], "delta_time_seconds", row_number)
            distance = _finite_float(row["step_distance_meters"], "step_distance_meters", row_number)
            speed = _finite_float(row["speed_meters_per_second"], "speed_meters_per_second", row_number)
            if observation_index < 0 or distance < 0 or speed < 0 or delta_time < 0:
                raise ValueError(f"Negative index, distance, speed, or time at CSV row {row_number}")
            partition, trip_id = row["partition"].strip(), row["trip_id"].strip()
            if not partition or not trip_id:
                raise ValueError(f"Empty partition or trip_id at CSV row {row_number}")
            grouped[(partition, trip_id)].append(
                GPSFix(
                    partition=partition,
                    trip_id=trip_id,
                    observation_index=observation_index,
                    longitude=longitude,
                    latitude=latitude,
                    timestamp_seconds=timestamp,
                    delta_time_seconds=delta_time,
                    step_distance_meters=distance,
                    speed_meters_per_second=speed,
                    bearing_degrees=0.0,
                )
            )

    for key, fixes in grouped.items():
        fixes.sort(key=lambda fix: fix.observation_index)
        indices = [fix.observation_index for fix in fixes]
        if len(set(indices)) != len(indices):
            raise ValueError(f"Duplicate observation_index values in trip {key[0]}/{key[1]}")
        with_bearings: list[GPSFix] = []
        previous: GPSFix | None = None
        for fix in fixes:
            bearing = (
                _bearing(
                    (previous.longitude, previous.latitude),
                    (fix.longitude, fix.latitude),
                )
                if previous is not None
                and (fix.longitude != previous.longitude or fix.latitude != previous.latitude)
                else 0.0
            )
            with_bearings.append(
                GPSFix(
                    partition=fix.partition,
                    trip_id=fix.trip_id,
                    observation_index=fix.observation_index,
                    longitude=fix.longitude,
                    latitude=fix.latitude,
                    timestamp_seconds=fix.timestamp_seconds,
                    delta_time_seconds=fix.delta_time_seconds,
                    step_distance_meters=fix.step_distance_meters,
                    speed_meters_per_second=fix.speed_meters_per_second,
                    bearing_degrees=bearing,
                )
            )
            previous = fix
        grouped[key] = with_bearings
    return dict(grouped)


def load_folds(path: Path) -> dict[int, dict[str, set[str]]]:
    """Load the precomputed trip-level train/validation assignment."""
    if not path.is_file():
        raise FileNotFoundError(f"Cross-validation folds CSV not found: {path}")
    folds: dict[int, dict[str, set[str]]] = defaultdict(
        lambda: {"training": set(), "validation": set()}
    )
    with path.open("r", encoding="utf-8-sig", newline="") as source:
        reader = csv.DictReader(source)
        required = {"fold", "trip_id", "cv_role"}
        missing = required.difference(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {', '.join(sorted(missing))}")
        for row_number, row in enumerate(reader, start=2):
            try:
                fold = int(row["fold"])
            except (TypeError, ValueError) as error:
                raise ValueError(f"Invalid fold number at CSV row {row_number}") from error
            trip_id, role = row["trip_id"].strip(), row["cv_role"].strip().lower()
            if fold < 1 or not trip_id or role not in {"training", "validation"}:
                raise ValueError(f"Invalid fold assignment at CSV row {row_number}: {row}")
            if trip_id in folds[fold][role]:
                raise ValueError(f"Duplicate fold assignment at row {row_number}: {row}")
            folds[fold][role].add(trip_id)
    if not folds:
        raise ValueError(f"No fold assignments found in {path}")
    for fold, roles in folds.items():
        overlap = roles["training"].intersection(roles["validation"])
        if overlap:
            raise ValueError(f"Fold {fold} assigns trips to both roles: {sorted(overlap)}")
        if not roles["training"] or not roles["validation"]:
            raise ValueError(f"Fold {fold} must have training and validation trips")
    return dict(folds)


def _read_integer_pairs(path: Path) -> list[tuple[int, int] | None]:
    pairs: list[tuple[int, int] | None] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line_number, line in enumerate(source, start=1):
            tokens = line.split()
            if len(tokens) != 2:
                pairs.append(None)
                continue
            try:
                pairs.append((int(tokens[0]), int(tokens[1])))
            except ValueError:
                pairs.append(None)
    return pairs


def _load_nodes(path: Path) -> list[tuple[float, float] | None]:
    nodes: list[tuple[float, float] | None] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            tokens = line.split()
            if len(tokens) != 2:
                nodes.append(None)
                continue
            try:
                longitude, latitude = map(float, tokens)
            except ValueError:
                nodes.append(None)
                continue
            if (
                math.isfinite(longitude) and math.isfinite(latitude)
                and -180 <= longitude <= 180 and -90 <= latitude <= 90
            ):
                nodes.append((longitude, latitude))
            else:
                nodes.append(None)
    return nodes


def _load_route(path: Path) -> list[int]:
    route: list[int] = []
    with path.open("r", encoding="utf-8-sig", errors="replace") as source:
        for line in source:
            tokens = line.split()
            if len(tokens) == 1:
                try:
                    route.append(int(tokens[0]))
                except ValueError:
                    continue
    return route


def _haversine(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    delta_lon, delta_lat = lon2 - lon1, lat2 - lat1
    value = (
        math.sin(delta_lat / 2.0) ** 2
        + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(value)))


def _bearing(first: tuple[float, float], second: tuple[float, float]) -> float:
    lon1, lat1 = map(math.radians, first)
    lon2, lat2 = map(math.radians, second)
    delta_lon = lon2 - lon1
    y = math.sin(delta_lon) * math.cos(lat2)
    x = (
        math.cos(lat1) * math.sin(lat2)
        - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
    )
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def _bearing_difference(first: float, second: float) -> float:
    difference = abs(first - second) % 360.0
    return min(difference, 360.0 - difference)


def _projected_xy(
    longitude: float, latitude: float, origin_latitude: float
) -> tuple[float, float]:
    cosine = max(math.cos(math.radians(origin_latitude)), 1e-6)
    return (
        math.radians(longitude) * EARTH_RADIUS_METERS * cosine,
        math.radians(latitude) * EARTH_RADIUS_METERS,
    )


def _candidate_on_segment(
    fix: GPSFix,
    arc_id: int,
    from_node: int,
    to_node: int,
    nodes: Sequence[tuple[float, float] | None],
) -> ArcCandidate:
    first, second = nodes[from_node], nodes[to_node]
    assert first is not None and second is not None
    origin_latitude = (fix.latitude + first[1] + second[1]) / 3.0
    px, py = _projected_xy(fix.longitude, fix.latitude, origin_latitude)
    ax, ay = _projected_xy(first[0], first[1], origin_latitude)
    bx, by = _projected_xy(second[0], second[1], origin_latitude)
    dx, dy = bx - ax, by - ay
    squared_length = dx * dx + dy * dy
    fraction = (
        max(0.0, min(1.0, ((px - ax) * dx + (py - ay) * dy) / squared_length))
        if squared_length > 1e-12
        else 0.0
    )
    closest_x, closest_y = ax + fraction * dx, ay + fraction * dy
    distance = math.hypot(px - closest_x, py - closest_y)
    start_distance = math.hypot(px - ax, py - ay)
    end_distance = math.hypot(px - bx, py - by)
    length = _haversine(first, second)
    return ArcCandidate(
        arc_id=arc_id,
        from_node=from_node,
        to_node=to_node,
        distance_meters=distance,
        distance_to_start_meters=start_distance,
        distance_to_end_meters=end_distance,
        length_meters=length,
        fraction_along_arc=fraction,
        bearing_degrees=_bearing(first, second),
        closest_longitude=first[0] + fraction * (second[0] - first[0]),
        closest_latitude=first[1] + fraction * (second[1] - first[1]),
    )


def _candidate_features(fix: GPSFix, candidate: ArcCandidate) -> list[float]:
    return [
        fix.speed_meters_per_second,
        fix.latitude,
        fix.longitude,
        fix.step_distance_meters,
        fix.bearing_degrees,
        candidate.distance_meters,
        candidate.distance_to_start_meters,
        candidate.distance_to_end_meters,
        candidate.length_meters,
        float(candidate.rank),
        _bearing_difference(fix.bearing_degrees, candidate.bearing_degrees),
        candidate.fraction_along_arc,
        fix.delta_time_seconds,
    ]


def _find_trip_files(dataset_root: Path, partition: str, trip_id: str) -> tuple[Path, Path, Path]:
    trip_dir = dataset_root / partition / trip_id
    paths = (
        trip_dir / f"{trip_id}.nodes",
        trip_dir / f"{trip_id}.arcs",
        trip_dir / f"{trip_id}.route",
    )
    missing = [path for path in paths if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            f"Missing required network/reference file(s) for {partition}/{trip_id}: "
            + ", ".join(str(path) for path in missing)
        )
    return paths


def _build_trip_data(
    partition: str,
    trip_id: str,
    fixes: list[GPSFix],
    dataset_root: Path,
    cKDTree: Any,
    candidate_radius_meters: float,
    maximum_candidates: int,
    alignment_radius_meters: float,
) -> TripData:
    nodes_path, arcs_path, route_path = _find_trip_files(dataset_root, partition, trip_id)
    nodes = _load_nodes(nodes_path)
    arcs = _read_integer_pairs(arcs_path)
    route_arc_ids = _load_route(route_path)

    valid_arc_entries: list[tuple[int, int, int]] = []
    for arc_id, endpoints in enumerate(arcs):
        if endpoints is None:
            continue
        from_node, to_node = endpoints
        if (
            0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)
            and nodes[from_node] is not None and nodes[to_node] is not None
        ):
            valid_arc_entries.append((arc_id, from_node, to_node))
    if not valid_arc_entries:
        raise ValueError(f"No valid road arcs in {arcs_path}")

    origin_latitude = sum(fix.latitude for fix in fixes) / len(fixes)
    midpoints = []
    half_lengths = []
    for arc_id, from_node, to_node in valid_arc_entries:
        first, second = nodes[from_node], nodes[to_node]
        assert first is not None and second is not None
        midpoint = ((first[0] + second[0]) / 2.0, (first[1] + second[1]) / 2.0)
        midpoints.append(_projected_xy(midpoint[0], midpoint[1], origin_latitude))
        length = _haversine(first, second)
        half_lengths.append(length / 2.0)
    tree = cKDTree(midpoints)
    search_radius = candidate_radius_meters + max(half_lengths, default=0.0)

    candidate_rows: list[list[ArcCandidate]] = []
    for fix in fixes:
        point_xy = _projected_xy(fix.longitude, fix.latitude, origin_latitude)
        nearby_indices = tree.query_ball_point(point_xy, r=search_radius)
        candidates = []
        for index in nearby_indices:
            arc_id, from_node, to_node = valid_arc_entries[index]
            candidate = _candidate_on_segment(fix, arc_id, from_node, to_node, nodes)
            if candidate.distance_meters <= candidate_radius_meters:
                candidates.append(candidate)
        candidates.sort(key=lambda item: (item.distance_meters, item.arc_id))
        candidates = candidates[:maximum_candidates]
        candidate_rows.append([
            ArcCandidate(**{**candidate.__dict__, "rank": rank})
            for rank, candidate in enumerate(candidates, start=1)
        ])

    valid_route_ids = [
        arc_id for arc_id in route_arc_ids
        if 0 <= arc_id < len(arcs) and arcs[arc_id] is not None
        and 0 <= arcs[arc_id][0] < len(nodes)
        and 0 <= arcs[arc_id][1] < len(nodes)
        and nodes[arcs[arc_id][0]] is not None and nodes[arcs[arc_id][1]] is not None
    ]
    route_geometry = [
        (arc_id, arcs[arc_id][0], arcs[arc_id][1])
        for arc_id in valid_route_ids
    ]
    labels: list[list[int | None]] = []
    for fix, candidates in zip(fixes, candidate_rows):
        nearest_route_arc: int | None = None
        nearest_distance = math.inf
        for route_arc_id, from_node, to_node in route_geometry:
            reference = _candidate_on_segment(fix, route_arc_id, from_node, to_node, nodes)
            if reference.distance_meters < nearest_distance:
                nearest_distance = reference.distance_meters
                nearest_route_arc = route_arc_id
        if nearest_route_arc is None or nearest_distance > alignment_radius_meters:
            labels.append([None] * len(candidates))
        else:
            # If candidate generation missed the reference arc, omit this
            # observation from supervised training rather than label every
            # retained candidate as a false negative.
            candidate_ids = {candidate.arc_id for candidate in candidates}
            if nearest_route_arc not in candidate_ids:
                labels.append([None] * len(candidates))
            else:
                labels.append([
                    int(candidate.arc_id == nearest_route_arc) for candidate in candidates
                ])

    return TripData(
        partition=partition,
        trip_id=trip_id,
        fixes=fixes,
        nodes=nodes,
        arcs=arcs,
        route_arc_ids=route_arc_ids,
        candidates=candidate_rows,
        labels=labels,
    )


def _build_all_trip_data(
    gps_by_trip: dict[tuple[str, str], list[GPSFix]],
    dataset_root: Path,
    cKDTree: Any,
    candidate_radius_meters: float,
    maximum_candidates: int,
    alignment_radius_meters: float,
) -> dict[tuple[str, str], TripData]:
    trips: dict[tuple[str, str], TripData] = {}
    for (partition, trip_id), fixes in sorted(gps_by_trip.items()):
        print(f"Preparing candidates and route labels: {partition}/{trip_id}")
        trips[(partition, trip_id)] = _build_trip_data(
            partition,
            trip_id,
            fixes,
            dataset_root,
            cKDTree,
            candidate_radius_meters,
            maximum_candidates,
            alignment_radius_meters,
        )
    return trips


def _collect_training_examples(
    trips: dict[tuple[str, str], TripData],
    trip_ids: Iterable[str],
    partition: str = "train",
) -> tuple[list[list[float]], list[int], list[tuple[str, int, int]]]:
    features: list[list[float]] = []
    labels: list[int] = []
    identifiers: list[tuple[str, int, int]] = []
    selected_ids = set(trip_ids)
    for (trip_partition, trip_id), trip in sorted(trips.items()):
        if trip_partition != partition or trip_id not in selected_ids:
            continue
        for fix, candidates, candidate_labels in zip(trip.fixes, trip.candidates, trip.labels):
            for candidate, label in zip(candidates, candidate_labels):
                if label is not None:
                    features.append(_candidate_features(fix, candidate))
                    labels.append(label)
                    identifiers.append((trip_id, fix.observation_index, candidate.arc_id))
    if not features:
        raise ValueError(
            "No labeled candidate examples were found. Check route files, candidate "
            "radius, alignment radius, and that cleaned samples overlap the training trips."
        )
    return features, labels, identifiers


def _new_classifier(XGBClassifier: Any) -> Any:
    # This is a binary candidate-vs-reference-arc classifier, not an arc-ID
    # multiclass model; its positive-class score becomes the HMM emission.
    return XGBClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=6,
        random_state=42,
        objective="binary:logistic",
        eval_metric="logloss",
        n_jobs=-1,
        tree_method="hist",
    )


def _fit_classifier(
    XGBClassifier: Any,
    features: list[list[float]],
    labels: list[int],
    np: Any,
) -> Any:
    if set(labels) != {0, 1}:
        raise ValueError(
            "XGBoost training data must contain both positive and negative candidate "
            "labels; verify route alignment and fold coverage."
        )
    model = _new_classifier(XGBClassifier)
    model.fit(np.asarray(features, dtype=np.float32), np.asarray(labels, dtype=np.int32))
    return model


def _positive_probability(model: Any, feature_rows: list[list[float]], np: Any) -> list[float]:
    if not feature_rows:
        return []
    class_positions = list(model.classes_)
    if 1 not in class_positions:
        raise ValueError("The fitted classifier does not contain the positive class.")
    position = class_positions.index(1)
    probabilities = model.predict_proba(np.asarray(feature_rows, dtype=np.float32))[:, position]
    return [float(value) for value in probabilities]


def _evaluate_cross_validation(
    trips: dict[tuple[str, str], TripData],
    folds: dict[int, dict[str, set[str]]],
    XGBClassifier: Any,
    metrics: dict[str, Any],
    np: Any,
) -> list[dict[str, float | str]]:
    all_truth: list[int] = []
    all_probabilities: list[float] = []
    fold_metrics: list[dict[str, float | str]] = []
    for fold, roles in sorted(folds.items()):
        train_features, train_labels, _ = _collect_training_examples(
            trips, roles["training"], partition="train"
        )
        validation_features, validation_labels, _ = _collect_training_examples(
            trips, roles["validation"], partition="train"
        )
        model = _fit_classifier(XGBClassifier, train_features, train_labels, np)
        probabilities = _positive_probability(model, validation_features, np)
        predictions = [int(value >= 0.5) for value in probabilities]
        result = _metric_row(
            fold,
            validation_labels,
            predictions,
            probabilities,
            metrics,
        )
        fold_metrics.append(result)
        all_truth.extend(validation_labels)
        all_probabilities.extend(probabilities)
        print(
            f"Fold {fold}: accuracy={result['accuracy']:.4f}, "
            f"precision={result['precision']:.4f}, recall={result['recall']:.4f}, "
            f"F1={result['f1']:.4f}, log_loss={result['log_loss']:.4f}"
        )

    all_predictions = [int(value >= 0.5) for value in all_probabilities]
    overall = _metric_row(None, all_truth, all_predictions, all_probabilities, metrics)
    fold_metrics.append(overall)
    print(
        "OOF aggregate: "
        f"accuracy={overall['accuracy']:.4f}, precision={overall['precision']:.4f}, "
        f"recall={overall['recall']:.4f}, F1={overall['f1']:.4f}, "
        f"log_loss={overall['log_loss']:.4f}"
    )
    return fold_metrics


def _metric_row(
    fold: int | None,
    truth: list[int],
    predictions: list[int],
    probabilities: list[float],
    metrics: dict[str, Any],
) -> dict[str, float | str]:
    return {
        "fold": str(fold) if fold is not None else "overall_oof",
        "accuracy": float(metrics["accuracy"](truth, predictions)),
        "precision": float(metrics["precision"](truth, predictions, zero_division=0)),
        "recall": float(metrics["recall"](truth, predictions, zero_division=0)),
        "f1": float(metrics["f1"](truth, predictions, zero_division=0)),
        "log_loss": float(metrics["log_loss"](truth, probabilities, labels=[0, 1])),
    }


def _write_metrics(path: Path, rows: list[dict[str, float | str]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(
            output,
            fieldnames=["fold", "accuracy", "precision", "recall", "f1", "log_loss"],
        )
        writer.writeheader()
        writer.writerows(rows)


def _predict_trip_candidates(trip: TripData, model: Any, np: Any) -> None:
    for fix, candidates in zip(trip.fixes, trip.candidates):
        feature_rows = [_candidate_features(fix, candidate) for candidate in candidates]
        raw_probabilities = _positive_probability(model, feature_rows, np)
        total = sum(max(value, 1e-12) for value in raw_probabilities)
        for index, candidate in enumerate(candidates):
            probability = max(raw_probabilities[index], 1e-12) / total if total else 1.0 / len(candidates)
            candidates[index] = ArcCandidate(
                **{**candidate.__dict__, "emission_probability": probability}
            )


def _network_adjacency(
    nodes: Sequence[tuple[float, float] | None],
    arcs: Sequence[tuple[int, int] | None],
) -> dict[int, list[tuple[int, float]]]:
    adjacency: dict[int, list[tuple[int, float]]] = defaultdict(list)
    for endpoints in arcs:
        if endpoints is None:
            continue
        from_node, to_node = endpoints
        if (
            0 <= from_node < len(nodes) and 0 <= to_node < len(nodes)
            and nodes[from_node] is not None and nodes[to_node] is not None
        ):
            adjacency[from_node].append((to_node, _haversine(nodes[from_node], nodes[to_node])))
    return adjacency


def _network_distances_from(
    start_node: int,
    adjacency: dict[int, list[tuple[int, float]]],
    maximum_distance_meters: float,
) -> dict[int, float]:
    """Dijkstra distances over directed .arcs, bounded to the useful route range."""
    distances = {start_node: 0.0}
    queue = [(0.0, start_node)]
    while queue:
        distance, node = heapq.heappop(queue)
        if distance != distances.get(node) or distance > maximum_distance_meters:
            continue
        for next_node, edge_length in adjacency.get(node, ()):
            proposed = distance + edge_length
            if proposed <= maximum_distance_meters and proposed < distances.get(next_node, math.inf):
                distances[next_node] = proposed
                heapq.heappush(queue, (proposed, next_node))
    return distances


def _transition_log_probability(
    previous: ArcCandidate,
    current: ArcCandidate,
    fix_distance_meters: float,
    network_distances: dict[int, float],
    transition_sigma_meters: float,
) -> float:
    # For different arcs, only a directed path through the .arcs graph is
    # legal. Same-arc transitions use the projected along-arc displacement.
    if previous.arc_id == current.arc_id:
        route_distance = abs(current.fraction_along_arc - previous.fraction_along_arc) * previous.length_meters
    else:
        middle_distance = network_distances.get(current.from_node)
        if middle_distance is None:
            return -math.inf
        route_distance = (
            max(0.0, 1.0 - previous.fraction_along_arc) * previous.length_meters
            + middle_distance
            + current.fraction_along_arc * current.length_meters
        )
    mismatch = abs(route_distance - fix_distance_meters)
    return -mismatch / max(transition_sigma_meters, 1e-6)


def viterbi_decode(
    trip: TripData,
    transition_sigma_meters: float = 50.0,
    disconnected_transition_penalty: float = 20.0,
) -> tuple[list[ArcCandidate], int]:
    """Decode the most likely connected sequence of candidate directed arcs.

    If no connected transition exists at a particular observation boundary,
    the decoder starts a penalized segment there and returns its break count.
    """
    if not trip.candidates or any(not states for states in trip.candidates):
        raise ValueError(f"Cannot decode {trip.partition}/{trip.trip_id}: an observation has no candidates")
    scores = [math.log(max(state.emission_probability, 1e-12)) for state in trip.candidates[0]]
    backpointers: list[list[int]] = [[-1] * len(trip.candidates[0])]
    disconnected_boundaries = 0
    adjacency = _network_adjacency(trip.nodes, trip.arcs)
    maximum_route_distance = max(
        (fix.step_distance_meters for fix in trip.fixes[1:]),
        default=0.0,
    ) + 10.0 * transition_sigma_meters
    distance_cache: dict[int, dict[int, float]] = {}

    for time_index in range(1, len(trip.candidates)):
        previous_states = trip.candidates[time_index - 1]
        current_states = trip.candidates[time_index]
        fix_distance = trip.fixes[time_index].step_distance_meters
        normalized_transitions: list[list[float]] = []
        for previous in previous_states:
            if previous.to_node not in distance_cache:
                distance_cache[previous.to_node] = _network_distances_from(
                    previous.to_node, adjacency, maximum_route_distance
                )
            network_distances = distance_cache[previous.to_node]
            row = [
                _transition_log_probability(
                    previous,
                    current,
                    fix_distance,
                    network_distances,
                    transition_sigma_meters,
                )
                for current in current_states
            ]
            finite_values = [value for value in row if math.isfinite(value)]
            if finite_values:
                maximum = max(finite_values)
                log_normalizer = maximum + math.log(
                    sum(math.exp(value - maximum) for value in finite_values)
                )
                row = [
                    value - log_normalizer if math.isfinite(value) else -math.inf
                    for value in row
                ]
            normalized_transitions.append(row)

        next_scores = [-math.inf] * len(current_states)
        next_backpointers = [-1] * len(current_states)
        for current_index, current in enumerate(current_states):
            emission_log = math.log(max(current.emission_probability, 1e-12))
            for previous_index, _previous in enumerate(previous_states):
                transition = normalized_transitions[previous_index][current_index]
                score = scores[previous_index] + transition + emission_log
                if score > next_scores[current_index]:
                    next_scores[current_index] = score
                    next_backpointers[current_index] = previous_index

        if all(not math.isfinite(score) for score in next_scores):
            disconnected_boundaries += 1
            best_previous = max(range(len(scores)), key=scores.__getitem__)
            best_score = scores[best_previous]
            next_scores = [
                best_score - disconnected_transition_penalty
                + math.log(max(state.emission_probability, 1e-12))
                for state in current_states
            ]
            next_backpointers = [best_previous] * len(current_states)
        scores = next_scores
        backpointers.append(next_backpointers)

    state_index = max(range(len(scores)), key=scores.__getitem__)
    decoded = [trip.candidates[-1][state_index]]
    for time_index in range(len(trip.candidates) - 1, 0, -1):
        state_index = backpointers[time_index][state_index]
        decoded.append(trip.candidates[time_index - 1][state_index])
    decoded.reverse()
    return decoded, disconnected_boundaries


def _route_coordinates(trip: TripData) -> list[tuple[float, float]]:
    coordinates: list[tuple[float, float]] = []
    for arc_id in trip.route_arc_ids:
        if not 0 <= arc_id < len(trip.arcs) or trip.arcs[arc_id] is None:
            continue
        from_node, to_node = trip.arcs[arc_id]
        if not (
            0 <= from_node < len(trip.nodes)
            and 0 <= to_node < len(trip.nodes)
        ):
            continue
        first, second = trip.nodes[from_node], trip.nodes[to_node]
        if first is None or second is None:
            continue
        if coordinates and coordinates[-1] == (first[1], first[0]):
            coordinates.append((second[1], second[0]))
        else:
            coordinates.extend(((first[1], first[0]), (second[1], second[0])))
    return coordinates


def _save_map(
    path: Path,
    trip: TripData,
    decoded: list[ArcCandidate],
    top_candidates: list[ArcCandidate],
    folium: Any,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    center = [
        sum(fix.latitude for fix in trip.fixes) / len(trip.fixes),
        sum(fix.longitude for fix in trip.fixes) / len(trip.fixes),
    ]
    map_view = folium.Map(location=center, zoom_start=14, control_scale=True)
    raw_points = [(fix.latitude, fix.longitude) for fix in trip.fixes]
    if len(raw_points) >= 2:
        folium.PolyLine(
            raw_points, color="red", weight=3, opacity=0.8, tooltip="GPS observations"
        ).add_to(map_view)

    top_points = [(item.closest_latitude, item.closest_longitude) for item in top_candidates]
    for fix, candidate in zip(trip.fixes, top_candidates):
        folium.CircleMarker(
            location=(candidate.closest_latitude, candidate.closest_longitude),
            radius=4,
            color="blue",
            fill=True,
            fill_opacity=0.8,
            tooltip=(
                f"Top XGBoost arc {candidate.arc_id}; "
                f"emission={candidate.emission_probability:.4f}; "
                f"GPS #{fix.observation_index}"
            ),
        ).add_to(map_view)
    if len(top_points) >= 2:
        folium.PolyLine(
            top_points, color="blue", weight=3, opacity=0.75,
            tooltip="XGBoost top candidate path",
        ).add_to(map_view)

    decoded_points = [(item.closest_latitude, item.closest_longitude) for item in decoded]
    if len(decoded_points) >= 2:
        folium.PolyLine(
            decoded_points, color="green", weight=5, opacity=0.85,
            tooltip="HMM / Viterbi decoded path",
        ).add_to(map_view)
    for index, point in enumerate(raw_points):
        folium.CircleMarker(
            location=point, radius=2, color="red", fill=True, fill_opacity=0.6,
            tooltip=f"GPS observation #{trip.fixes[index].observation_index}",
        ).add_to(map_view)

    reference_points = _route_coordinates(trip)
    if len(reference_points) >= 2:
        folium.PolyLine(
            reference_points, color="orange", weight=7, opacity=0.65,
            tooltip="Reference route (underlay)",
        ).add_to(map_view)
        folium.PolyLine(
            reference_points, color="black", weight=3, opacity=0.9,
            tooltip="Reference ground-truth route",
        ).add_to(map_view)
    folium.LayerControl().add_to(map_view)
    map_view.save(str(path))


def _save_predictions(
    path: Path,
    trips: dict[tuple[str, str], TripData],
    decoded_by_trip: dict[tuple[str, str], list[ArcCandidate]],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as output:
        fields = [
            "partition", "trip_id", "observation_index", "arc_id",
            "from_node_id", "to_node_id", "emission_probability",
            "viterbi_arc_id", "viterbi_from_node_id", "viterbi_to_node_id",
        ]
        writer = csv.DictWriter(output, fieldnames=fields)
        writer.writeheader()
        for key, trip in sorted(trips.items()):
            decoded = decoded_by_trip.get(key)
            if decoded is None:
                continue
            for fix, candidates, selected in zip(trip.fixes, trip.candidates, decoded):
                for candidate in candidates:
                    writer.writerow({
                        "partition": trip.partition,
                        "trip_id": trip.trip_id,
                        "observation_index": fix.observation_index,
                        "arc_id": candidate.arc_id,
                        "from_node_id": candidate.from_node,
                        "to_node_id": candidate.to_node,
                        "emission_probability": f"{candidate.emission_probability:.10g}",
                        "viterbi_arc_id": selected.arc_id,
                        "viterbi_from_node_id": selected.from_node,
                        "viterbi_to_node_id": selected.to_node,
                    })


def run_pipeline(arguments: argparse.Namespace) -> None:
    np, cKDTree, XGBClassifier, metrics, folium = _require_dependencies()
    gps_by_trip = load_gps_samples(arguments.gps_samples)
    folds = load_folds(arguments.folds)
    training_trip_ids = {
        trip_id for roles in folds.values()
        for role in ("training", "validation")
        for trip_id in roles[role]
    }
    absent_training = sorted(
        trip_id for trip_id in training_trip_ids
        if ("train", trip_id) not in gps_by_trip
    )
    if absent_training:
        raise ValueError(
            "The fold CSV references training trips absent from the cleaned GPS input: "
            + ", ".join(absent_training[:20])
            + (" ..." if len(absent_training) > 20 else "")
        )
    if not arguments.dataset_root.is_dir():
        raise FileNotFoundError(
            f"Raw trip directory not found: {arguments.dataset_root}. "
            "Place train/<trip_id>/<trip_id>.nodes, .arcs, and .route files there, "
            "or pass --dataset-root."
        )
    missing_raw = [
        trip_id for trip_id in sorted(training_trip_ids)
        if not (arguments.dataset_root / "train" / trip_id / f"{trip_id}.nodes").is_file()
        or not (arguments.dataset_root / "train" / trip_id / f"{trip_id}.arcs").is_file()
        or not (arguments.dataset_root / "train" / trip_id / f"{trip_id}.route").is_file()
    ]
    if missing_raw:
        raise FileNotFoundError(
            "Required raw network/reference files are missing for training trips: "
            + ", ".join(missing_raw[:20])
            + (" ..." if len(missing_raw) > 20 else "")
        )

    trips = _build_all_trip_data(
        gps_by_trip,
        arguments.dataset_root,
        cKDTree,
        arguments.candidate_radius_meters,
        arguments.maximum_candidates,
        arguments.alignment_radius_meters,
    )
    fold_metrics = _evaluate_cross_validation(
        trips, folds, XGBClassifier, metrics, np
    )
    _write_metrics(arguments.metrics_output, fold_metrics)

    final_features, final_labels, _ = _collect_training_examples(
        trips, training_trip_ids, partition="train"
    )
    final_model = _fit_classifier(XGBClassifier, final_features, final_labels, np)
    for trip in trips.values():
        _predict_trip_candidates(trip, final_model, np)

    selected_trips = {
        key: trip for key, trip in sorted(trips.items())
        if arguments.map_partition == "all" or trip.partition == arguments.map_partition
    }
    if not selected_trips:
        raise ValueError(
            f"No trips are available for map partition {arguments.map_partition!r}."
        )
    decoded_by_trip: dict[tuple[str, str], list[ArcCandidate]] = {}
    for key, trip in selected_trips.items():
        decoded, broken_transitions = viterbi_decode(
            trip,
            transition_sigma_meters=arguments.transition_sigma_meters,
            disconnected_transition_penalty=arguments.disconnected_transition_penalty,
        )
        decoded_by_trip[key] = decoded
        top_candidates = [
            max(candidates, key=lambda candidate: candidate.emission_probability)
            for candidates in trip.candidates
        ]
        map_path = arguments.maps_dir / f"{trip.partition}_{trip.trip_id}.html"
        _save_map(map_path, trip, decoded, top_candidates, folium)
        print(
            f"Saved {map_path} "
            f"(Viterbi disconnected boundaries: {broken_transitions})"
        )
    _save_predictions(arguments.predictions_output, selected_trips, decoded_by_trip)
    print(f"Validation metrics: {arguments.metrics_output}")
    print(f"Predictions: {arguments.predictions_output}")


def _positive_float(value: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0:
        raise argparse.ArgumentTypeError("value must be a finite positive number")
    return result


def _positive_integer(value: str) -> int:
    result = int(value)
    if result <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return result


def build_parser() -> argparse.ArgumentParser:
    repository_root = Path(__file__).resolve().parent
    project_root = repository_root / "ML_Hackathon"
    outputs = project_root / "outputs"
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--dataset-root",
        type=Path,
        default=project_root / "Student folder" / "Student folder",
        help="Directory containing train/<trip_id>/<trip_id>.nodes/.arcs/.route.",
    )
    parser.add_argument(
        "--gps-samples",
        type=Path,
        default=outputs / "cleaned_gps_samples.csv",
        help="Cleaned GPS sample CSV.",
    )
    parser.add_argument(
        "--folds",
        type=Path,
        default=outputs / "cross_validation_folds.csv",
        help="Trip-level cross-validation assignments CSV.",
    )
    parser.add_argument(
        "--metrics-output",
        type=Path,
        default=outputs / "xgb_viterbi_cv_metrics.csv",
    )
    parser.add_argument(
        "--predictions-output",
        type=Path,
        default=outputs / "xgb_viterbi_predictions.csv",
    )
    parser.add_argument(
        "--maps-dir",
        type=Path,
        default=outputs / "maps",
    )
    parser.add_argument(
        "--map-partition",
        choices=("student_test", "train", "all"),
        default="student_test",
        help="Trips to decode and map with the final model (default: student_test).",
    )
    parser.add_argument("--candidate-radius-meters", type=_positive_float, default=100.0)
    parser.add_argument("--alignment-radius-meters", type=_positive_float, default=100.0)
    parser.add_argument("--maximum-candidates", type=_positive_integer, default=8)
    parser.add_argument("--transition-sigma-meters", type=_positive_float, default=50.0)
    parser.add_argument(
        "--disconnected-transition-penalty",
        type=_positive_float,
        default=20.0,
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    parser = build_parser()
    arguments = parser.parse_args(argv)
    for name in (
        "dataset_root", "gps_samples", "folds", "metrics_output",
        "predictions_output", "maps_dir",
    ):
        setattr(arguments, name, getattr(arguments, name).expanduser().resolve())
    try:
        run_pipeline(arguments)
    except (FileNotFoundError, ValueError, RuntimeError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 2
    except OSError as error:
        print(f"ERROR: Could not read or write pipeline files: {error}", file=sys.stderr)
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
