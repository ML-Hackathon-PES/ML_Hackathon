"""Run route reconstruction on raw trip folders and print scores.

Each trip goes through GPS cleaning, candidate generation, emission scoring and
Viterbi decoding. The .route file is read only after decoding, to score the
result. Decoder settings come from outputs/hmm_viterbi/metrics.json and are not
changed by this script.

Usage (from the repository root):
    python src/evaluate_trips.py evaluator_test.zip
    python src/evaluate_trips.py "Student folder/Student folder/student_test"
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import tempfile
import time
import zipfile
from dataclasses import fields
from pathlib import Path

SOURCE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SOURCE_DIR))

from candidate_feature_engineering import angle_difference, bearing_degrees  # noqa: E402
from dataset_preprocessing import clean_track  # noqa: E402
from hmm_viterbi import (DecoderConfig, Observation, RoadNetwork, State,  # noqa: E402
                         assign_emissions, classifier_path, count_breaks, decode,
                         haversine, position_along_arc, score_path, summarize)
from route_alignment import load_arcs, load_nodes, point_segment_distance  # noqa: E402

REPO_ROOT = SOURCE_DIR.parent
EARTH_RADIUS_METERS = 6_371_008.8
SEARCH_RADIUS_METERS = 100.0
MAXIMUM_CANDIDATES = 5
ALIGNMENT_RADIUS_METERS = 100.0
GRID_CELL_METERS = 250.0
GRID_MARGIN_METERS = 150.0


def endpoint_distances(point, first, second):
    latitude = math.radians((point[1] + first[1] + second[1]) / 3.0)
    scale = max(math.cos(latitude), 1e-6)

    def project(lon, lat):
        return (math.radians(lon) * EARTH_RADIUS_METERS * scale,
                math.radians(lat) * EARTH_RADIUS_METERS)

    px, py = project(*point)
    ax, ay = project(*first)
    bx, by = project(*second)
    return math.hypot(px - ax, py - ay), math.hypot(px - bx, py - by)


def build_observations(trip_dir: Path, trip: str):
    """Clean one trip and generate candidates without reading its route."""
    cleaned, _ = clean_track(trip_dir / f"{trip}.track")
    nodes = load_nodes(trip_dir / f"{trip}.nodes")
    arcs = load_arcs(trip_dir / f"{trip}.arcs")
    if not cleaned:
        return cleaned, [], nodes, arcs

    scale = max(math.cos(math.radians(cleaned[0]["latitude"])), 1e-6)

    def cell_range(low, high):
        return range(math.floor(low / GRID_CELL_METERS), math.floor(high / GRID_CELL_METERS) + 1)

    def to_xy(lon, lat):
        return (math.radians(lon) * EARTH_RADIUS_METERS * scale,
                math.radians(lat) * EARTH_RADIUS_METERS)

    gps_cells = []
    for row in cleaned:
        x, y = to_xy(row["longitude"], row["latitude"])
        gps_cells.append((math.floor(x / GRID_CELL_METERS), math.floor(y / GRID_CELL_METERS)))
    index = {cell: [] for cell in set(gps_cells)}

    for arc_id, edge in enumerate(arcs):
        if edge is None or not all(0 <= node < len(nodes) and nodes[node] for node in edge):
            continue
        ax, ay = to_xy(*nodes[edge[0]])
        bx, by = to_xy(*nodes[edge[1]])
        for cx in cell_range(min(ax, bx) - GRID_MARGIN_METERS, max(ax, bx) + GRID_MARGIN_METERS):
            for cy in cell_range(min(ay, by) - GRID_MARGIN_METERS, max(ay, by) + GRID_MARGIN_METERS):
                if (cx, cy) in index:
                    index[(cx, cy)].append(arc_id)

    observations = []
    for i, row in enumerate(cleaned):
        point = (row["longitude"], row["latitude"])
        heading = math.nan
        if i > 0 and (row["delta_longitude_degrees"] or row["delta_latitude_degrees"]):
            before = cleaned[i - 1]
            heading = bearing_degrees((before["longitude"], before["latitude"]), point)
        found = []
        for arc_id in index[gps_cells[i]]:
            start, end = arcs[arc_id]
            distance = point_segment_distance(point, nodes[start], nodes[end])
            if distance <= SEARCH_RADIUS_METERS:
                found.append((distance, arc_id))
        found.sort()
        states = []
        for distance, arc_id in found[:MAXIMUM_CANDIDATES]:
            start, end = arcs[arc_id]
            length = haversine(*nodes[start], *nodes[end])
            to_start, to_end = endpoint_distances(point, nodes[start], nodes[end])
            states.append(State(
                arc_id=arc_id, from_node=start, to_node=end, length=length,
                position=position_along_arc(to_start, to_end, length), distance=distance,
                heading_difference=angle_difference(heading, bearing_degrees(nodes[start], nodes[end])),
                speed=row["speed_meters_per_second"],
            ))
        if states:
            observations.append(Observation(i, point[0], point[1], row["timestamp_seconds"], states))
    return cleaned, observations, nodes, arcs


def reference_arcs(trip_dir: Path, trip: str, cleaned, observations, nodes, arcs):
    """Align reference route arcs to GPS points, after decoding has finished."""
    route = []
    for line in (trip_dir / f"{trip}.route").read_text(encoding="utf-8-sig", errors="replace").splitlines():
        parts = line.split()
        if len(parts) == 1 and parts[0].lstrip("-").isdigit():
            route.append(int(parts[0]))
    segments = []
    for arc_id in dict.fromkeys(route):
        edge = arcs[arc_id] if 0 <= arc_id < len(arcs) else None
        if edge and all(0 <= node < len(nodes) and nodes[node] for node in edge):
            segments.append((arc_id, nodes[edge[0]], nodes[edge[1]]))
    reference = {}
    for observation in observations:
        point = (observation.lon, observation.lat)
        best_distance, best_arc = math.inf, None
        for arc_id, first, second in segments:
            distance = point_segment_distance(point, first, second)
            if distance < best_distance:
                best_distance, best_arc = distance, arc_id
        if best_arc is not None and best_distance <= ALIGNMENT_RADIUS_METERS:
            reference[observation.index] = best_arc
    return reference


def load_frozen_config(path: Path) -> DecoderConfig:
    saved = json.loads(path.read_text(encoding="utf-8"))["config"]
    return DecoderConfig(**{f.name: saved[f.name] for f in fields(DecoderConfig)})


def evaluate(root: Path, config: DecoderConfig):
    rows, classifier_rows, hmm_rows = [], [], []
    missed = aligned_total = gps_total = without_candidates = 0
    header = (f"{'Trip':<10}{'GPS':>7}{'Aligned':>9}{'No-HMM':>9}{'Viterbi':>9}"
              f"{'RouteF1':>9}{'Contin.':>9}{'Breaks':>8}{'Time s':>8}")
    print(header)
    print("-" * len(header))
    for trip_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        trip = trip_dir.name
        started = time.perf_counter()
        cleaned, observations, nodes, arcs = build_observations(trip_dir, trip)
        assign_emissions({trip: observations}, config, None)
        network = RoadNetwork.from_files(trip_dir / f"{trip}.nodes", trip_dir / f"{trip}.arcs")
        path, flags = decode(observations, network, config)
        independent = classifier_path(observations)
        elapsed = time.perf_counter() - started

        reference = reference_arcs(trip_dir, trip, cleaned, observations, nodes, arcs)
        hmm = score_path(observations, path, reference, sum(flags), elapsed)
        base = score_path(observations, independent, reference,
                          count_breaks(observations, independent, network, config))
        by_index = {o.index: {s.arc_id for s in o.states} for o in observations}
        missed += sum(arc not in by_index[i] for i, arc in reference.items())
        aligned_total += len(reference)
        gps_total += len(cleaned)
        without_candidates += len(cleaned) - len(observations)
        hmm_rows.append(hmm)
        classifier_rows.append(base)
        rows.append({"trip_id": trip, "gps_observations": len(cleaned),
                     "aligned_observations": len(reference),
                     "no_hmm_top1": base["top1_accuracy"], "viterbi_top1": hmm["top1_accuracy"],
                     "viterbi_route_f1": hmm["route_arc_f1"],
                     "viterbi_continuity": hmm["route_continuity"],
                     "viterbi_breaks": hmm["route_breaks"], "pipeline_seconds": elapsed})
        print(f"{trip:<10}{len(cleaned):>7}{len(reference):>9}{base['top1_accuracy']:>9.2%}"
              f"{hmm['top1_accuracy']:>9.2%}{hmm['route_arc_f1']:>9.3f}"
              f"{hmm['route_continuity']:>9.2%}{hmm['route_breaks']:>8}{elapsed:>8.2f}", flush=True)
    report = {
        "trips": len(rows),
        "no_hmm": summarize(classifier_rows),
        "hmm_viterbi": summarize(hmm_rows),
        "gps_observations": gps_total,
        "aligned_observations": aligned_total,
        "observations_without_candidates": without_candidates,
        "true_arc_not_in_candidates_rate": missed / aligned_total if aligned_total else math.nan,
    }
    return rows, report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("trips", type=Path, help="zip file or folder of trip folders")
    parser.add_argument("--config", type=Path, default=REPO_ROOT / "outputs" / "hmm_viterbi" / "metrics.json")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "evaluator_test")
    args = parser.parse_args()
    config = load_frozen_config(args.config)

    print("PathSnap route reconstruction")
    print("Emissions : distance-and-heading baseline (no trained classifier)")
    print("Decoder   : HMM/Viterbi on the full road network of each trip")
    print(f"Settings  : scale={config.transition_scale_meters:g} m, "
          f"break penalty={config.break_penalty:g} (fixed before this run)")
    print("Mode      : standalone script, not through the API\n")

    with tempfile.TemporaryDirectory() as folder:
        root = args.trips
        if args.trips.suffix.lower() == ".zip":
            with zipfile.ZipFile(args.trips) as archive:
                archive.extractall(folder)
            root = Path(folder)
            children = [p for p in root.iterdir() if p.is_dir()]
            if len(children) == 1 and not (children[0] / f"{children[0].name}.track").exists():
                root = children[0]
        rows, report = evaluate(root, config)

    no_hmm, hmm = report["no_hmm"], report["hmm_viterbi"]
    print("\nOverall")
    print(f"  Top-1 accuracy, no HMM     : {no_hmm['top1_accuracy']:.2%}")
    print(f"  Top-1 accuracy, Viterbi    : {hmm['top1_accuracy']:.2%}")
    print(f"  Mean trip accuracy         : {hmm['mean_trip_top1_accuracy']:.2%}")
    print(f"  Mean route arc F1          : {hmm['mean_route_arc_f1']:.3f}")
    print(f"  Route continuity           : {hmm['route_continuity']:.2%}")
    print(f"  Route breaks (no HMM -> Viterbi): {no_hmm['route_breaks']} -> {hmm['route_breaks']}")
    print(f"  True road not in candidates: {report['true_arc_not_in_candidates_rate']:.2%}")

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    with (args.out / "per_trip.csv").open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
