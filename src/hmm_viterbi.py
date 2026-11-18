"""HMM/Viterbi route reconstruction over candidate road arcs.

Hidden states are the candidate directed arcs at each GPS observation.
Emissions come from classifier probabilities, or from a distance-and-heading
baseline when no probability file is given. Transitions compare the distance
travelled along the road network with the distance between GPS points.

Reference labels are read only after decoding, to score the result.

Usage (from the repository root):
    python src/hmm_viterbi.py --partition student_test
    python src/hmm_viterbi.py --probabilities outputs/random_forest/rf_student_test_probabilities.csv
    python src/hmm_viterbi.py --tune --tuning-probabilities outputs/random_forest/rf_train_oof_probabilities.csv

Add --dataset-root "Student folder/Student folder" to use the full road network
from the .nodes and .arcs files. Without it, the network contains only the arcs
that appear as candidates in the trip.
"""
from __future__ import annotations

import argparse
import csv
import heapq
import json
import math
import time
from collections import defaultdict
from dataclasses import asdict, dataclass
from pathlib import Path

EARTH_RADIUS_METERS = 6_371_008.8
REPO_ROOT = Path(__file__).resolve().parents[1]
PROBABILITY_COLUMNS = ("rf_probability", "predicted_prob", "emission_probability")
MINIMUM_PROBABILITY = 1e-6
MAXIMUM_SEARCH_METERS = 5_000.0


@dataclass(frozen=True)
class DecoderConfig:
    transition_scale_meters: float = 25.0
    break_penalty: float = 20.0
    emission_weight: float = 1.0
    distance_sigma_meters: float = 20.0
    heading_sigma_degrees: float = 45.0
    minimum_heading_speed_mps: float = 1.0


@dataclass
class State:
    arc_id: int
    from_node: int
    to_node: int
    length: float
    position: float
    distance: float
    heading_difference: float
    speed: float
    log_emission: float = 0.0


@dataclass
class Observation:
    index: int
    lon: float
    lat: float
    timestamp: float
    states: list[State]


def normalize_trip_id(value: str) -> str:
    return value.strip().zfill(8)


def to_float(value: str | None) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def haversine(lon1: float, lat1: float, lon2: float, lat2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    d_phi = phi2 - phi1
    d_lambda = math.radians(lon2 - lon1)
    value = math.sin(d_phi / 2) ** 2 + math.cos(phi1) * math.cos(phi2) * math.sin(d_lambda / 2) ** 2
    return 2 * EARTH_RADIUS_METERS * math.asin(min(1.0, math.sqrt(value)))


def position_along_arc(to_start: float, to_end: float, length: float) -> float:
    if length <= 1e-9:
        return 0.0
    position = (to_start ** 2 - to_end ** 2 + length ** 2) / (2 * length)
    return min(length, max(0.0, position))


def read_csv(path: Path):
    with path.open(encoding="utf-8-sig", newline="") as source:
        yield from csv.DictReader(source)


def trip_ids(data_dir: Path, partition: str) -> list[str]:
    rows = read_csv(data_dir / "candidate_generation_summary.csv")
    return sorted(normalize_trip_id(r["trip_id"]) for r in rows if r["partition"] == partition)


def load_trips(data_dir: Path, partition: str, selected: set[str]) -> dict[str, list[Observation]]:
    """Load candidate geometry and non-label features for the selected trips."""
    extras = {}
    for row in read_csv(data_dir / "candidate_feature_table.csv"):
        trip = normalize_trip_id(row["trip_id"])
        if row["partition"] == partition and trip in selected:
            key = (trip, int(row["observation_index"]), int(row["arc_id"]))
            extras[key] = (to_float(row["gps_road_direction_difference_degrees"]),
                           to_float(row["gps_speed_mps"]))

    grouped: dict[str, dict[int, Observation]] = defaultdict(dict)
    for row in read_csv(data_dir / "candidate_table.csv"):
        trip = normalize_trip_id(row["trip_id"])
        if row["partition"] != partition or trip not in selected:
            continue
        index, arc_id = int(row["observation_index"]), int(row["arc_id"])
        length = float(row["segment_length_meters"])
        heading, speed = extras.get((trip, index, arc_id), (math.nan, math.nan))
        state = State(
            arc_id=arc_id,
            from_node=int(row["from_node_id"]),
            to_node=int(row["to_node_id"]),
            length=length,
            position=position_along_arc(float(row["gps_to_segment_start_meters"]),
                                        float(row["gps_to_segment_end_meters"]), length),
            distance=float(row["gps_to_segment_distance_meters"]),
            heading_difference=heading,
            speed=speed,
        )
        observation = grouped[trip].get(index)
        if observation is None:
            observation = Observation(index, float(row["gps_longitude"]), float(row["gps_latitude"]),
                                      float(row["timestamp_seconds"]), [])
            grouped[trip][index] = observation
        observation.states.append(state)
    return {trip: [obs[i] for i in sorted(obs)] for trip, obs in grouped.items()}


def load_probabilities(path: Path, partition: str, column: str | None) -> dict:
    probabilities = {}
    for row in read_csv(path):
        if column is None:
            column = next((c for c in PROBABILITY_COLUMNS if c in row), None)
            if column is None:
                raise SystemExit(f"{path} needs one of {PROBABILITY_COLUMNS} or --probability-column.")
        if row.get("partition", partition) == partition:
            key = (normalize_trip_id(row["trip_id"]), int(row["observation_index"]), int(row["arc_id"]))
            probabilities[key] = float(row[column])
    return probabilities


def baseline_score(state: State, config: DecoderConfig) -> float:
    score = -0.5 * (state.distance / config.distance_sigma_meters) ** 2
    if math.isfinite(state.heading_difference) and state.speed >= config.minimum_heading_speed_mps:
        score -= 0.5 * (state.heading_difference / config.heading_sigma_degrees) ** 2
    return score


def assign_emissions(trips: dict, config: DecoderConfig, probabilities: dict | None) -> None:
    """Set log emissions, normalized across the candidates of each observation."""
    for trip, observations in trips.items():
        for observation in observations:
            if probabilities is None:
                scores = [baseline_score(s, config) for s in observation.states]
            else:
                try:
                    scores = [math.log(max(probabilities[(trip, observation.index, s.arc_id)],
                                           MINIMUM_PROBABILITY))
                              for s in observation.states]
                except KeyError as error:
                    raise SystemExit(f"No probability for candidate {error.args[0]}.") from error
            best = max(scores)
            total = math.log(sum(math.exp(score - best) for score in scores)) + best
            for state, score in zip(observation.states, scores):
                state.log_emission = score - total


class RoadNetwork:
    """Directed road graph with bounded shortest-path searches."""

    def __init__(self, edges: dict[int, dict[int, float]]):
        self.edges = edges
        self.cache: dict[int, tuple[float, dict[int, float]]] = {}

    @classmethod
    def from_candidates(cls, observations: list[Observation]) -> "RoadNetwork":
        edges: dict[int, dict[int, float]] = defaultdict(dict)
        for observation in observations:
            for s in observation.states:
                edges[s.from_node][s.to_node] = min(s.length, edges[s.from_node].get(s.to_node, math.inf))
        return cls(edges)

    @classmethod
    def from_files(cls, nodes_path: Path, arcs_path: Path) -> "RoadNetwork":
        nodes = []
        for line in nodes_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = line.split()
            nodes.append((float(parts[0]), float(parts[1])) if len(parts) == 2 else None)
        edges: dict[int, dict[int, float]] = defaultdict(dict)
        for line in arcs_path.read_text(encoding="utf-8-sig", errors="replace").splitlines():
            parts = line.split()
            if len(parts) != 2:
                continue
            start, end = int(parts[0]), int(parts[1])
            if 0 <= start < len(nodes) and 0 <= end < len(nodes) and nodes[start] and nodes[end]:
                length = haversine(*nodes[start], *nodes[end])
                edges[start][end] = min(length, edges[start].get(end, math.inf))
        return cls(edges)

    def distances_from(self, start: int, radius: float) -> dict[int, float]:
        cached = self.cache.get(start)
        if cached is not None and cached[0] >= radius:
            return cached[1]
        distances = {start: 0.0}
        queue = [(0.0, start)]
        while queue:
            distance, node = heapq.heappop(queue)
            if distance > distances.get(node, math.inf):
                continue
            for neighbor, length in self.edges.get(node, {}).items():
                candidate = distance + length
                if candidate <= radius and candidate < distances.get(neighbor, math.inf):
                    distances[neighbor] = candidate
                    heapq.heappush(queue, (candidate, neighbor))
        self.cache[start] = (radius, distances)
        return distances


def step_values(previous: Observation, current: Observation, config: DecoderConfig):
    """Return GPS distance, gap-adjusted scale and search radius for one step."""
    gps_distance = haversine(previous.lon, previous.lat, current.lon, current.lat)
    elapsed = max(1.0, current.timestamp - previous.timestamp)
    scale = config.transition_scale_meters * math.sqrt(elapsed)
    radius = min(MAXIMUM_SEARCH_METERS, gps_distance + 10 * scale)
    return gps_distance, scale, radius


def transition_score(a: State, b: State, gps_distance: float, scale: float, radius: float,
                     network: RoadNetwork) -> float | None:
    """Log transition score, or None when no directed route connects the arcs."""
    if a.arc_id == b.arc_id:
        # A negative displacement means moving backwards along a directed arc.
        road_distance = b.position - a.position
    else:
        between = network.distances_from(a.to_node, radius).get(b.from_node)
        if between is None:
            return None
        road_distance = (a.length - a.position) + between + b.position
    return -abs(road_distance - gps_distance) / scale


def decode(observations: list[Observation], network: RoadNetwork,
           config: DecoderConfig) -> tuple[list[int], list[bool]]:
    """Return the chosen state index per observation and break flags."""
    if not observations:
        return [], []
    weight = config.emission_weight
    scores = [weight * s.log_emission for s in observations[0].states]
    pointers: list[list[int]] = [[-1] * len(scores)]
    jumped: list[list[bool]] = [[False] * len(scores)]
    for previous, current in zip(observations, observations[1:]):
        gps_distance, scale, radius = step_values(previous, current, config)
        next_scores, next_pointers, next_jumped = [], [], []
        for b in current.states:
            best_score, best_index, best_jump = -math.inf, 0, True
            for index, a in enumerate(previous.states):
                move = transition_score(a, b, gps_distance, scale, radius, network)
                is_jump = move is None or move < -config.break_penalty
                total = scores[index] + (-config.break_penalty if is_jump else move)
                if total > best_score:
                    best_score, best_index, best_jump = total, index, is_jump
            next_scores.append(best_score + weight * b.log_emission)
            next_pointers.append(best_index)
            next_jumped.append(best_jump)
        scores = next_scores
        pointers.append(next_pointers)
        jumped.append(next_jumped)

    choice = max(range(len(scores)), key=scores.__getitem__)
    path = [0] * len(observations)
    breaks = [False] * len(observations)
    for t in range(len(observations) - 1, -1, -1):
        path[t] = choice
        breaks[t] = jumped[t][choice]
        choice = pointers[t][choice]
    return path, breaks


def classifier_path(observations: list[Observation]) -> list[int]:
    return [max(range(len(o.states)), key=lambda i: o.states[i].log_emission) for o in observations]


def count_breaks(observations, path, network, config) -> int:
    breaks = 0
    for t in range(1, len(observations)):
        previous, current = observations[t - 1], observations[t]
        values = step_values(previous, current, config)
        move = transition_score(previous.states[path[t - 1]], current.states[path[t]], *values, network)
        breaks += move is None or move < -config.break_penalty
    return breaks


def load_references(data_dir: Path, partition: str, selected: set[str]) -> dict:
    """Reference arc per aligned observation, used only for scoring."""
    references: dict[str, dict[int, int]] = defaultdict(dict)
    for row in read_csv(data_dir / "candidate_labels.csv"):
        trip = normalize_trip_id(row["trip_id"])
        if row["partition"] == partition and trip in selected and row["label"] != "":
            references[trip][int(row["observation_index"])] = int(row["reference_arc_id"])
    return references


def score_path(observations, path, reference, breaks, elapsed_seconds=None) -> dict:
    chosen = {o.index: o.states[i].arc_id for o, i in zip(observations, path)}
    correct = sum(chosen.get(index) == arc for index, arc in reference.items())
    predicted_arcs = {chosen[i] for i in reference if i in chosen}
    reference_arcs = set(reference.values())
    overlap = len(predicted_arcs & reference_arcs)
    precision = overlap / len(predicted_arcs) if predicted_arcs else 0.0
    recall = overlap / len(reference_arcs) if reference_arcs else 0.0
    steps = max(1, len(observations) - 1)
    return {
        "aligned_observations": len(reference),
        "correct_observations": correct,
        "top1_accuracy": correct / len(reference) if reference else math.nan,
        "route_arc_precision": precision,
        "route_arc_recall": recall,
        "route_arc_f1": 2 * precision * recall / (precision + recall) if overlap else 0.0,
        "route_breaks": breaks,
        "route_continuity": 1 - breaks / steps,
        "transitions": steps,
        **({"decode_seconds": elapsed_seconds} if elapsed_seconds is not None else {}),
    }


def build_network(trip, observations, partition, dataset_root) -> RoadNetwork:
    if dataset_root is None:
        return RoadNetwork.from_candidates(observations)
    folder = dataset_root / partition / trip
    return RoadNetwork.from_files(folder / f"{trip}.nodes", folder / f"{trip}.arcs")


def summarize(rows: list[dict]) -> dict:
    aligned = sum(r["aligned_observations"] for r in rows)
    steps = sum(r["transitions"] for r in rows)
    breaks = sum(r["route_breaks"] for r in rows)
    return {
        "trips": len(rows),
        "top1_accuracy": sum(r["correct_observations"] for r in rows) / aligned,
        "mean_trip_top1_accuracy": sum(r["top1_accuracy"] for r in rows) / len(rows),
        "mean_route_arc_f1": sum(r["route_arc_f1"] for r in rows) / len(rows),
        "route_breaks": breaks,
        "route_continuity": 1 - breaks / steps,
    }


def run_partition(trips, references, partition, config, dataset_root):
    outputs = {}
    per_trip = {"classifier_only": [], "hmm_viterbi": []}
    for trip in sorted(trips):
        observations = trips[trip]
        network = build_network(trip, observations, partition, dataset_root)
        started = time.perf_counter()
        path, flags = decode(observations, network, config)
        elapsed = time.perf_counter() - started
        independent = classifier_path(observations)
        reference = references.get(trip, {})
        hmm = score_path(observations, path, reference, sum(flags), elapsed)
        classifier = score_path(observations, independent, reference,
                                count_breaks(observations, independent, network, config))
        per_trip["hmm_viterbi"].append({"trip_id": trip, **hmm})
        per_trip["classifier_only"].append({"trip_id": trip, **classifier})
        outputs[trip] = (path, independent, flags)
    return outputs, per_trip


def tune(args, base: DecoderConfig) -> tuple[DecoderConfig, list[dict]]:
    """Grid-search decoder settings on a spaced subset of training trips."""
    all_trips = trip_ids(args.data_dir, "train")
    step = max(1, len(all_trips) // args.tuning_trips)
    selected = set(all_trips[::step][:args.tuning_trips])
    trips = load_trips(args.data_dir, "train", selected)
    probabilities = None
    if args.tuning_probabilities:
        probabilities = load_probabilities(args.tuning_probabilities, "train", args.probability_column)
    assign_emissions(trips, base, probabilities)
    references = load_references(args.data_dir, "train", selected)
    networks = {t: build_network(t, o, "train", args.dataset_root) for t, o in trips.items()}
    results = []
    for scale in (10.0, 25.0, 50.0, 100.0):
        for penalty in (10.0, 20.0, 40.0):
            config = DecoderConfig(**{**asdict(base), "transition_scale_meters": scale,
                                      "break_penalty": penalty})
            rows = []
            for trip, observations in trips.items():
                path, flags = decode(observations, networks[trip], config)
                rows.append(score_path(observations, path, references.get(trip, {}), sum(flags)))
            summary = summarize(rows)
            results.append({"transition_scale_meters": scale, "break_penalty": penalty, **summary})
            print(f"scale={scale:g} penalty={penalty:g} top1={summary['top1_accuracy']:.4f} "
                  f"continuity={summary['route_continuity']:.4f}", flush=True)
    best = max(results, key=lambda r: (r["top1_accuracy"], r["route_continuity"]))
    chosen = DecoderConfig(**{**asdict(base),
                              "transition_scale_meters": best["transition_scale_meters"],
                              "break_penalty": best["break_penalty"]})
    return chosen, results


def write_table(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.DictWriter(target, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def write_predictions(path: Path, partition: str, trips: dict, outputs: dict) -> None:
    with path.open("w", newline="", encoding="utf-8") as target:
        writer = csv.writer(target)
        writer.writerow(["partition", "trip_id", "observation_index", "viterbi_arc_id",
                         "viterbi_from_node_id", "viterbi_to_node_id", "classifier_arc_id",
                         "route_break_before"])
        for trip in sorted(outputs):
            path_states, independent, flags = outputs[trip]
            for t, observation in enumerate(trips[trip]):
                chosen = observation.states[path_states[t]]
                writer.writerow([partition, trip, observation.index, chosen.arc_id, chosen.from_node,
                                 chosen.to_node, observation.states[independent[t]].arc_id,
                                 int(flags[t])])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--data-dir", type=Path, default=REPO_ROOT / "PathSnap_Validated_Dataset")
    parser.add_argument("--dataset-root", type=Path, help="folder holding raw <partition>/<trip> files")
    parser.add_argument("--partition", default="student_test", choices=("student_test", "train"))
    parser.add_argument("--probabilities", type=Path, help="candidate probabilities to use as emissions")
    parser.add_argument("--probability-column")
    parser.add_argument("--out", type=Path, default=REPO_ROOT / "outputs" / "hmm_viterbi")
    parser.add_argument("--transition-scale", type=float, default=DecoderConfig.transition_scale_meters)
    parser.add_argument("--break-penalty", type=float, default=DecoderConfig.break_penalty)
    parser.add_argument("--emission-weight", type=float, default=DecoderConfig.emission_weight)
    parser.add_argument("--tune", action="store_true", help="choose settings on training trips")
    parser.add_argument("--tuning-trips", type=int, default=12)
    parser.add_argument("--tuning-probabilities", type=Path,
                        help="out-of-fold training probabilities for --tune")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    config = DecoderConfig(transition_scale_meters=args.transition_scale,
                           break_penalty=args.break_penalty,
                           emission_weight=args.emission_weight)
    if args.tune:
        if args.probabilities and not args.tuning_probabilities:
            raise SystemExit("--tune with --probabilities also needs --tuning-probabilities.")
        config, results = tune(args, config)
        write_table(args.out / "tuning_results.csv", results)
        print("selected:", config, flush=True)

    selected = set(trip_ids(args.data_dir, args.partition))
    trips = load_trips(args.data_dir, args.partition, selected)
    probabilities = None
    if args.probabilities:
        probabilities = load_probabilities(args.probabilities, args.partition, args.probability_column)
    assign_emissions(trips, config, probabilities)
    references = load_references(args.data_dir, args.partition, selected)
    outputs, per_trip = run_partition(trips, references, args.partition, config, args.dataset_root)

    report = {
        "partition": args.partition,
        "emission_source": str(args.probabilities.name) if args.probabilities else "distance_heading_baseline",
        "network_source": "raw_arcs" if args.dataset_root else "candidate_arcs_only",
        "config": asdict(config),
        "classifier_only": summarize(per_trip["classifier_only"]),
        "hmm_viterbi": summarize(per_trip["hmm_viterbi"]),
        "decode_seconds_total": sum(r["decode_seconds"] for r in per_trip["hmm_viterbi"]),
    }
    (args.out / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    write_table(args.out / "per_trip_classifier_only.csv", per_trip["classifier_only"])
    write_table(args.out / "per_trip_hmm_viterbi.csv", per_trip["hmm_viterbi"])
    write_predictions(args.out / "route_predictions.csv", args.partition, trips, outputs)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
