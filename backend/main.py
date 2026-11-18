"""PathSnap FastAPI backend.

Serves GPS observations, LR map-matching results, and model metrics to the
React frontend.  Uses pre-computed outputs wherever available to avoid
retraining or slow live inference.

Run:
    uvicorn backend.main:app --reload --port 8000
or from the ML_Hackathon directory:
    python -m uvicorn backend.main:app --reload --port 8000
"""

from __future__ import annotations

import csv
import json
import math
from functools import lru_cache
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# ── Paths ──────────────────────────────────────────────────────────────────
ROOT         = Path(__file__).resolve().parent.parent   # ML_Hackathon/
OUTPUTS      = ROOT / "outputs"
LR_OUT       = OUTPUTS / "logistic_regression"
DATASET_ROOT = ROOT / "Student folder" / "Student folder"

STUDENT_TEST_TRIPS = [
    "00000000", "00000007", "00000009", "00000040", "00000052",
    "00000056", "00000065", "00000076", "00000077", "00000091",
]

# ── App ────────────────────────────────────────────────────────────────────
app = FastAPI(title="PathSnap API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Helpers ────────────────────────────────────────────────────────────────

def norm_trip_id(raw: str) -> str:
    """Normalise trip IDs: '7' → '00000007', '00000007' → '00000007'."""
    try:
        return str(int(raw)).zfill(8)
    except ValueError:
        return raw


@lru_cache(maxsize=1)
def _load_gps_table() -> dict[str, list[dict]]:
    """Load gps_feature_table.csv; keyed by trip_id."""
    result: dict[str, list[dict]] = {}
    with open(OUTPUTS / "gps_feature_table.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            if row["partition"] != "student_test":
                continue
            tid = norm_trip_id(row["trip_id"])
            result.setdefault(tid, []).append(row)
    return result


@lru_cache(maxsize=1)
def _load_predictions() -> dict[str, list[dict]]:
    """Load lr_student_predictions.csv; keyed by normalised trip_id."""
    result: dict[str, list[dict]] = {}
    with open(LR_OUT / "lr_student_predictions.csv", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            tid = norm_trip_id(row["trip_id"])
            result.setdefault(tid, []).append(row)
    return result


@lru_cache(maxsize=20)
def _load_nodes(trip_id: str) -> list[tuple[float, float] | None]:
    path = DATASET_ROOT / "student_test" / trip_id / f"{trip_id}.nodes"
    if not path.exists():
        return []
    nodes: list[tuple[float, float] | None] = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    nodes.append((float(parts[0]), float(parts[1])))
                    continue
                except ValueError:
                    pass
            nodes.append(None)
    return nodes


@lru_cache(maxsize=20)
def _load_arcs(trip_id: str) -> list[tuple[int, int] | None]:
    path = DATASET_ROOT / "student_test" / trip_id / f"{trip_id}.arcs"
    if not path.exists():
        return []
    arcs: list[tuple[int, int] | None] = []
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        for line in f:
            parts = line.split()
            if len(parts) == 2:
                try:
                    arcs.append((int(parts[0]), int(parts[1])))
                    continue
                except ValueError:
                    pass
            arcs.append(None)
    return arcs


def _arc_midpoint(arc_id: int, nodes, arcs) -> tuple[float, float] | None:
    if arc_id >= len(arcs) or arcs[arc_id] is None:
        return None
    from_id, to_id = arcs[arc_id]
    if from_id >= len(nodes) or to_id >= len(nodes):
        return None
    a, b = nodes[from_id], nodes[to_id]
    if a is None or b is None:
        return None
    return ((a[0] + b[0]) / 2, (a[1] + b[1]) / 2)


def _top1_matched_route(trip_id: str) -> list[dict[str, Any]]:
    """Return one route point per GPS obs: highest-prob candidate's arc midpoint."""
    preds = _load_predictions().get(trip_id, [])
    nodes = _load_nodes(trip_id)
    arcs  = _load_arcs(trip_id)

    # Group by observation_index, keep highest prob
    best: dict[int, dict] = {}
    for row in preds:
        idx  = int(row["observation_index"])
        prob = float(row["predicted_prob"])
        if idx not in best or prob > float(best[idx]["predicted_prob"]):
            best[idx] = row

    route_points = []
    for idx in sorted(best):
        row    = best[idx]
        arc_id = int(row["arc_id"])
        mp     = _arc_midpoint(arc_id, nodes, arcs)
        if mp is None:
            continue
        route_points.append({
            "observation_index": idx,
            "arc_id":            arc_id,
            "lat":               mp[1],   # nodes store (lon, lat)
            "lon":               mp[0],
            "predicted_prob":    float(row["predicted_prob"]),
            "true_label":        row.get("label", ""),
        })

    return route_points


# ── Response models ────────────────────────────────────────────────────────

class TripSummary(BaseModel):
    id: str
    partition: str
    observations: int
    distance_km: float
    duration_s: float


class GpsPoint(BaseModel):
    observation_index: int
    lat: float
    lon: float
    timestamp_s: float
    speed_mps: float


class RoutePoint(BaseModel):
    observation_index: int
    arc_id: int
    lat: float
    lon: float
    predicted_prob: float
    true_label: str


class TripMatchResult(BaseModel):
    trip_id: str
    partition: str
    gps_points: list[GpsPoint]
    lr_route: list[RoutePoint]
    candidate_miss_rate: float
    observations_matched: int
    observations_total: int


# ── Endpoints ─────────────────────────────────────────────────────────────

@app.get("/api/trips", response_model=list[TripSummary])
def get_trips():
    gps_table = _load_gps_table()
    trips = []
    for tid in STUDENT_TEST_TRIPS:
        rows = gps_table.get(tid, [])
        if not rows:
            continue
        # Compute distance from step_distance_meters
        dist_km = sum(float(r.get("step_distance_meters", 0)) for r in rows) / 1000
        timestamps = [float(r["timestamp_seconds"]) for r in rows
                      if r.get("timestamp_seconds")]
        dur = max(timestamps) - min(timestamps) if len(timestamps) > 1 else 0.0
        trips.append(TripSummary(
            id=tid,
            partition="student_test",
            observations=len(rows),
            distance_km=round(dist_km, 2),
            duration_s=round(dur, 1),
        ))
    return trips


@app.get("/api/trips/{trip_id}", response_model=TripMatchResult)
def get_trip_match(trip_id: str):
    tid = norm_trip_id(trip_id)
    if tid not in STUDENT_TEST_TRIPS:
        raise HTTPException(status_code=404, detail=f"Trip {tid} not found")

    gps_table = _load_gps_table()
    gps_rows  = gps_table.get(tid, [])
    if not gps_rows:
        raise HTTPException(status_code=404, detail=f"No GPS data for trip {tid}")

    gps_points = [
        GpsPoint(
            observation_index=int(r["observation_index"]),
            lat=float(r["latitude"]),
            lon=float(r["longitude"]),
            timestamp_s=float(r["timestamp_seconds"]),
            speed_mps=float(r.get("speed_meters_per_second", 0)),
        )
        for r in gps_rows
    ]

    lr_route_dicts = _top1_matched_route(tid)
    lr_route = [RoutePoint(**p) for p in lr_route_dicts]

    # Candidate miss rate from generation summary
    miss_rate = 0.0
    gen_summary_path = OUTPUTS / "candidate_generation_summary.csv"
    if gen_summary_path.exists():
        with open(gen_summary_path) as f:
            for row in csv.DictReader(f):
                if norm_trip_id(row["trip_id"]) == tid:
                    miss_rate = float(row["candidate_miss_rate"])
                    break

    return TripMatchResult(
        trip_id=tid,
        partition="student_test",
        gps_points=gps_points,
        lr_route=lr_route,
        candidate_miss_rate=miss_rate,
        observations_matched=len(lr_route),
        observations_total=len(gps_points),
    )


@app.get("/api/metrics")
def get_metrics():
    def _load(path: Path) -> dict:
        return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}

    cv      = _load(LR_OUT / "lr_baseline_cv_overall_metrics.json")
    student = _load(LR_OUT / "lr_student_overall_metrics.json")
    folds   = json.loads((LR_OUT / "lr_baseline_fold_metrics.json").read_text()) \
              if (LR_OUT / "lr_baseline_fold_metrics.json").exists() else []
    timing  = _load(LR_OUT / "lr_timing.json")

    return {
        "logistic_regression": {
            "source": "actual",
            "cv_folds": 5,
            "accuracy":       cv.get("accuracy_mean"),
            "precision":      cv.get("precision_mean"),
            "recall":         cv.get("recall_mean"),
            "f1":             cv.get("f1_mean"),
            "macro_f1":       cv.get("macro_f1_mean"),
            "roc_auc":        cv.get("roc_auc_mean"),
            "log_loss":       cv.get("log_loss_mean"),
            "top1_accuracy":  cv.get("top1_mean"),
            "training_time_s": timing.get("cv_time_seconds"),
            "inference_time_s": student.get("inference_time_seconds"),
            "student_test": {
                "accuracy":      student.get("accuracy"),
                "f1":            student.get("f1"),
                "roc_auc":       student.get("roc_auc"),
                "top1_accuracy": student.get("top1_candidate_accuracy"),
                "rows":          student.get("rows_evaluated"),
            },
            "fold_results": [
                {
                    "fold":      f["fold"],
                    "accuracy":  f.get("accuracy"),
                    "f1":        f.get("f1"),
                    "macro_f1":  f.get("macro_f1"),
                    "roc_auc":   f.get("roc_auc"),
                    "log_loss":  f.get("log_loss"),
                    "top1":      f.get("top1_candidate_accuracy"),
                }
                for f in folds
            ],
        },
        "knn": {
            "source": "partial_placeholder",
            "status": "training_in_progress",
            "fold_results": [],
        },
        "hmm_viterbi": {
            "source": "placeholder",
            "status": "not_implemented",
        },
    }


@app.get("/api/status")
def get_status():
    return {
        "candidate_generation": "complete",
        "feature_engineering":  "complete",
        "route_alignment":      "complete",
        "lr_training":          "complete",
        "knn_training":         "running",
        "hmm_viterbi":          "pending",
    }


@app.get("/api/health")
def health():
    return {"status": "ok", "lr_model": str(LR_OUT / "lr_model.joblib")}
