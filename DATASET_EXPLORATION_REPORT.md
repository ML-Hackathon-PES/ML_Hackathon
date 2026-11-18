# PATHSNAP Dataset Exploration Report

This report is generated from the raw trip files found on disk by `src/dataset_exploration.py`. It performs dataset inspection and EDA only; it does not train models, reconstruct routes, or modify raw files.

## Dataset structure and partition use

| Partition | Trips | Expected files present |
|---|---:|---:|
| `student_test` | 10 | 40/40 |
| `train` | 80 | 320/320 |

Missing expected files: **0**. Files with malformed rows or numeric parse errors: **0**.

The `train` partition is summarized separately and is the training/development partition specified by the faculty. `student_test` is kept separate and is described only as development-stage evaluation data; it is not mixed into training summaries at trip level. No evaluator-testing partition was found locally, so final evaluator results cannot be explored. No predictions, feature selection, or tuning were performed.

The observed layout is `<partition>/<trip_id>/<trip_id>.<track|nodes|arcs|route>`. Each expected file is parsed as whitespace-separated and checked for expected column count; tabs are valid whitespace. The raw sample uses tab separators.

## GPS trajectory summary

| Partition | Trips | Observations (median, range) | Duration seconds (median, range) | Distance km (median, range) | Invalid coordinates | Duplicate full observations | Irregular intervals | Non-increasing time intervals | Gaps >5 s |
|---|---:|---|---|---|---:|---:|---:|---:|---:|
| `student_test` | 10 | median 1,420.00 (range 262.00 to 3,958.00) | median 1,809.50 (range 265.00 to 4,000.00) | median 20.94 km (range 5.55 to 71.72 km) | 0 | 0 | 264 | 0 | 44 |
| `train` | 80 | median 1,529.00 (range 361.00 to 19,313.00) | median 1,710.00 (range 451.00 to 154,030.00) | median 24.07 km (range 6.33 to 169.25 km) | 0 | 0 | 7273 | 0 | 1216 |

Coordinates are interpreted in the documented order: longitude, latitude. The script checks legal WGS84 longitude/latitude ranges and records per-trip coordinate extents in `gps_summary.csv`. Intervals within 0.1 s of the nominal 1 Hz interval are treated as regular; positive intervals over 5 s are counted as large gaps. These are explicit EDA thresholds, not data-cleaning rules.

Trajectory distance is the sum of Haversine great-circle distances between consecutive valid coordinate rows, using a spherical Earth radius of 6,371,008.8 m. It is an approximation to ellipsoidal geodesic distance; longitude/latitude degrees are never used as Euclidean distance. A distance segment can span a timestamp gap, so gap counts should be read alongside the distance statistic. Timestamps are treated as seconds since the trip-relative origin because the files contain no timezone/date metadata.

Observed GPS coordinate extents by partition:
- `student_test`: longitude -93.424337 to 77.271115; latitude 42.645529 to 61.699590.
- `train`: longitude -100.418538 to 137.237029; latitude -34.640291 to 61.267258.

## Road-network summary

| Partition | Nodes (total; median) | Arcs (total; median) | Invalid arc references | Isolated nodes | Trips with >1 weak component | Median largest-component share |
|---|---:|---:|---:|---:|---:|---:|
| `student_test` | 90,689; median 6,596.50 (range 1,770.00 to 26,635.00) | 185,041; median 13,145.00 (range 3,577.00 to 56,027.00) | 0 | 0 | 10 | 0.9706 |
| `train` | 1,933,485; median 9,608.00 (range 299.00 to 155,003.00) | 3,876,904; median 19,526.50 (range 649.00 to 307,392.00) | 0 | 0 | 79 | 0.9811 |

Node IDs and arc IDs are interpreted as zero-based row indices. Invalid arc references are edges with malformed rows or endpoints outside the node-row range. Isolated nodes have zero in-degree and out-degree. Weak components describe connectivity when direction is ignored; strong components preserve directed reachability. Component counts include isolated nodes. Geographic network extents and per-trip component counts are in `road_network_summary.csv`.

Observed network-node coordinate extents and track/network overlap:
- `student_test` nodes: longitude -93.434290 to 77.280906; latitude 42.635672 to 61.709158. GPS and node bounding boxes overlap for 10/10 trips.
- `train` nodes: longitude -100.428454 to 137.247005; latitude -34.650477 to 61.277181. GPS and node bounding boxes overlap for 80/80 trips.

GPS and node bounding boxes overlap for every available trip; together with the coordinate ranges and plotted GPS/network alignment, this supports the longitude-first, latitude-second interpretation in the raw files.

## Reference-route summary

| Partition | Route entries (median) | Invalid arc IDs | Disconnected consecutive pairs | Checked pairs | Median route length km |
|---|---:|---:|---:|---:|---:|
| `student_test` | median 250.00 (range 95.00 to 1,323.00) | 0 | 1 | 3467 | median 14.73 km (range 5.28 to 64.75 km) |
| `train` | median 280.00 (range 59.00 to 2,266.00) | 0 | 13 | 31256 | median 14.30 km (range 5.00 to 94.15 km) |

Route IDs are looked up against arc-row IDs, not GPS observations. A consecutive route pair is checked only when both route IDs and both referenced arcs are valid; continuity requires the previous arc's destination node to equal the next arc's source node. Route length sums geodesic distances between the referenced node coordinates and is not inferred from the number of GPS fixes.

## Visual outputs

Static trajectory figures contain all trips in each partition, in panels sorted by trip ID. Coordinate and sampling-interval distributions are partition-separated; the interval histogram clips its display at the 99.5th percentile only to keep the main distribution legible. Road-network local figures use the selected trip's network near its GPS extent and color links by weak component. Folium maps have separate GPS, local-road, and reference-route layers, with layer controls. For map readability, roads are clipped to a 0.02 degree GPS bounding-box buffer; when more than 12,000 local arcs are selected, the displayed ordinary roads are deterministically thinned, while route arcs are retained.

| Partition | Map trip | Interactive map | Network plot |
|---|---|---|---|
| `train` | `00000083` | `outputs/maps/pathsnap_map_train_00000083.html` | `outputs/figures/road_network_local_train_00000083.png` |
| `student_test` | `00000007` | `outputs/maps/pathsnap_map_student_test_00000007.html` | `outputs/figures/road_network_local_student_test_00000007.png` |

## Output files

- `outputs/dataset_inventory.csv`: one record per expected trip file, plus unexpected files.
- `outputs/gps_summary.csv`: per-trip GPS counts, anomalies, timing, distance, and extent.
- `outputs/missing_value_analysis.csv`: per-trip and per-field missing/nonfinite values and parse errors.
- `outputs/timestamp_analysis.csv`: per-trip timestamp order and interval analysis.
- `outputs/road_network_summary.csv`: per-trip graph validity, component, isolation, and extent statistics.
- `outputs/route_summary.csv`: per-trip reference route ID, continuity, and distance statistics.
- `outputs/gps_cleaning_log.csv`: raw-versus-cleaned counts for every trip.
- `outputs/cleaned_gps_samples.csv`: 25 cleaned observations per trip.
- `outputs/gps_feature_table.csv`: cleaned observation-level basic features.
- `outputs/dataset_splits.csv`: trip-level training-pool and student-test roles.
- `outputs/cross_validation_folds.csv`: deterministic 5-fold assignments for all 80 train trips.
- `outputs/figures/`: actual-data trajectory, distribution, and network plots.
- `outputs/maps/`: interactive Folium maps based on selected trips.

## Cleaning and partition artifacts

Raw files remain unchanged. Observations above 150 m/s are preserved and flagged.
Train trip `00000079`, observation 90, has speed 180.14 m/s and is marked
`speed_suspicious=1`. No missing, nonfinite, duplicate, or decreasing-timestamp
observations were reported before preprocessing. The cleaned GPS feature table
contains 227,577 observation rows from all 90 trips.

All 80 train trips form the cross-validation pool; there is no permanent holdout.
Each deterministic fold assigns 64 trips to training and 16 to validation, with
every train trip appearing in validation once across five folds. The 10
student-test trips remain separate for development-stage evaluation and are not
used for training, feature selection, or cleaning-rule selection. No
evaluator-testing partition was found locally.

## Scope and limitations

This is dataset preparation/EDA only. No raw file is rewritten or imputed in place.
Raw invalid observations are reported, and cleaned copies plus basic feature tables
are generated separately. The local map view is a readability-focused subset, not a
complete rendering of a potentially large trip network. Haversine distances are
spherical approximations; no projected CRS was assumed because the data spans
multiple geographic regions. Install the plotting/map dependencies with
`python -m pip install -r requirements-eda.txt`, then run
`python src/dataset_exploration.py` and `python src/dataset_preprocessing.py`
to regenerate the outputs.
