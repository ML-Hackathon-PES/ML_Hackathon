# PATHSNAP dataset pipeline handoff

**Current branch:** `feature/complete-feature-table`

**Pipeline status:** candidate generation, candidate features, reference-route labels, and validation are complete for the 80 training and 10 student-test trips. No ML models have been trained.

## Dataset location and structure

The raw data is kept outside this repository in the sibling `Student folder`. From the repository root, its path is `..\Student folder`.

```text
workspace/
├── ML_Hackathon/
│   ├── src/
│   ├── config/
│   ├── outputs/
│   └── README.md
└── Student folder/
    ├── train/          # 80 trips
    └── student_test/   # 10 trips
```

Each trip directory contains `<trip_id>.track`, `.nodes`, `.arcs`, and `.route`. The raw files are never modified. Do not add them to Git.

## Reproducing the pipeline

Run these PowerShell commands from the `ML_Hackathon` repository root. Use a Python 3.13 environment with the dependencies in `requirements-eda.txt` installed.

```powershell
$datasetRoot = (Resolve-Path "..\Student folder").Path

# GPS cleaning and observation-level features
python src/dataset_preprocessing.py --dataset-root "$datasetRoot" --output-dir outputs

# Nearby road-arc candidates and per-trip summary
python src/candidate_generation.py `
  --dataset-root "$datasetRoot" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-output outputs\candidate_table.csv `
  --summary-output outputs\candidate_generation_summary.csv `
  --search-radius-meters 100 `
  --maximum-candidates 5

# Candidate-level features
python src/candidate_feature_engineering.py `
  --dataset-root "$datasetRoot" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-table outputs\candidate_table.csv `
  --output outputs\candidate_feature_table.csv

# Reference route alignment and labels
python src/route_alignment.py `
  --dataset-root "$datasetRoot" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-table outputs\candidate_table.csv `
  --labels-output outputs\candidate_labels.csv `
  --summary-output outputs\route_alignment_summary.csv `
  --alignment-radius-meters 100

# Validate without replacing the checked-in validation reports
python src/dataset_validation.py `
  --repository-root . `
  --report-output-dir "$env:TEMP\pathsnap-validation"
```

The processing scripts open their destination files in write mode. Check and back up existing outputs before rerunning a processing step. The validator reads inputs from `outputs/`; `--report-output-dir` directs only its three reports elsewhere.

The candidate generator, feature engineering, and route alignment scripts still have nested default dataset paths. Pass `--dataset-root` explicitly as above. The validator now defaults to the actual sibling data location.

## Current generated outputs

| File | Contents and current size in rows |
|---|---|
| `outputs/gps_feature_table.csv` | 227,577 cleaned GPS observations across 90 trips |
| `outputs/candidate_table.csv` | 1,010,103 observation-candidate pairs |
| `outputs/candidate_generation_summary.csv` | One row for each of the 90 trips |
| `outputs/candidate_feature_table.csv` | 1,010,103 candidate rows; 22 columns |
| `outputs/candidate_labels.csv` | 1,010,103 candidate labels, including explicitly unlabeled rows |
| `outputs/route_alignment_summary.csv` | Per-trip route alignment and miss counts for 90 trips |
| `outputs/dataset_validation_summary.csv` | Individual validation checks and statuses |
| `outputs/candidate_feature_missing_values.csv` | Non-finite value counts by numeric feature |
| `outputs/dataset_validation_report.md` | Human-readable validation report |

The feature and label files join on `(partition, trip_id, observation_index, arc_id)`. There is not yet a separate merged feature-plus-label CSV.

## Candidate-level feature columns

One row represents one cleaned GPS observation paired with one candidate road arc. The key is `(partition, trip_id, observation_index, arc_id)`. Identifiers are retained for joins and grouping; they are not all model predictors.

| Column | Meaning |
|---|---|
| `partition` | `train` or `student_test` |
| `trip_id` | Trip identifier |
| `observation_index` | Index of the cleaned GPS observation within the trip |
| `arc_id` | Zero-based road-arc row ID for this candidate |
| `from_node_id` | Candidate arc's start node ID |
| `to_node_id` | Candidate arc's end node ID |
| `gps_to_segment_distance_meters` | GPS point to candidate segment distance |
| `gps_to_segment_start_meters` | GPS point to candidate arc's start node distance |
| `gps_to_segment_end_meters` | GPS point to candidate arc's end node distance |
| `segment_length_meters` | Candidate arc length |
| `candidate_rank` | Rank by segment distance, then arc ID; starts at 1 |
| `candidate_count` | Number of retained candidates for that observation, at most 5 |
| `gps_speed_mps` | Speed at the current GPS observation |
| `gps_movement_direction_degrees` | Bearing from the preceding GPS observation to the current one |
| `previous_gps_speed_mps` | Speed at the preceding GPS observation |
| `previous_movement_direction_degrees` | Bearing of the movement into the preceding observation |
| `previous_delta_time_seconds` | Time interval associated with the preceding observation |
| `road_segment_direction_degrees` | Bearing from the candidate arc's start node to its end node |
| `gps_road_direction_difference_degrees` | Smallest circular angle between GPS movement and candidate-arc bearings |
| `timestamp_seconds` | Current GPS observation timestamp |
| `delta_time_seconds` | Time since the preceding GPS observation |
| `speed_suspicious` | 1 when the observation speed exceeds 150 m/s; otherwise 0 |

## Labels, unmatched routes, and missing features

For each GPS observation, the alignment code finds the nearest arc from that trip's `.route` reference sequence. If that arc is within 100 m, a candidate on the same arc receives label `1`; other candidates receive label `0`. If no reference arc is within 100 m, every candidate row for that observation is retained with an empty `label` and `alignment_status=unmatched_reference_route`. No label is fabricated.

The current label distribution is 203,420 positive, 747,464 negative, and 59,219 unlabeled rows. The summary distinguishes:

- **Candidate-generation misses:** 1,366 GPS observations had no candidate road arc within the candidate search radius.
- **Unmatched GPS observations during reference alignment:** 19,995 observations had no reference route arc within 100 m. These account for the 59,219 unlabeled candidate rows.
- **Unmatched reference arcs:** 3,984 reference route arc IDs were not the nearest aligned arc to any GPS observation.

Undefined direction features remain NaN in the CSV; they are not replaced with zero. The first observation has no current movement bearing. Previous-observation fields are undefined at the start of each trip. Current validation reports 468 missing `gps_movement_direction_degrees`, 468 missing `gps_road_direction_difference_degrees`, and 881 missing values in each previous-observation field (`previous_gps_speed_mps`, `previous_movement_direction_degrees`, `previous_delta_time_seconds`). All other feature columns have no missing values.

No imputation is performed in this pipeline. Any later model-specific imputer should be fit on that fold's training trips only; student-test data must not be used to fit preprocessing. Models that support missing values may handle them directly, according to the model workflow.

## Speed-outlier policy

The processed table follows the selected **preserve and flag** policy. The observation above 150 m/s is retained and `speed_suspicious` is set. One such observation is present: train trip `00000079`, observation 90, at about 180.14 m/s. The checked-in preparation config, the config generated by preprocessing, and the exploration report all document this policy. The row was not removed.

## Train and student-test separation

- All 80 `train` trips make up the group-aware cross-validation pool. The five deterministic folds split by `trip_id`; each fold uses 64 trips for training and 16 for validation, and each trip appears in validation once.
- The 10 `student_test` trips remain separate and are for development-stage evaluation only. Do not use them to fit models, select features, tune parameters, or fit imputers.
- The local route files allow labels to be generated for student-test rows; those labels are evaluation references, not training labels.
- No evaluator-testing partition is available locally. Evaluator data and labels must not enter this pipeline.

## Validation results

The validator reports **PASS** for 80 train trips, 10 student-test trips, zero train/test trip overlap, zero fold-overlap errors, zero duplicate candidate/feature/label keys, complete candidate-to-feature and candidate-to-label key coverage, zero invalid candidate arc/node references, and zero label-semantic errors.

Warnings are expected and documented: 1,366 GPS observations without candidates, 59,219 unlabeled candidate rows, and 3,579 non-finite feature values in the undefined direction fields listed above. These warnings do not indicate that missing candidates or unknown labels were filled in.

## Git exclusions and team sharing

`.gitignore` excludes these generated CSVs:

```text
outputs/gps_feature_table.csv
outputs/candidate_table.csv
outputs/candidate_feature_table.csv
outputs/candidate_labels.csv
outputs/candidate_generation_summary.csv
outputs/route_alignment_summary.csv
outputs/dataset_validation_summary.csv
outputs/candidate_feature_missing_values.csv
```

It also excludes `outputs/dataset_validation_report.md`, raw data extensions (`.track`, `.nodes`, `.arcs`, `.route`), and generated Folium HTML maps. The EDA/preprocessing CSVs already tracked in `outputs/` are not covered by these generated-table rules.

For the five-person team, share a versioned, access-controlled folder on the team's approved shared drive rather than putting these large files in Git. Replace the placeholder below with that shared folder path; run this from the repository root after confirming the outputs are the validated version:

```powershell
$shareRoot = "<approved-team-storage>\PathSnap\dataset-pipeline\2026-11-18"
$files = @(
  "outputs\gps_feature_table.csv",
  "outputs\candidate_table.csv",
  "outputs\candidate_feature_table.csv",
  "outputs\candidate_labels.csv",
  "outputs\candidate_generation_summary.csv",
  "outputs\route_alignment_summary.csv",
  "outputs\dataset_validation_summary.csv",
  "outputs\candidate_feature_missing_values.csv",
  "outputs\dataset_validation_report.md"
)
New-Item -ItemType Directory -Path $shareRoot -Force | Out-Null
Copy-Item -LiteralPath $files -Destination $shareRoot
$manifest = foreach ($file in $files) {
  [pscustomobject]@{
    File = Split-Path $file -Leaf
    Bytes = (Get-Item -LiteralPath $file).Length
    SHA256 = (Get-FileHash -LiteralPath $file -Algorithm SHA256).Hash
  }
}
$manifest | Export-Csv -LiteralPath (Join-Path $shareRoot "SHA256SUMS.csv") -NoTypeInformation
```

Give all five members read access to the versioned folder and ask them to verify each SHA-256 value after copying. Keep student-test labels clearly marked for development evaluation only; do not use those rows for training or tuning, and do not distribute evaluator labels.

## Source and documentation changes in this handoff

- `src/route_alignment.py`: adds `unmatched_gps_observations` to each alignment-summary row, making the unmatched GPS count explicit alongside aligned observations, unlabeled candidate rows, and unmatched reference arc IDs. Label rules are unchanged.
- `src/dataset_validation.py`: points to the sibling `Student folder`; streams large candidate, feature, and label CSVs; checks duplicate keys, exact candidate-to-feature/label key coverage, and positive/negative/unlabeled label semantics; and supports `--report-output-dir` so final validation can run without replacing existing reports.
- `src/dataset_preprocessing.py` and `config/feature_preparation.json`: describe the existing preserve-and-flag speed policy consistently; GPS feature calculations are unchanged.
- `DATASET_EXPLORATION_REPORT.md`: corrects the speed-outlier disposition, cleaned observation count, and cross-validation pool description.

No model training, commit, push, or merge has been performed.
