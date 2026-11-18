# PATHSNAP Dataset Preparation — Pipeline Status

> **Current status (DatasetProcess branch):**
> - ✅ EDA, GPS cleaning, partition assignment, and basic feature preparation are complete.
> - ⚠️ Candidate road-arc generation is **incomplete** — `candidate_table.csv` currently
>   covers only 13 of 90 trips due to an interrupted run. Regeneration is required.
> - ❌ Candidate-level feature engineering (`candidate_feature_table.csv`) has not been run.
> - ❌ Route alignment and labels (`candidate_labels.csv`) have not been run.
> - ❌ The final ML dataset (features + labels joined) does not yet exist.
> - **Do not begin model training until the pipeline steps below are completed in order.**

This repository stage performs raw dataset inspection, cleaning, basic observation-level
feature preparation, partition documentation, and exploratory visualisation.

## Expected raw dataset location

Place the provided student data below:

```text
ML_Hackathon/Student folder/Student folder/
  train/<trip_id>/<trip_id>.track
  train/<trip_id>/<trip_id>.nodes
  train/<trip_id>/<trip_id>.arcs
  train/<trip_id>/<trip_id>.route
  student_test/<trip_id>/<trip_id>.track
  student_test/<trip_id>/<trip_id>.nodes
  student_test/<trip_id>/<trip_id>.arcs
  student_test/<trip_id>/<trip_id>.route
```

If an evaluator-testing partition is supplied later, create a separate folder and keep
its trips and labels outside every training, feature-selection, tuning, and development
evaluation process. Do not commit evaluator labels or use them for feature selection.

## Setup

```powershell
python -m pip install -r requirements-eda.txt
```

## Generate dataset outputs

> **Important:** All scripts default to a dataset path that assumes a nested
> `ML_Hackathon/ML_Hackathon/` directory which does not exist. Always pass
> `--dataset-root` explicitly as shown below.

**Step 1 — EDA (run first):**

```powershell
python src/dataset_exploration.py --dataset-root "Student folder\Student folder"
```

**Step 2 — GPS cleaning and basic features:**

```powershell
python src/dataset_preprocessing.py --dataset-root "Student folder\Student folder"
```

**Step 3 — Candidate road-arc generation (INCOMPLETE — must be run):**

```powershell
python src/candidate_generation.py `
  --dataset-root "Student folder\Student folder" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-output outputs\candidate_table.csv `
  --summary-output outputs\candidate_generation_summary.csv `
  --search-radius-meters 100 `
  --maximum-candidates 5
```

**Step 4 — Candidate-level feature engineering (NOT YET RUN):**

```powershell
python src/candidate_feature_engineering.py `
  --dataset-root "Student folder\Student folder" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-table outputs\candidate_table.csv `
  --output outputs\candidate_feature_table.csv
```

**Step 5 — Route alignment and label generation (NOT YET RUN):**

```powershell
python src/route_alignment.py `
  --dataset-root "Student folder\Student folder" `
  --gps-feature-table outputs\gps_feature_table.csv `
  --candidate-table outputs\candidate_table.csv `
  --labels-output outputs\candidate_labels.csv `
  --summary-output outputs\route_alignment_summary.csv `
  --alignment-radius-meters 100
```

**Step 6 — Dataset validation (NOT YET RUN):**

```powershell
python src/dataset_validation.py
```

> **Known issue in `dataset_validation.py`:** The dataset root is currently hardcoded
> inside `run_validation()` to the wrong path. Fix the path to
> `repository_root / "Student folder" / "Student folder"` before running.

Open `notebooks/01_dataset_exploration.ipynb` for the reproducible notebook workflow
(Steps 1–2 only; notebook cells have not yet been executed).

## Partition rules

The script keeps each trip entirely inside one partition:

- All 80 `train` trips form the **cross-validation pool** — there is no permanently
  held-out validation set. Each 5-fold CV round uses 64 trips for training and 16 for
  fold-validation, rotating across all 80 trips so every trip contributes to validation
  exactly once.
- `student_test` (10 trips) is used for development-stage evaluation only; it is never
  mixed into training, feature selection, or cleaning-rule decisions.
- Evaluator-testing data is excluded from all code paths.

> **README correction:** An earlier version of this document incorrectly described a
> permanent "64-trip training pool and 16-trip validation holdout". That description was
> wrong. The actual implementation (all 80 trips in the CV pool, no permanent holdout)
> is correct and matches the faculty requirements.

The exact assignments are committed in `outputs/dataset_splits.csv` and
`outputs/cross_validation_folds.csv`.

## Cleaning rules

Raw files are never modified. Cleaning rules are:

- Longitude must be finite and between -180 and 180 degrees.
- Latitude must be finite and between -90 and 90 degrees.
- Timestamps must be finite and strictly increasing within a trip.
- Exact duplicate coordinate/timestamp observations are removed.
- Haversine step distance is calculated between consecutive accepted observations.
- Observations implying speed above 150 m/s are reported and removed.

## Generated feature fields

The feature table contains longitude, latitude, timestamp, delta time, degree deltas,
Haversine step distance, Haversine-derived speed, and cumulative trip distance.
It does not generate road-network candidates, route labels, candidate features, or
model-ready encodings.

## Output structure

```text
src/
  dataset_exploration.py          # Step 1: EDA
  dataset_preprocessing.py        # Step 2: GPS cleaning + basic features
  candidate_generation.py         # Step 3: Road-arc candidate generation (INCOMPLETE)
  candidate_feature_engineering.py # Step 4: Candidate-level ML features (NOT RUN)
  route_alignment.py              # Step 5: Reference route alignment + labels (NOT RUN)
  dataset_validation.py           # Step 6: Dataset integrity validation (NOT RUN)
notebooks/
  01_dataset_exploration.ipynb    # Reproducible notebook for Steps 1–2 (not yet executed)
outputs/
  dataset_inventory.csv           # ✅ committed
  gps_summary.csv                 # ✅ committed
  missing_value_analysis.csv      # ✅ committed
  timestamp_analysis.csv          # ✅ committed
  road_network_summary.csv        # ✅ committed
  route_summary.csv               # ✅ committed
  gps_cleaning_log.csv            # ✅ committed
  cleaned_gps_samples.csv         # ✅ committed (25 obs/trip sample only)
  speed_outlier_audit.csv         # ✅ committed
  speed_outlier_sensitivity.csv   # ✅ committed
  dataset_splits.csv              # ✅ committed
  cross_validation_folds.csv      # ✅ committed
  figures/                        # ✅ committed (EDA plots)
  maps/                           # ✅ committed (Folium HTML maps)
  gps_feature_table.csv           # ⛔ NOT committed (33 MB, regenerable)
  candidate_table.csv             # ⛔ NOT committed (incomplete 13/90 trips, 45 MB)
  candidate_feature_table.csv     # ⛔ does not exist yet
  candidate_labels.csv            # ⛔ does not exist yet
  candidate_generation_summary.csv # ⛔ does not exist yet
config/
  dataset_partitions.json         # ✅ committed
  feature_preparation.json        # ✅ committed
  candidate_generation.json       # ✅ committed
  route_alignment.json            # ✅ committed
DATASET_EXPLORATION_REPORT.md     # ✅ committed
README.md                         # ✅ committed
requirements-eda.txt              # ✅ committed
```

Raw `.track`, `.nodes`, `.arcs`, and `.route` files are not committed (excluded via `.gitignore`).
System Architecture & Workflow

Data Pipeline & Candidate Generation: Raw GPS pings are matched to nearby candidate road arcs within a spatial buffer.

Feature Extraction: Features like distance to arc, bearing difference, position along arc, and speed are computed and saved into a shared feature table.

Model Experimentation: Five distinct families (Logistic Regression, KNN, Random Forest, XGBoost/HistGradientBoosting, and MLP) are trained using Group-Aware Cross-Validation (split by trip_id) to prevent data leakage.

HMM / Viterbi Decoding: Classifier probabilities act as emission probabilities, while network connectivity (.arcs and .nodes) forms transition probabilities to smooth out erratic jumps.

Deployment & UI: FastAPI serves the inference pipeline while an interactive web UI displays trip routes and accuracy metrics.

 Repository Directory Structure

PathSnap/
│
├── data/                    # Raw .nodes, .arcs, .route, and .track datasets
├── src/
│   ├── data_pipeline.py     # Shreya NB: Candidate generation & features
│   ├── models_exp.py        # Shrungashree & Sharanabasava: ML models (LR, KNN, RF, XGB)
│   ├── mlp_model.py         # Shreya Ajith: Neural Network (MLP)
│   └── viterbi_hmm.py       # Sharanabasava: HMM & Viterbi path smoothing
├── api/
│   └── main.py              # Shreya Ajith: FastAPI backend server
├── ui/
│   └── app.py               # Shreya Ajith: Interactive UI frontend
├── notebooks/               # EDA and experimentation Jupyter notebooks
├── requirements.txt         # Project dependencies
└── README.md                # Project documentation


 Installation & Local Setup

1. Clone & Setup Virtual Environment

git clone https://github.com/ML-Hackathon-PES/ML_Hackathon.git
cd pathsnap
python -m venv venv


2. Activate Virtual Environment

Windows (PowerShell):

.\venv\Scripts\Activate.ps1


macOS / Linux:

source venv/bin/activate


3. Install Dependencies

pip install -r requirements.txt


Running the Pipeline

Step 1: Feature Generation

Run Shreya NB's data pipeline to parse raw .nodes and .arcs files and generate the feature table:

python src/data_pipeline.py


Step 2: Model Training & Experimentation

Train the models (Logistic Regression, KNN, Random Forest, XGBoost, MLP) and run cross-validation:

python src/models_exp.py


Step 3: HMM & Viterbi Smoothing

Apply the Viterbi decoder on classifier probabilities to generate smoothed route predictions:

python src/viterbi_hmm.py


Step 4: Launch API & UI (Deployment)

Start FastAPI Server (Terminal 1):

uvicorn api.main:app --reload --port 8000


Start UI Frontend (Terminal 2):

streamlit run ui/app.py

 Evaluation Metrics

Models are evaluated across the 10 student-testing trips using:

Classification: Accuracy, Precision, Recall, F1-score, Macro F1.

Probability Quality: Log Loss, Top-1 Accuracy.

Efficiency: Training time and Inference time.

Continuity: Route jump counts and candidate miss rates.
