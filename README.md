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
