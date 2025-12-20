# Antibiotic Resistance Prediction System

End-to-end system using Python + CatBoost (backend ML) + Streamlit (frontend UI).

## Dataset

Place your dataset CSV `microbiology_combined_clean.csv` in the project root (already present).
Each row: `(patient_id, encounter_id, culture_time, organism, candidate_antibiotic)` plus clinical features.
Susceptibility values must be one of: `Susceptible`, `Intermediate`, `Resistant`.

## Train Models

1. Install dependencies:

```bash
pip install -r requirements.txt
```

2. Train models:

```bash
python backend/train.py --dataset microbiology_combined_clean.csv --models_dir models --mode binary_rs
```

This trains:
- CatBoostClassifier for current resistance (Resistant=1, Susceptible/Intermediate=0)
- CatBoostRegressor for time-to-resistance if a time column exists

Outputs:
- `models/classifier_cb.cbm` and `models/regressor_cb.cbm` (if time column detected)
- `models/metadata.json` with feature/categorical columns and value lists
- `models/metrics.json` with ROC-AUC, F1, Precision, Recall and RMSE

Group-based train/test split is done by `patient_id` or `encounter_id`.

## Run Frontend

```bash
streamlit run app.py
```

UI features:
- Sidebar for organism and candidate antibiotic selection
- Dynamic inputs for patient & clinical features
- Run Prediction button: shows resistance probability and predicted time to resistance
- Ranked list of alternative antibiotics (lowest probability, longest predicted time)

## Implementation Notes

- Only data available up to culture time is used; common leakage columns are excluded by name hints.
- CatBoost handles missing values natively. Categorical features are passed as strings.
- Regression excludes censored rows (missing time-to-resistance) from training.

## Configuration

- Classification head mode: `--mode binary_rs` or `--mode binary_ni`.
- Adjust `--test_size` if needed (default 0.2).

## Troubleshooting

- If susceptibility column is not detected, ensure a column contains only `Susceptible`, `Intermediate`, `Resistant` values.
- If organism/antibiotic columns are custom-named, detection scans for common names or substrings.
