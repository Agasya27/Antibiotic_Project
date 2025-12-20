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

## Deploy on Vercel (API + Static Frontend)

Vercel does not host a persistent Streamlit server. Instead, this repo includes a serverless FastAPI endpoint and a minimal static page:

- API function: `api/predict.py` (FastAPI)
- Static frontend: `index.html`
- Config: `vercel.json`

The API loads trained models from `models/metadata.json` and the `.cbm` binaries. If the `.cbm` files are not committed, the function can download them at cold start from URLs you provide as environment variables.

### Environment Variables (on Vercel)

Configure these in your Vercel Project Settings → Environment Variables:

- `CLASSIFIER_MODEL_URL`: HTTPS URL to `classifier_cb.cbm`
- `REGRESSOR_MODEL_URL`: HTTPS URL to `regressor_cb.cbm` (omit or still provide if regression not used)

Notes:
- If you commit the `.cbm` files into `models/`, the API will use them directly and ignore the URLs.
- If you use URLs, they will be downloaded to `/tmp/models` on cold start and referenced via a patched `metadata.json` in `/tmp`.

### Deploy Steps

1. Ensure models are available:
	- Option A: Commit `models/classifier_cb.cbm` and `models/regressor_cb.cbm` to the repo (beware size limits).
	- Option B: Upload the binaries to a stable hosting location (e.g., GitHub Releases, cloud storage) and set the env vars above.

2. Push to GitHub (already done). Then on Vercel:
	- Import the repository `Agasya27/Antibiotic_Project`.
	- Set env vars as needed.
	- Trigger a deployment.

3. Use the deployed site:
	- Open the root URL to access the simple page (`index.html`).
	- It posts to `/api/predict` with JSON and renders results.

### Local API Test

You can run the API locally (requires models present locally):

```bash
uvicorn api.predict:app --host 0.0.0.0 --port 8000
```

Test with:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
		  "organism": "ESCHERICHIA COLI",
		  "antibiotic": "Trimethoprim/Sulfamethoxazole",
		  "features": {"age": 45, "sex": "F"},
		  "top_k": 5
		}'
```

### Limitations

- Serverless functions have memory/time limits; large models may increase cold start time.
- Training should be done offline; deploy only inference artifacts.
