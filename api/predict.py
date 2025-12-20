import os
import sys
import json
import shutil
from typing import Dict, Any, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

# Ensure project root is importable when running from Vercel function directory
PROJECT_ROOT = os.path.dirname(os.path.dirname(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from backend.infer import ModelBundle, predict_for_new_patient
from backend.recommend import rank_antibiotics


class PredictRequest(BaseModel):
    organism: str
    antibiotic: str
    features: Dict[str, Any]
    top_k: Optional[int] = 5


app = FastAPI(title="Antibiotic Resistance Prediction API")

_bundle: Optional[ModelBundle] = None


def _download_file(url: str, dest_path: str) -> None:
    import requests
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    r = requests.get(url, stream=True, timeout=60)
    if r.status_code != 200:
        raise HTTPException(status_code=502, detail=f"Failed to download model from {url}: {r.status_code}")
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(r.raw, f)


def _ensure_models_available(models_dir: str) -> str:
    """Ensure classifier/regressor .cbm exist locally. If missing, try downloading
    from env vars CLASSIFIER_MODEL_URL and REGRESSOR_MODEL_URL into /tmp/models.

    Returns the directory that holds usable models for the bundle.
    """
    metadata_path = os.path.join(models_dir, "metadata.json")
    if not os.path.exists(metadata_path):
        raise HTTPException(status_code=500, detail="metadata.json not found in models directory")

    with open(metadata_path, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    classifier_path = metadata.get("classifier_model_path") or metadata.get("classifier_path")
    regressor_path = metadata.get("regressor_model_path") or metadata.get("regressor_path")

    # If paths are relative, resolve against models_dir
    def resolve_path(p: Optional[str]) -> Optional[str]:
        if not p:
            return None
        if os.path.isabs(p):
            return p
        return os.path.join(models_dir, p)

    classifier_path = resolve_path(classifier_path)
    regressor_path = resolve_path(regressor_path)

    has_classifier = classifier_path and os.path.exists(classifier_path)
    has_regressor = regressor_path and os.path.exists(regressor_path)

    # If we have at least the classifier locally, proceed with local models_dir
    if has_classifier:
        return models_dir

    # Try to download into /tmp/models
    tmp_models_dir = os.path.join("/tmp", "models")
    os.makedirs(tmp_models_dir, exist_ok=True)
    classifier_url = os.getenv("CLASSIFIER_MODEL_URL")
    regressor_url = os.getenv("REGRESSOR_MODEL_URL")

    if not classifier_url:
        raise HTTPException(status_code=500, detail=(
            "Classifier model is missing. Provide CLASSIFIER_MODEL_URL env var to download at runtime, "
            "or commit classifier .cbm into the repository."
        ))

    # Derive filenames from original metadata or generic names
    classifier_dest = os.path.join(tmp_models_dir, os.path.basename(classifier_path or "classifier_cb.cbm"))
    _download_file(classifier_url, classifier_dest)

    regressor_dest: Optional[str] = None
    if regressor_url:
        regressor_dest = os.path.join(tmp_models_dir, os.path.basename(regressor_path or "regressor_cb.cbm"))
        _download_file(regressor_url, regressor_dest)

    # Patch metadata.json in /tmp to point to downloaded models
    tmp_metadata_path = os.path.join(tmp_models_dir, "metadata.json")
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        # Use absolute paths so ModelBundle.load() can find them
        metadata["classifier_model_path"] = classifier_dest
        metadata["classifier_path"] = classifier_dest
        if regressor_dest:
            metadata["regressor_model_path"] = regressor_dest
            metadata["regressor_path"] = regressor_dest
        json.dump(metadata, f, indent=2)

    return tmp_models_dir


@app.on_event("startup")
def _load_bundle() -> None:
    global _bundle
    models_dir = os.path.join(PROJECT_ROOT, "models")
    use_dir = _ensure_models_available(models_dir)
    _bundle = ModelBundle(models_dir=use_dir)
    _bundle.load()


@app.post("/predict")
def predict(req: PredictRequest):
    if _bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not initialized")

    # Assemble a single-row patient record
    row = req.features.copy()
    org_col = _bundle.organism_col()
    ab_col = _bundle.antibiotic_col()
    if org_col:
        row[org_col] = req.organism
    if ab_col:
        row[ab_col] = req.antibiotic

    try:
        result = predict_for_new_patient(_bundle, row)
        ranked_df = rank_antibiotics(_bundle, row, req.organism, top_k=req.top_k or 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "resistance_probability": result["resistance_probability"],
        "predicted_time_to_resistance_days": result["predicted_time_to_resistance_days"],
        "top_antibiotics": ranked_df.to_dict(orient="records"),
        "feature_columns": _bundle.feature_columns(),
        "organism_column": org_col,
        "antibiotic_column": ab_col,
    }
