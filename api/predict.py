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

from backend.infer import ModelBundle
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

    if has_classifier and has_regressor:
        return models_dir

    # Try to download into /tmp/models
    tmp_models_dir = os.path.join("/tmp", "models")
    os.makedirs(tmp_models_dir, exist_ok=True)
    classifier_url = os.getenv("CLASSIFIER_MODEL_URL")
    regressor_url = os.getenv("REGRESSOR_MODEL_URL")

    if not classifier_url or not regressor_url:
        raise HTTPException(status_code=500, detail=(
            "Model binaries are missing. Provide CLASSIFIER_MODEL_URL and REGRESSOR_MODEL_URL env vars "
            "to download models at runtime, or commit .cbm files into the repository."
        ))

    # Derive filenames from original metadata or generic names
    classifier_dest = os.path.join(tmp_models_dir, os.path.basename(classifier_path or "classifier_cb.cbm"))
    regressor_dest = os.path.join(tmp_models_dir, os.path.basename(regressor_path or "regressor_cb.cbm"))

    _download_file(classifier_url, classifier_dest)
    _download_file(regressor_url, regressor_dest)

    # Patch metadata.json in /tmp to point to downloaded models
    tmp_metadata_path = os.path.join(tmp_models_dir, "metadata.json")
    with open(tmp_metadata_path, "w", encoding="utf-8") as f:
        metadata["classifier_model_path"] = os.path.basename(classifier_dest)
        metadata["regressor_model_path"] = os.path.basename(regressor_dest)
        metadata["classifier_path"] = metadata["classifier_model_path"]
        metadata["regressor_path"] = metadata["regressor_model_path"]
        json.dump(metadata, f, indent=2)

    return tmp_models_dir


@app.on_event("startup")
def _load_bundle() -> None:
    global _bundle
    models_dir = os.path.join(PROJECT_ROOT, "models")
    use_dir = _ensure_models_available(models_dir)
    _bundle = ModelBundle(models_dir=use_dir)


@app.post("/predict")
def predict(req: PredictRequest):
    if _bundle is None:
        raise HTTPException(status_code=500, detail="Model bundle not initialized")

    # Assemble a single-row patient record
    row = req.features.copy()
    row[_bundle.org_col] = req.organism
    row[_bundle.ab_col] = req.antibiotic

    try:
        result = _bundle.predict_for_new_patient(row)
        ranked = rank_antibiotics(_bundle, row, req.organism, top_k=req.top_k or 5)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Inference error: {e}")

    return {
        "probability_resistance": result["probability_resistance"],
        "predicted_time_to_resistance": result["predicted_time_to_resistance"],
        "top_antibiotics": ranked,
        "feature_columns": _bundle.feature_cols,
        "categorical_features": _bundle.cat_features,
        "organism_column": _bundle.org_col,
        "antibiotic_column": _bundle.ab_col,
    }
