import os
from typing import Dict, Optional

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from backend.utils import load_dataset


class ModelBundle:
    def __init__(self, models_dir: str):
        self.models_dir = models_dir
        self.metadata: Dict = {}
        self.classifier: Optional[CatBoostClassifier] = None
        self.regressor: Optional[CatBoostRegressor] = None

    def load(self):
        meta_path = os.path.join(self.models_dir, "metadata.json")
        if not os.path.exists(meta_path):
            raise FileNotFoundError("metadata.json not found. Train models first.")
        self.metadata = pd.read_json(meta_path, typ="series").to_dict()

        cls_path = self.metadata.get("classifier_model_path")
        if not cls_path or not os.path.exists(cls_path):
            raise FileNotFoundError("Classifier model not found. Train classifier first.")
        self.classifier = CatBoostClassifier()
        self.classifier.load_model(cls_path)

        reg_path = self.metadata.get("regressor_model_path")
        if reg_path and os.path.exists(reg_path):
            self.regressor = CatBoostRegressor()
            self.regressor.load_model(reg_path)

    def feature_columns(self):
        return self.metadata.get("feature_cols", [])

    def organism_col(self):
        return self.metadata.get("organism_col")

    def antibiotic_col(self):
        return self.metadata.get("antibiotic_col")


def _prepare_row_df(bundle: ModelBundle, row: Dict) -> pd.DataFrame:
    # Ensure all feature columns exist; fill missing
    cols = bundle.feature_columns()
    df = pd.DataFrame([{c: row.get(c, np.nan) for c in cols}])
    return df[cols]


def predict_for_new_patient(bundle: ModelBundle, row: Dict) -> Dict:
    if bundle.classifier is None:
        raise RuntimeError("Models not loaded. Call bundle.load().")

    X = _prepare_row_df(bundle, row)

    prob_resistant = float(bundle.classifier.predict_proba(X)[:, 1][0])

    time_to_resistance = None
    if bundle.regressor is not None:
        time_to_resistance = float(bundle.regressor.predict(X)[0])

    return {
        "resistance_probability": prob_resistant,
        "predicted_time_to_resistance_days": time_to_resistance,
    }
