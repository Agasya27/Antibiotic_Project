from typing import Dict, Optional

import numpy as np
import pandas as pd

from backend.infer import ModelBundle


class CatBoostInfer:
    def __init__(self, models_dir: str = "models"):
        self.bundle = ModelBundle(models_dir)
        self.bundle.load()

    def feature_columns(self):
        return self.bundle.feature_columns()

    def organism_col(self):
        return self.bundle.organism_col()

    def antibiotic_col(self):
        return self.bundle.antibiotic_col()

    def predict_catboost(self, row: Dict) -> Dict:
        """Return CatBoost classifier prob and regressor time-to-resistance."""
        from backend.infer import predict_for_new_patient

        out = predict_for_new_patient(self.bundle, row)
        return {
            "p_catboost": out["resistance_probability"],
            "ttr_days": out["predicted_time_to_resistance_days"],
        }
