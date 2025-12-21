from typing import Dict, Optional

import numpy as np

from backend.infer_catboost import CatBoostInfer


def ensemble_predict(catboost: CatBoostInfer, lstm: Optional[object], row: Dict, sequence_df, alpha: float = 0.6) -> Dict:
    """Compute CatBoost prob, LSTM prob, ensemble prob and time-to-resistance.

    alpha: weight for CatBoost probability.
    """
    c_out = catboost.predict_catboost(row)
    p_cb = float(c_out["p_catboost"]) if c_out["p_catboost"] is not None else np.nan
    ttr_days = c_out["ttr_days"]

    p_lstm = float(lstm.predict_lstm(sequence_df)) if sequence_df is not None else np.nan

    # Handle NaNs: if one model missing, fallback to the other
    if np.isnan(p_cb) and not np.isnan(p_lstm):
        p_final = p_lstm
    elif np.isnan(p_lstm) and not np.isnan(p_cb):
        p_final = p_cb
    elif np.isnan(p_cb) and np.isnan(p_lstm):
        p_final = np.nan
    else:
        p_final = float(alpha * p_cb + (1.0 - alpha) * p_lstm)

    return {
        "p_catboost": p_cb,
        "p_lstm": p_lstm,
        "p_final": p_final,
        "ttr_days": ttr_days,
    }
