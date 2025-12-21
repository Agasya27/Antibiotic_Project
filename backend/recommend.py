from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.infer import ModelBundle, predict_for_new_patient
from backend.ensemble import ensemble_predict


def rank_antibiotics(bundle: ModelBundle, lstm: Optional[object], patient_row: Dict, patient_sequence: Optional[pd.DataFrame], organism_value: Optional[str], alpha: float = 0.6, top_k: int = 10) -> pd.DataFrame:
    """
    For a given patient + organism, score all antibiotics using ensemble.
    Rank by lowest final probability, then longest predicted time to resistance.
    If LSTM is unavailable or sequence missing, falls back to CatBoost-only.
    """
    antibiotics: List[str] = bundle.metadata.get("antibiotics", [])
    abx_col = bundle.antibiotic_col()
    org_col = bundle.organism_col()

    results = []
    for abx in antibiotics:
        row = dict(patient_row)
        if abx_col:
            row[abx_col] = abx
        if organism_value is not None and org_col:
            row[org_col] = organism_value
        # If sequence provided, clone it and set the last timestep antibiotic to candidate
        seq_df = None
        if patient_sequence is not None:
            seq_df = patient_sequence.copy()
            if abx_col and abx_col in seq_df.columns and not seq_df.empty:
                seq_df.iloc[-1, seq_df.columns.get_loc(abx_col)] = abx
        # Ensemble prediction
        # Wrap CatBoostInfer with the bundle
        from backend.infer_catboost import CatBoostInfer
        catboost = CatBoostInfer(models_dir=bundle.models_dir)
        ens = ensemble_predict(catboost, lstm, row, seq_df, alpha=alpha) if lstm else {
            "p_catboost": predict_for_new_patient(bundle, row)["resistance_probability"],
            "p_lstm": None,
            "p_final": predict_for_new_patient(bundle, row)["resistance_probability"],
            "ttr_days": predict_for_new_patient(bundle, row)["predicted_time_to_resistance_days"],
        }
        results.append({
            "antibiotic": abx,
            "p_catboost": ens["p_catboost"],
            "p_lstm": ens["p_lstm"],
            "p_final": ens["p_final"],
            "predicted_time_to_resistance_days": ens["ttr_days"],
        })

    df = pd.DataFrame(results)
    # Sort: lowest final probability ascending, then time descending (None handled as -inf)
    time_series = df["predicted_time_to_resistance_days"].fillna(-np.inf)
    df = df.assign(_time_sort=time_series)
    df = df.sort_values(by=["p_final", "_time_sort"], ascending=[True, False]).drop(columns=["_time_sort"]) 

    if top_k is not None and top_k > 0:
        df = df.head(top_k)
    return df
