from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from backend.infer import ModelBundle, predict_for_new_patient


def rank_antibiotics(bundle: ModelBundle, patient_row: Dict, organism_value: Optional[str], top_k: int = 10) -> pd.DataFrame:
    """
    For a given patient + organism, score all antibiotics in metadata.
    Rank by lowest resistance probability, then longest predicted time to resistance.
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
        out = predict_for_new_patient(bundle, row)
        results.append({
            "antibiotic": abx,
            "resistance_probability": out["resistance_probability"],
            "predicted_time_to_resistance_days": out["predicted_time_to_resistance_days"],
        })

    df = pd.DataFrame(results)
    # Sort: lowest probability ascending, then time descending (None handled as -inf)
    time_series = df["predicted_time_to_resistance_days"].fillna(-np.inf)
    df = df.assign(_time_sort=time_series)
    df = df.sort_values(by=["resistance_probability", "_time_sort"], ascending=[True, False]).drop(columns=["_time_sort"]) 

    if top_k is not None and top_k > 0:
        df = df.head(top_k)
    return df
