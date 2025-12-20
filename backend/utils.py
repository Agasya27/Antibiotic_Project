import json
import os
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error


SUSCEPTIBILITY_LABELS = {"Susceptible", "Intermediate", "Resistant"}


def load_dataset(csv_path: str) -> pd.DataFrame:
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Dataset not found at {csv_path}")
    df = pd.read_csv(csv_path)
    return df


def detect_group_column(df: pd.DataFrame) -> str:
    for col in ["patient_id", "encounter_id"]:
        if col in df.columns:
            return col
    # Fallback: if neither exists, create synthetic group by patient-like hash
    # But warn the user
    return "patient_id" if "patient_id" in df.columns else df.columns[0]


def detect_susceptibility_column(df: pd.DataFrame) -> Optional[str]:
    # Prefer common names
    candidates = [
        "susceptibility", "antibiotic_susceptibility", "ast_result", "result", "final_result"
    ]
    for c in candidates:
        if c in df.columns:
            vals = set(str(v) for v in df[c].dropna().unique())
            # Accept if the column contains at least the expected labels, even with extras like 'Null'
            if SUSCEPTIBILITY_LABELS.issubset(vals) or vals.issubset(SUSCEPTIBILITY_LABELS):
                return c
    # Generic detection: any column where expected labels appear predominantly
    for c in df.columns:
        vals = set(str(v) for v in df[c].dropna().unique())
        if len(vals) == 0:
            continue
        intersection = SUSCEPTIBILITY_LABELS.intersection(vals)
        if len(intersection) >= 2:  # at least two of the canonical labels present
            return c
    return None


def build_classification_target(
    df: pd.DataFrame, susceptibility_col: str, mode: str = "binary_rs"
) -> pd.Series:
    # Map labels according to mode
    s = df[susceptibility_col].astype(str)
    if mode == "binary_rs":
        # Resistant=1, Susceptible=0, Intermediate=0
        return s.map({"Resistant": 1, "Susceptible": 0, "Intermediate": 0}).fillna(np.nan)
    elif mode == "binary_ni":
        # (R + I) vs S
        return s.map({"Resistant": 1, "Intermediate": 1, "Susceptible": 0}).fillna(np.nan)
    else:
        raise ValueError("Unsupported classification mode. Use 'binary_rs' or 'binary_ni'.")


def detect_time_to_resistance_column(df: pd.DataFrame) -> Optional[str]:
    candidates = [
        "time_to_resistance_days",
        "resistant_time_to_culturetime",
        "time_to_resistance",
        "days_to_resistance",
        "resistance_after_days",
    ]
    for c in candidates:
        if c in df.columns:
            return c
    return None


def detect_organism_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["organism", "organism_species", "organism_name"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: any column containing 'organism'
    for c in df.columns:
        if "organism" in c.lower():
            return c
    return None


def detect_antibiotic_column(df: pd.DataFrame) -> Optional[str]:
    candidates = ["antibiotic_name", "candidate_antibiotic", "antibiotic", "abx_name"]
    for c in candidates:
        if c in df.columns:
            return c
    # Fallback: any column containing 'antibiotic'
    for c in df.columns:
        if "antibiotic" in c.lower():
            return c
    return None


LEAKAGE_COLS_HINTS = {
    "implied_susceptibility",
    "future_mic",
    "post_culture",
}


def is_leakage_col(col: str) -> bool:
    cl = col.lower()
    return any(h in cl for h in LEAKAGE_COLS_HINTS)


def get_feature_columns(
    df: pd.DataFrame,
    target_cols: List[str],
    group_col: Optional[str],
    organism_col: Optional[str],
    antibiotic_col: Optional[str],
) -> List[str]:
    exclude = set([c for c in target_cols if c])
    if group_col:
        exclude.add(group_col)
    # Common timestamps / ids to exclude
    for c in ["culture_time", "encounter_id", "patient_id"]:
        if c in df.columns:
            exclude.add(c)
    # Exclude clear leakage columns by hint
    for c in df.columns:
        if is_leakage_col(c):
            exclude.add(c)
    features = [c for c in df.columns if c not in exclude]
    return features


def get_categorical_features(df: pd.DataFrame, feature_cols: List[str]) -> List[str]:
    cats = [c for c in feature_cols if str(df[c].dtype) in ("object", "category")]
    return cats


def group_train_test_split(
    X: pd.DataFrame, y: pd.Series, groups: pd.Series, test_size: float = 0.2, random_state: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    splitter = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    train_idx, test_idx = next(splitter.split(X, y, groups))
    return X.iloc[train_idx], X.iloc[test_idx], y.iloc[train_idx], y.iloc[test_idx]


def compute_classification_metrics(y_true: pd.Series, y_prob: np.ndarray, threshold: float = 0.5) -> Dict[str, float]:
    # y_prob is prob of class 1
    y_pred = (y_prob >= threshold).astype(int)
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)),
        "f1": float(f1_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred)),
        "recall": float(recall_score(y_true, y_pred)),
    }
    return metrics


def compute_regression_metrics(y_true: pd.Series, y_pred: np.ndarray) -> Dict[str, float]:
    rmse = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    return {"rmse": rmse}


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)


def collect_metadata(df: pd.DataFrame, organism_col: Optional[str], antibiotic_col: Optional[str]) -> Dict:
    organisms = []
    antibiotics = []
    if organism_col and organism_col in df.columns:
        organisms = sorted([str(v) for v in df[organism_col].dropna().unique()])
    # Detect potential antibiotic identity columns as well
    if antibiotic_col and antibiotic_col in df.columns:
        antibiotics = sorted([str(v) for v in df[antibiotic_col].dropna().unique()])
    else:
        # As fallback, scan for any 'antibiotic' in column names and aggregate uniques
        for c in df.columns:
            if "antibiotic" in c.lower():
                antibiotics = sorted([str(v) for v in df[c].dropna().unique()])
                break
    return {
        "organisms": organisms,
        "antibiotics": antibiotics,
    }
