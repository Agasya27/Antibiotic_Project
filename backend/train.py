import argparse
import os
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor

from backend.utils import (
    load_dataset,
    detect_group_column,
    detect_susceptibility_column,
    build_classification_target,
    detect_time_to_resistance_column,
    detect_organism_column,
    detect_antibiotic_column,
    get_feature_columns,
    get_categorical_features,
    group_train_test_split,
    compute_classification_metrics,
    compute_regression_metrics,
    save_json,
    collect_metadata,
)


DEFAULT_DATASET = "microbiology_combined_clean.csv"
DEFAULT_MODELS_DIR = "models"


def _optimize_f1_threshold(y_true, y_prob) -> Tuple[float, Dict[str, float]]:
    # Scan thresholds 0.05..0.95 to maximize F1
    best_t, best_f1 = 0.5, -1.0
    best_metrics = None
    for t in np.linspace(0.05, 0.95, 19):
        m = compute_classification_metrics(y_true, y_prob, threshold=t)
        if m["f1"] > best_f1:
            best_f1 = m["f1"]
            best_t = float(t)
            best_metrics = m
    return best_t, best_metrics


def train_classifier(
    df: pd.DataFrame,
    susceptibility_col: str,
    feature_cols: list,
    cat_cols: list,
    group_col: str,
    mode: str,
    test_size: float,
    random_state: int = 42,
    iterations: int = 1000,
    balanced: bool = True,
    optimize_threshold: bool = True,
) -> Dict:
    y = build_classification_target(df, susceptibility_col, mode)
    # Drop rows with missing target
    mask = ~y.isna()
    X = df.loc[mask, feature_cols]
    y = y.loc[mask].astype(int)
    groups = df.loc[mask, group_col]

    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups, test_size, random_state)

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=50,
        auto_class_weights="Balanced" if balanced else None,
    )

    clf.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_idx)
    y_prob = clf.predict_proba(X_test)[:, 1]
    metrics = compute_classification_metrics(y_test, y_prob)
    best_threshold = 0.5
    best_metrics = metrics
    if optimize_threshold:
        best_threshold, best_metrics = _optimize_f1_threshold(y_test, y_prob)

    return {
        "model": clf,
        "metrics": best_metrics,
        "best_threshold": best_threshold,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }


def train_regressor(
    df: pd.DataFrame,
    time_col: str,
    feature_cols: list,
    cat_cols: list,
    group_col: str,
    test_size: float,
    random_state: int = 42,
    iterations: int = 800,
) -> Dict:
    # Only rows where time_to_resistance is known (non-censored)
    mask = df[time_col].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, time_col].astype(float)
    groups = df.loc[mask, group_col]

    if len(X) < 100:
        iterations = min(iterations, 300)

    X_train, X_test, y_train, y_test = group_train_test_split(X, y, groups, test_size, random_state)

    cat_idx = [X_train.columns.get_loc(c) for c in cat_cols]

    reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=6,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=100,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=50,
    )

    reg.fit(X_train, y_train, eval_set=(X_test, y_test), cat_features=cat_idx)
    y_pred = reg.predict(X_test)
    metrics = compute_regression_metrics(y_test, y_pred)

    return {
        "model": reg,
        "metrics": metrics,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
    }


def main():
    parser = argparse.ArgumentParser(description="Train CatBoost models for antibiotic resistance")
    parser.add_argument("--dataset", type=str, default=DEFAULT_DATASET, help="Path to CSV dataset")
    parser.add_argument("--models_dir", type=str, default=DEFAULT_MODELS_DIR, help="Output models directory")
    parser.add_argument("--mode", type=str, default="binary_rs", choices=["binary_rs", "binary_ni"], help="Classifier target mode")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test size for group split")
    parser.add_argument("--random_state", type=int, default=42)
    parser.add_argument("--iterations", type=int, default=1000, help="Classifier iterations with early stopping")
    parser.add_argument("--balanced", action="store_true", help="Enable auto class weights 'Balanced' for classifier")
    parser.add_argument("--no-threshold-opt", action="store_true", help="Disable F1 threshold optimization")

    args = parser.parse_args()
    os.makedirs(args.models_dir, exist_ok=True)

    df = load_dataset(args.dataset)

    group_col = detect_group_column(df)
    susceptibility_col = detect_susceptibility_column(df)
    if not susceptibility_col:
        raise ValueError("Could not detect susceptibility column. Ensure values are 'Susceptible', 'Intermediate', 'Resistant'.")

    time_col = detect_time_to_resistance_column(df)
    organism_col = detect_organism_column(df)
    antibiotic_col = detect_antibiotic_column(df)

    target_cols = [susceptibility_col, time_col]
    feature_cols = get_feature_columns(df, target_cols, group_col, organism_col, antibiotic_col)
    cat_cols = get_categorical_features(df, feature_cols)

    # Train classifier
    cls_out = train_classifier(
        df,
        susceptibility_col,
        feature_cols,
        cat_cols,
        group_col,
        args.mode,
        args.test_size,
        args.random_state,
        iterations=args.iterations,
        balanced=args.balanced,
        optimize_threshold=not args.no_threshold_opt,
    )

    # Save classifier
    cls_model_path = os.path.join(args.models_dir, "classifier_cb.cbm")
    cls_out["model"].save_model(cls_model_path)

    # Train regressor if time_col exists
    reg_metrics = None
    reg_model_path = None
    if time_col:
        reg_out = train_regressor(
            df,
            time_col,
            feature_cols,
            cat_cols,
            group_col,
            args.test_size,
            args.random_state,
            iterations=800,
        )
        reg_model_path = os.path.join(args.models_dir, "regressor_cb.cbm")
        reg_out["model"].save_model(reg_model_path)
        reg_metrics = reg_out["metrics"]

    # Save metrics
    metrics = {
        "classifier": cls_out["metrics"],
        "classifier_best_threshold": cls_out.get("best_threshold", 0.5),
        "regressor": reg_metrics if reg_metrics else {},
    }
    save_json(metrics, os.path.join(args.models_dir, "metrics.json"))

    # Save metadata
    meta = {
        "group_col": group_col,
        "susceptibility_col": susceptibility_col,
        "time_col": time_col,
        "organism_col": organism_col,
        "antibiotic_col": antibiotic_col,
        "feature_cols": feature_cols,
        "categorical_feature_cols": cat_cols,
        "classifier_model_path": cls_model_path,
        "regressor_model_path": reg_model_path,
        "classification_mode": args.mode,
        "best_threshold": cls_out.get("best_threshold", 0.5),
    }
    meta.update(collect_metadata(df, organism_col, antibiotic_col))

    save_json(meta, os.path.join(args.models_dir, "metadata.json"))

    print("Training complete. Models and metadata saved to:", args.models_dir)
    print("Classifier metrics:", metrics["classifier"]) 
    if reg_metrics:
        print("Regressor metrics:", metrics["regressor"]) 


if __name__ == "__main__":
    main()
