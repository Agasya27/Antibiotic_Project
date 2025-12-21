import argparse
import os
from typing import Dict, Tuple, List

import numpy as np
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from sklearn.model_selection import GroupKFold, GroupShuffleSplit

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
    iterations: int = 1500,
    depth: int = 6,
    balanced: bool = True,
    optimize_threshold: bool = True,
    cv_folds: int = 5,
) -> Dict:
    y = build_classification_target(df, susceptibility_col, mode)
    # Drop rows with missing target
    mask = ~y.isna()
    X = df.loc[mask, feature_cols]
    y = y.loc[mask].astype(int)
    groups = df.loc[mask, group_col]

    # Cross-validation across groups for robust metrics
    cat_idx = [X.columns.get_loc(c) for c in cat_cols]
    cv = GroupKFold(n_splits=cv_folds)
    oof_true: List[np.ndarray] = []
    oof_prob: List[np.ndarray] = []
    cv_metrics: List[Dict[str, float]] = []

    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        clf = CatBoostClassifier(
            loss_function="Logloss",
            eval_metric="AUC",
            iterations=iterations,
            depth=depth,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100,
            auto_class_weights="Balanced" if balanced else None,
        )
        clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx)
        y_val_prob = clf.predict_proba(X_val)[:, 1]
        cv_metrics.append(compute_classification_metrics(y_val, y_val_prob))
        oof_true.append(y_val.values)
        oof_prob.append(y_val_prob)

    # Aggregate CV metrics
    def _avg(name: str) -> float:
        vals = [m[name] for m in cv_metrics if name in m]
        return float(np.mean(vals)) if vals else float("nan")

    best_threshold = 0.5
    if optimize_threshold and len(oof_true) > 0:
        y_true_all = np.concatenate(oof_true)
        y_prob_all = np.concatenate(oof_prob)
        best_threshold, _ = _optimize_f1_threshold(pd.Series(y_true_all), y_prob_all)

    # Final model trained with a small holdout for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=max(0.1, test_size), random_state=random_state)
    tr_idx, val_idx = next(gss.split(X, y, groups))
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    final_clf = CatBoostClassifier(
        loss_function="Logloss",
        eval_metric="AUC",
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
        auto_class_weights="Balanced" if balanced else None,
    )
    final_clf.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx)
    holdout_prob = final_clf.predict_proba(X_val)[:, 1]

    best_metrics = {
        "roc_auc": _avg("roc_auc"),
        "f1": _avg("f1"),
        "precision": _avg("precision"),
        "recall": _avg("recall"),
    }
    holdout_metrics = compute_classification_metrics(y_val, holdout_prob, threshold=best_threshold)

    return {
        "model": final_clf,
        "metrics": best_metrics,
        "holdout_metrics": holdout_metrics,
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
    iterations: int = 1200,
    depth: int = 6,
    cv_folds: int = 5,
) -> Dict:
    # Only rows where time_to_resistance is known (non-censored)
    mask = df[time_col].notna()
    X = df.loc[mask, feature_cols]
    y = df.loc[mask, time_col].astype(float)
    groups = df.loc[mask, group_col]

    if len(X) < 100:
        iterations = min(iterations, 300)

    cat_idx = [X.columns.get_loc(c) for c in cat_cols]

    cv = GroupKFold(n_splits=cv_folds)
    rmses: List[float] = []
    for train_idx, val_idx in cv.split(X, y, groups):
        X_tr, X_val = X.iloc[train_idx], X.iloc[val_idx]
        y_tr, y_val = y.iloc[train_idx], y.iloc[val_idx]
        reg = CatBoostRegressor(
            loss_function="RMSE",
            iterations=iterations,
            depth=depth,
            learning_rate=0.05,
            random_seed=random_state,
            verbose=False,
            allow_writing_files=False,
            od_type="Iter",
            od_wait=100,
        )
        reg.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx)
        y_val_pred = reg.predict(X_val)
        m = compute_regression_metrics(y_val, y_val_pred)
        rmses.append(m["rmse"])

    cv_rmse = float(np.mean(rmses)) if rmses else float("nan")

    # Final model with small holdout for early stopping
    gss = GroupShuffleSplit(n_splits=1, test_size=max(0.1, test_size), random_state=random_state)
    tr_idx, val_idx = next(gss.split(X, y, groups))
    X_tr, X_val = X.iloc[tr_idx], X.iloc[val_idx]
    y_tr, y_val = y.iloc[tr_idx], y.iloc[val_idx]

    final_reg = CatBoostRegressor(
        loss_function="RMSE",
        iterations=iterations,
        depth=depth,
        learning_rate=0.05,
        random_seed=random_state,
        verbose=False,
        allow_writing_files=False,
        od_type="Iter",
        od_wait=100,
    )
    final_reg.fit(X_tr, y_tr, eval_set=(X_val, y_val), cat_features=cat_idx)
    holdout_pred = final_reg.predict(X_val)
    holdout_metrics = compute_regression_metrics(y_val, holdout_pred)

    return {
        "model": final_reg,
        "metrics": {"rmse_cv": cv_rmse},
        "holdout_metrics": holdout_metrics,
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
    parser.add_argument("--iterations_clf", type=int, default=1500, help="Classifier iterations with early stopping")
    parser.add_argument("--iterations_reg", type=int, default=1200, help="Regressor iterations with early stopping")
    parser.add_argument("--depth_clf", type=int, default=6, help="Classifier tree depth")
    parser.add_argument("--depth_reg", type=int, default=6, help="Regressor tree depth")
    parser.add_argument("--cv_folds", type=int, default=5, help="Number of group-aware CV folds")
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
        iterations=args.iterations_clf,
        depth=args.depth_clf,
        cv_folds=args.cv_folds,
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
            iterations=args.iterations_reg,
            depth=args.depth_reg,
            cv_folds=args.cv_folds,
        )
        reg_model_path = os.path.join(args.models_dir, "regressor_cb.cbm")
        reg_out["model"].save_model(reg_model_path)
        reg_metrics = {**reg_out.get("metrics", {}), **reg_out.get("holdout_metrics", {})}

    # Save metrics
    metrics = {
        "classifier_cv": cls_out["metrics"],
        "classifier_holdout": cls_out.get("holdout_metrics", {}),
        "classifier_best_threshold": cls_out.get("best_threshold", 0.5),
        "regressor_cv": reg_metrics if reg_metrics else {},
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
        "cv_folds": args.cv_folds,
        "iterations_clf": args.iterations_clf,
        "iterations_reg": args.iterations_reg,
        "depth_clf": args.depth_clf,
        "depth_reg": args.depth_reg,
    }
    meta.update(collect_metadata(df, organism_col, antibiotic_col))

    save_json(meta, os.path.join(args.models_dir, "metadata.json"))

    print("Training complete. Models and metadata saved to:", args.models_dir)
    print("Classifier metrics:", metrics["classifier"]) 
    if reg_metrics:
        print("Regressor metrics:", metrics["regressor"]) 


if __name__ == "__main__":
    main()
