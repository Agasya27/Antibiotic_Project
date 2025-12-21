import argparse
import os
from typing import List

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score

from backend.utils import detect_group_column, detect_susceptibility_column
from backend.infer_lstm import LSTMInfer


def main():
    parser = argparse.ArgumentParser(description="Evaluate LSTM model on last-timestep resistance")
    parser.add_argument("--dataset", type=str, default="microbiology_combined_clean.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()

    df = pd.read_csv(args.dataset)
    group_col = detect_group_column(df)
    susc_col = detect_susceptibility_column(df)
    if susc_col is None:
        raise ValueError("Could not detect susceptibility column for evaluation.")

    # Sort by time if available
    time_col = 'order_time_jittered_utc' if 'order_time_jittered_utc' in df.columns else None
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            pass

    lstm = LSTMInfer(models_dir=args.models_dir, dataset_csv=args.dataset)

    probs: List[float] = []
    labels: List[int] = []

    for _, g in df.groupby(group_col):
        if time_col:
            try:
                g = g.sort_values(time_col)
            except Exception:
                pass
        g = g.reset_index(drop=True)
        # True label at last timestep
        true = g[susc_col].iloc[-1]
        y_last = 1 if str(true) == "Resistant" else 0
        labels.append(y_last)
        # Probability at last timestep
        p = lstm.predict_lstm(g)
        probs.append(float(p))

    y_true = np.array(labels)
    y_prob = np.array(probs)
    y_pred = (y_prob >= args.threshold).astype(int)

    # Metrics
    metrics = {
        "roc_auc": float(roc_auc_score(y_true, y_prob)) if len(np.unique(y_true)) > 1 else float('nan'),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "f1": float(f1_score(y_true, y_pred, zero_division=0)),
        "threshold": args.threshold,
        "samples": int(len(y_true)),
    }

    print("LSTM Evaluation (last timestep):")
    for k, v in metrics.items():
        print(f"- {k}: {v}")


if __name__ == "__main__":
    main()
