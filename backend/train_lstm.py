import argparse
import json
import os
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pack_padded_sequence

from sklearn.model_selection import GroupShuffleSplit

from backend.utils import (
    detect_group_column,
    detect_susceptibility_column,
)


CAT_FEATURES = [
    'medication_category', 'medication_name', 'antibiotic_class',
    'ordering_mode', 'culture_description', 'organism',
    'antibiotic', 'age', 'gender', 'prior_organism'
]
NUM_FEATURES = [
    'time_to_culturetime',
    'resistant_time_to_culturetime',
    'medication_time_to_culturetime',
    'prior_infecting_organism_days_to_culutre'
]


class ResistanceLSTM(nn.Module):
    def __init__(self, vocab_sizes: Dict[str, int], embed_dims: Dict[str, int], num_num: int):
        super().__init__()
        self.cat_features = list(vocab_sizes.keys())
        self.emb = nn.ModuleDict({
            c: nn.Embedding(vocab_sizes[c], embed_dims[c], padding_idx=0)
            for c in self.cat_features
        })
        input_dim = sum(embed_dims.values()) + num_num
        self.lstm = nn.LSTM(input_dim, 128, batch_first=True)
        self.fc = nn.Linear(128, 1)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        num_cat = len(self.cat_features)
        cats = x[:, :, :num_cat].long()
        nums = x[:, :, num_cat:]
        emb = torch.cat([self.emb[c](cats[:, :, i]) for i, c in enumerate(self.cat_features)], dim=2)
        feats = torch.cat([emb, nums], dim=2)
        packed = pack_padded_sequence(feats, lengths.cpu(), batch_first=True, enforce_sorted=False)
        packed_out, _ = self.lstm(packed)
        out, _ = torch.nn.utils.rnn.pad_packed_sequence(packed_out, batch_first=True, total_length=feats.size(1))
        logits = self.fc(out).squeeze(-1)
        return logits


class SeqDataset(Dataset):
    def __init__(self, X: np.ndarray, Y: np.ndarray, L: np.ndarray):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32)
        self.L = torch.tensor(L, dtype=torch.long)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, i):
        return self.X[i], self.Y[i], self.L[i]


def build_encoders_and_scalers(df: pd.DataFrame) -> Tuple[Dict[str, Dict[str, int]], Dict[str, int], Dict[str, Tuple[float, float]]]:
    label_encoders: Dict[str, Dict[str, int]] = {}
    vocab_sizes: Dict[str, int] = {}
    for c in CAT_FEATURES:
        df[c] = df[c].astype(str)
        classes = sorted(df[c].dropna().astype(str).unique().tolist())
        label_encoders[c] = {cls: i + 1 for i, cls in enumerate(classes)}  # 0 reserved for padding
        vocab_sizes[c] = len(classes) + 1
    scalers: Dict[str, Tuple[float, float]] = {}
    for n in NUM_FEATURES:
        vals = df[[n]].astype(float)
        mn = float(vals.min().values[0])
        mx = float(vals.max().values[0])
        if mx == mn:
            mx = mn + 1.0
        scalers[n] = (mn, mx)
    return label_encoders, vocab_sizes, scalers


def encode_scale(df: pd.DataFrame, encoders: Dict[str, Dict[str, int]], scalers: Dict[str, Tuple[float, float]]) -> np.ndarray:
    cats = np.stack([df[c].astype(str).map(lambda v: encoders[c].get(v, 0)).values for c in CAT_FEATURES], axis=1)
    nums = np.stack([df[n].astype(float).map(lambda v: (float(v) - scalers[n][0]) / (scalers[n][1] - scalers[n][0])).values for n in NUM_FEATURES], axis=1)
    x = np.concatenate([cats, nums], axis=1).astype(np.float32)
    return x


def build_sequences(df: pd.DataFrame, group_col: str, time_col: str, encoders, scalers) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    groups = df.groupby(group_col)
    max_seq_len = int(groups.size().max())
    X_list: List[np.ndarray] = []
    Y_list: List[np.ndarray] = []
    L_list: List[int] = []

    # Determine label column and mapping
    susc_col = detect_susceptibility_column(df)
    if not susc_col:
        raise ValueError("Could not detect susceptibility column for LSTM training.")

    for _, g in groups:
        if time_col in g.columns:
            try:
                g = g.sort_values(time_col)
            except Exception:
                pass
        g = g.reset_index(drop=True)
        seq_len = len(g)

        # Ensure feature columns exist
        for c in CAT_FEATURES:
            if c not in g.columns:
                g[c] = ""
        for n in NUM_FEATURES:
            if n not in g.columns:
                g[n] = 0.0

        x = encode_scale(g, encoders, scalers)
        # Binary target: Resistant=1, others=0
        y = g[susc_col].astype(str).map({"Resistant": 1.0, "Intermediate": 0.0, "Susceptible": 0.0}).fillna(0.0).astype(np.float32).values

        # Pad to max_seq_len
        x = np.pad(x, ((0, max_seq_len - seq_len), (0, 0)))
        y = np.pad(y, (0, max_seq_len - seq_len))

        X_list.append(x)
        Y_list.append(y)
        L_list.append(seq_len)

    X = np.array(X_list)
    Y = np.array(Y_list)
    L = np.array(L_list)
    return X, Y, L


def compute_pos_weight(Y: np.ndarray, L: np.ndarray) -> float:
    valid_pos = 0
    valid_neg = 0
    for y, l in zip(Y, L):
        valid_pos += int((y[:l] == 1).sum())
        valid_neg += int((y[:l] == 0).sum())
    if valid_pos == 0:
        return 1.0
    w = valid_neg / valid_pos
    return float(min(w, 3.0))


def main():
    parser = argparse.ArgumentParser(description="Train LSTM for antibiotic resistance sequences")
    parser.add_argument("--dataset", type=str, default="microbiology_combined_clean.csv")
    parser.add_argument("--models_dir", type=str, default="models")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=1e-3)
    args = parser.parse_args()

    os.makedirs(args.models_dir, exist_ok=True)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("Using device:", device)

    df = pd.read_csv(args.dataset)

    # Detect grouping and time columns
    group_col = detect_group_column(df)
    time_col = 'order_time_jittered_utc' if 'order_time_jittered_utc' in df.columns else None
    if time_col:
        try:
            df[time_col] = pd.to_datetime(df[time_col])
        except Exception:
            pass

    # Build encoders/scalers and sequences
    encoders, vocab_sizes, scalers = build_encoders_and_scalers(df.copy())
    X, Y, L = build_sequences(df.copy(), group_col, time_col or '', encoders, scalers)

    # Group-aware train/test split on sequences
    # Create group-level indices consistent with grouping above
    unique_groups = df[group_col].dropna().unique()
    # Map each sequence to a group id
    seq_groups = df.groupby(group_col).apply(lambda g: g[group_col].iloc[0]).reset_index(drop=True)
    # Use GroupShuffleSplit on the list of sequences
    gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
    # Build a synthetic array of indices for sequences
    seq_idx = np.arange(len(L))
    train_idx, test_idx = next(gss.split(seq_idx, groups=seq_idx))

    X_train, X_test = X[train_idx], X[test_idx]
    Y_train, Y_test = Y[train_idx], Y[test_idx]
    L_train, L_test = L[train_idx], L[test_idx]

    train_loader = DataLoader(SeqDataset(X_train, Y_train, L_train), batch_size=args.batch_size, shuffle=True)
    test_loader = DataLoader(SeqDataset(X_test, Y_test, L_test), batch_size=args.batch_size)

    # Embedding dims heuristic
    embed_dims = {c: min(50, vocab_sizes[c] // 2 + 1) for c in CAT_FEATURES}
    model = ResistanceLSTM(vocab_sizes, embed_dims, num_num=len(NUM_FEATURES)).to(device)

    pos_weight = torch.tensor(compute_pos_weight(Y_train, L_train), dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)

    for epoch in range(args.epochs):
        model.train()
        total_loss = 0.0
        for x, y, l in train_loader:
            x, y, l = x.to(device), y.to(device), l.to(device)
            optimizer.zero_grad()
            logits = model(x, l)
            bsz, max_len = y.shape
            mask = (torch.arange(max_len, device=device).expand(bsz, max_len) < l.unsqueeze(1))
            loss = criterion(logits.reshape(-1)[mask.reshape(-1)], y.reshape(-1)[mask.reshape(-1)])
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch [{epoch+1}/{args.epochs}] Loss: {total_loss/len(train_loader):.4f}")

    # Quick evaluation
    model.eval()
    with torch.no_grad():
        x, y, l = next(iter(test_loader))
        x, y, l = x.to(device), y.to(device), l.to(device)
        logits = model(x, l)
        probs = torch.sigmoid(logits)
        print("Eval batch probs shape:", probs.shape)

    # Save model and meta
    model_path = os.path.join(args.models_dir, 'lstm_model.pt')
    torch.save({'state_dict': model.state_dict()}, model_path)
    meta = {
        'cat_features': CAT_FEATURES,
        'num_features': NUM_FEATURES,
        'vocab_sizes': vocab_sizes,
        'embed_dims': embed_dims,
        'encoders': {c: list(encoders[c].keys()) for c in CAT_FEATURES},
        'scalers': scalers,
        'group_col': group_col,
        'time_col': time_col,
    }
    with open(os.path.join(args.models_dir, 'lstm_meta.json'), 'w', encoding='utf-8') as f:
        json.dump(meta, f, indent=2)
    print("Saved LSTM model and metadata to:", args.models_dir)


if __name__ == "__main__":
    main()
