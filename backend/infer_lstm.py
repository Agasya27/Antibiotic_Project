from typing import Dict, List, Tuple

import os
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ResistanceLSTM(nn.Module):
    """Minimal LSTM architecture matching saved weights.
    Embeds categorical features and concatenates with numeric features per timestep.
    """
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
        out, _ = pad_packed_sequence(packed_out, batch_first=True, total_length=feats.size(1))
        logits = self.fc(out).squeeze(-1)
        return logits


class LSTMInfer:
    """Loads LSTM model and provides sequence preprocessing and probability inference.

    The sequence is expected as a pandas DataFrame with columns:
      - categorical: medication_category, medication_name, antibiotic_class,
        ordering_mode, culture_description, organism, antibiotic, age, gender, prior_organism
      - numeric: time_to_culturetime, resistant_time_to_culturetime,
        medication_time_to_culturetime, prior_infecting_organism_days_to_culutre
      - optional time column order_time_jittered_utc for sorting
    """

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

    def __init__(self, models_dir: str = "models", dataset_csv: str = "microbiology_combined_clean.csv", device: str = None):
        self.models_dir = models_dir
        self.dataset_csv = dataset_csv
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # Prefer training metadata if available
        meta_path = os.path.join(models_dir, "lstm_meta.json")
        if os.path.exists(meta_path):
            with open(meta_path, "r", encoding="utf-8") as f:
                meta = json.load(f)
            self.CAT_FEATURES = meta.get('cat_features', self.CAT_FEATURES)
            self.NUM_FEATURES = meta.get('num_features', self.NUM_FEATURES)
            classes_map = meta.get('encoders', {})
            self.label_encoders = {c: {cls: i + 1 for i, cls in enumerate(classes_map.get(c, []))} for c in self.CAT_FEATURES}
            self.vocab_sizes = {c: int(meta.get('vocab_sizes', {}).get(c, 1)) for c in self.CAT_FEATURES}
            self.scalers = {k: tuple(v) for k, v in meta.get('scalers', {}).items()}
            self.embed_dims = {c: int(meta.get('embed_dims', {}).get(c, min(50, self.vocab_sizes[c] // 2 + 1))) for c in self.CAT_FEATURES}
        else:
            self._build_encoders()
        self._load_model()

    def _build_encoders(self):
        # Fit label encoders and numeric scalers on dataset for inference consistency
        df = pd.read_csv(self.dataset_csv)
        self.label_encoders = {}
        self.vocab_sizes = {}
        for c in self.CAT_FEATURES:
            df[c] = df[c].astype(str)
            classes = sorted(df[c].dropna().astype(str).unique().tolist())
            # index: class -> id starting at 1 (0 reserved for padding)
            self.label_encoders[c] = {cls: i + 1 for i, cls in enumerate(classes)}
            self.vocab_sizes[c] = len(classes) + 1
        self.scalers = {}
        for n in self.NUM_FEATURES:
            vals = df[[n]].astype(float)
            mn = float(vals.min().values[0])
            mx = float(vals.max().values[0])
            if mx == mn:
                mx = mn + 1.0
            self.scalers[n] = (mn, mx)

        # Embedding dims similar to training heuristic
        self.embed_dims = {c: min(50, self.vocab_sizes[c] // 2 + 1) for c in self.CAT_FEATURES}

    def _load_model(self):
        path = os.path.join(self.models_dir, "lstm_model.pt")
        if not os.path.exists(path):
            raise FileNotFoundError("LSTM model weights not found at models/lstm_model.pt")
        self.model = ResistanceLSTM(self.vocab_sizes, self.embed_dims, num_num=len(self.NUM_FEATURES)).to(self.device)
        state = torch.load(path, map_location=self.device)
        # Support both raw state_dict and wrapped dict
        sd = state.get('state_dict', state)
        self.model.load_state_dict(sd)
        self.model.eval()

    def _encode_cat(self, col: str, val: str) -> int:
        enc = self.label_encoders[col]
        return enc.get(str(val), 0)  # unknown -> padding

    def _scale_num(self, col: str, val: float) -> float:
        mn, mx = self.scalers[col]
        try:
            x = float(val)
        except Exception:
            x = mn
        return (x - mn) / (mx - mn)

    def build_sequence_tensor(self, seq_df: pd.DataFrame) -> Tuple[torch.Tensor, torch.Tensor]:
        """Build padded tensor [T, F] for a single patient sequence and return (tensor[1,T,F], length[1]).
        Handles sorting by time column if present, padding and masking implicitly done downstream.
        """
        df = seq_df.copy()
        if 'order_time_jittered_utc' in df.columns:
            try:
                df['order_time_jittered_utc'] = pd.to_datetime(df['order_time_jittered_utc'])
                df = df.sort_values('order_time_jittered_utc')
            except Exception:
                pass
        # Ensure required columns exist
        for c in self.CAT_FEATURES:
            if c not in df.columns:
                df[c] = ""
        for n in self.NUM_FEATURES:
            if n not in df.columns:
                df[n] = 0.0

        cats = np.stack([df[c].astype(str).map(lambda v: self._encode_cat(c, v)).values for c in self.CAT_FEATURES], axis=1)
        nums = np.stack([df[n].astype(float).map(lambda v: self._scale_num(n, v)).values for n in self.NUM_FEATURES], axis=1)
        x = np.concatenate([cats, nums], axis=1).astype(np.float32)

        seq_len = x.shape[0]
        # Add batch dimension
        x = torch.tensor(x, dtype=torch.float32).unsqueeze(0)
        lengths = torch.tensor([seq_len], dtype=torch.long)
        return x.to(self.device), lengths.to(self.device)

    def predict_lstm(self, seq_df: pd.DataFrame) -> float:
        """Return probability of resistance at the last timestep of the sequence."""
        with torch.no_grad():
            x, lengths = self.build_sequence_tensor(seq_df)
            logits = self.model(x, lengths)  # [1, T]
            probs = torch.sigmoid(logits)
            last_index = int(lengths[0].item()) - 1
            p = float(probs[0, last_index].item())
            return p
