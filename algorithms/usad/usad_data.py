"""
USAD Data Loader
Loads combined CSV, selects relevant pods/metrics, normalizes, and creates sliding windows.
Supports both pure normal training and combined (normal+anomaly) detection.
"""

import numpy as np
import pandas as pd
import torch
from sklearn.preprocessing import MinMaxScaler


class USADDataset(torch.utils.data.Dataset):
    """
    Each sample is a window of (window_size, n_features).
    Pivots data from long-format (pod, metric, timestamp) to wide-format.
    """

    def __init__(self, csv_path, window_size=12, downsample=5, pods=None,
                 normalize=True, label_col="label"):
        self.window_size = window_size
        self.downsample = downsample
        self.pods = pods or []
        self.normalize = normalize
        self.label_col = label_col

        df = pd.read_csv(csv_path)
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s")
        df = df.sort_values(["pod", "metric", "timestamp"])

        if self.pods:
            mask = df["pod"].apply(lambda p: any(c in p for c in self.pods))
            df = df[mask]

        if df.empty:
            raise ValueError(f"No data found for pods: {self.pods}")

        self._build_wide(df)
        self._create_windows()
        self._fit_normalize()

    def _build_wide(self, df):
        # Separate metric columns from label column
        if self.label_col in df.columns:
            df_vals = df.drop(columns=[self.label_col, "fault_service"], errors="ignore")
        else:
            df_vals = df

        pivot = df_vals.pivot_table(
            index="timestamp",
            columns=["pod", "metric"],
            values="value",
            aggfunc="mean"
        )
        pivot = pivot.ffill().bfill()
        pivot.columns = [f"{pod}|{m}" for pod, m in pivot.columns]
        pivot = pivot.sort_index()

        # Downsample
        if self.downsample > 1:
            pivot = pivot.iloc[::self.downsample].copy()
            pivot.index.name = "timestamp"

        self.data = pivot
        self.feature_names = list(pivot.columns)
        self.timestamps = pivot.index.tolist()

        # Build labels: per timestamp, 1 if any anomaly
        if self.label_col in df.columns:
            lbl_df = df.drop_duplicates("timestamp").set_index("timestamp")
            lbl_df = lbl_df.sort_index()
            if self.downsample > 1:
                lbl_df = lbl_df.iloc[::self.downsample].copy()
            lbl_df["label_int"] = (lbl_df[self.label_col] == "anomaly").astype(int)
            self._raw_labels = lbl_df["label_int"].reindex(self.timestamps).fillna(0).values
        else:
            self._raw_labels = np.zeros(len(self.timestamps), dtype=int)

    def _create_windows(self):
        k = self.window_size
        wide = self.data.values.astype(np.float32)
        n = len(wide)
        self.X_windows = []
        self.idx_map = []
        self.labels = []

        for i in range(k - 1, n):
            window = wide[i - k + 1 : i + 1]  # (k, n_features)
            self.X_windows.append(window)
            self.idx_map.append(i)
            # Window-level label: point-adjust (any anomaly in window = anomaly)
            window_labels = self._raw_labels[i - k + 1 : i + 1]
            self.labels.append(int(window_labels.max()))

        self.X_windows = np.stack(self.X_windows, axis=0)
        self.n_windows = len(self.X_windows)
        self.labels = np.array(self.labels, dtype=np.int64)

    def _fit_normalize(self):
        if self.normalize:
            shape = self.X_windows.shape
            flat = self.X_windows.reshape(-1, shape[-1])
            self.scaler = MinMaxScaler()
            flat_norm = self.scaler.fit_transform(flat)
            self.X_windows = flat_norm.reshape(shape).astype(np.float32)
        else:
            self.X_windows = self.X_windows.astype(np.float32)

        self.X = self.X_windows.reshape(self.n_windows, -1).astype(np.float32)

    def __len__(self):
        return self.n_windows

    def __getitem__(self, idx):
        x = torch.from_numpy(self.X[idx])
        y = torch.tensor(self.labels[idx], dtype=torch.long)
        return x, y

    def get_timestamp(self, idx):
        return self.timestamps[self.idx_map[idx]]

    def get_scores(self, model, alpha=0.5, beta=0.5, device="cpu"):
        """Compute anomaly scores for all windows using a trained USAD model."""
        model.eval()
        scores = []
        with torch.no_grad():
            for i in range(len(self)):
                x = torch.from_numpy(self.X[i]).unsqueeze(0).to(device)
                score, _, _ = model.anomaly_score(x, alpha=alpha, beta=beta)
                scores.append(score.item())
        return np.array(scores)

    def get_labels(self):
        return self.labels.copy()
