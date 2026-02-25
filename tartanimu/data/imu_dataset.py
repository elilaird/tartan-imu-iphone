"""IMU Dataset with window slicing and coordinate normalization.

Loads IMU recordings and slices them into sequences of 1-second windows
(200 samples each) with 1 Hz velocity labels.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path


class IMUDataset(Dataset):
    """Dataset that produces (imu_sequence, velocity_label) pairs.

    Each item:
        imu_sequence: [seq_len, 6, window_size] — sequence of IMU windows
        velocity_label: [3] — body-frame velocity at the end of the sequence

    Expected data format per file:
        - NPZ with keys "imu" [N, 6] at 200 Hz and "velocity" [M, 3] at 1 Hz
        - N = M * window_size (each velocity label covers one window)
    """

    def __init__(
        self,
        data_dir: str,
        seq_len: int = 10,
        window_size: int = 200,
        normalize: bool = True,
        head: str = "car",
    ):
        self.seq_len = seq_len
        self.window_size = window_size
        self.normalize = normalize
        self.head = head

        self.data_dir = Path(data_dir)
        self.files = sorted(self.data_dir.glob("*.npz"))
        if not self.files:
            raise FileNotFoundError(f"No .npz files found in {data_dir}")

        # Build index: (file_idx, start_window_idx)
        self.samples = []
        self.stats = {"mean": None, "std": None}
        self._build_index()

    def _build_index(self):
        all_imu = []
        for file_idx, f in enumerate(self.files):
            data = np.load(f)
            imu = data["imu"]       # [N, 6]
            vel = data["velocity"]   # [M, 3]
            n_windows = len(vel)

            for start in range(n_windows - self.seq_len + 1):
                self.samples.append((file_idx, start))

            if self.normalize:
                all_imu.append(imu)

        if self.normalize and all_imu:
            combined = np.concatenate(all_imu, axis=0)
            self.stats["mean"] = torch.tensor(combined.mean(axis=0), dtype=torch.float32)
            self.stats["std"] = torch.tensor(combined.std(axis=0).clip(min=1e-6), dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        file_idx, start_win = self.samples[idx]
        data = np.load(self.files[file_idx])

        imu_raw = data["imu"]       # [N, 6]
        vel_raw = data["velocity"]   # [M, 3]

        # Extract sequence of windows
        windows = []
        for w in range(start_win, start_win + self.seq_len):
            t_start = w * self.window_size
            t_end = t_start + self.window_size
            window = torch.tensor(imu_raw[t_start:t_end], dtype=torch.float32)  # [200, 6]
            windows.append(window.T)  # [6, 200]

        imu_seq = torch.stack(windows)  # [seq_len, 6, 200]

        # Normalize
        if self.normalize and self.stats["mean"] is not None:
            mean = self.stats["mean"].unsqueeze(0)  # [1, 6]
            std = self.stats["std"].unsqueeze(0)
            for i in range(self.seq_len):
                imu_seq[i] = (imu_seq[i] - mean.T) / std.T  # broadcast over time dim

        # Velocity label for the last window in sequence
        vel = torch.tensor(vel_raw[start_win + self.seq_len - 1], dtype=torch.float32)

        return imu_seq, vel
