"""Multi-head decoder for TartanIMU.

Each head outputs 6 values: 3D velocity + 3D log-diagonal covariance.
Four heads: car, human, quadruped, drone.
"""

import torch
import torch.nn as nn


HEAD_NAMES = ["car", "human", "quadruped", "drone"]


class VelocityHead(nn.Module):
    """Single velocity prediction head: [B, 512] -> [B, 6]."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(inplace=True),
            nn.Linear(hidden_dim, 6),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class MultiHeadDecoder(nn.Module):
    """Four-head decoder producing per-domain velocity predictions."""

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.heads = nn.ModuleDict({
            name: VelocityHead(input_dim, hidden_dim) for name in HEAD_NAMES
        })

    def forward(self, x: torch.Tensor, head: str = "car") -> torch.Tensor:
        """
        Args:
            x: [B, 512] LSTM output
            head: one of "car", "human", "quadruped", "drone"
        Returns:
            [B, 6] = [v_x, v_y, v_z, log_sigma_x, log_sigma_y, log_sigma_z]
        """
        return self.heads[head](x)

    def forward_all(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        return {name: head(x) for name, head in self.heads.items()}
