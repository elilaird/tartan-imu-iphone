"""LSTM temporal encoder for TartanIMU.

Processes a sequence of per-window features [B, seq_len, 512]
and returns the final hidden representation [B, 512].
"""

import torch
import torch.nn as nn


class LSTMTemporalEncoder(nn.Module):
    """2-layer LSTM that encodes a sequence of window features."""

    def __init__(
        self,
        input_size: int = 512,
        hidden_size: int = 512,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers

        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, seq_len, 512] sequence of window features
            hx: optional (h0, c0) each [num_layers, B, hidden_size]
        Returns:
            output: [B, hidden_size] final time-step hidden state
            (hn, cn): updated LSTM state
        """
        out, (hn, cn) = self.lstm(x, hx)
        return out[:, -1, :], (hn, cn)

    def init_hidden(self, batch_size: int, device: torch.device) -> tuple[torch.Tensor, torch.Tensor]:
        h0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        c0 = torch.zeros(self.num_layers, batch_size, self.hidden_size, device=device)
        return h0, c0
