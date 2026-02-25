"""Full TartanIMU model assembly.

ResNet1D backbone -> LSTM temporal encoder -> Multi-head decoder.
Supports LoRA fine-tuning on the LSTM layers.
"""

import torch
import torch.nn as nn

from .backbone import ResNet1D
from .temporal import LSTMTemporalEncoder
from .heads import MultiHeadDecoder, HEAD_NAMES
from .lora import apply_lora_to_lstm, LoRALSTM


class TartanIMU(nn.Module):
    """TartanIMU: heterogeneous IMU velocity estimator.

    Input: sequence of 1-second IMU windows [B, seq_len, 6, 200]
    Output: 3D velocity + 3D log-covariance [B, 6]
    """

    def __init__(
        self,
        in_channels: int = 6,
        window_size: int = 200,
        hidden_size: int = 512,
        lstm_layers: int = 2,
        lstm_dropout: float = 0.1,
        lora_rank: int | None = None,
        lora_alpha: float = 16.0,
    ):
        super().__init__()
        self.window_size = window_size

        self.backbone = ResNet1D(in_channels=in_channels)
        self.temporal = LSTMTemporalEncoder(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=lstm_layers,
            dropout=lstm_dropout,
        )
        self.decoder = MultiHeadDecoder(input_dim=hidden_size)

        # Optionally wrap LSTM with LoRA
        self.lora_enabled = lora_rank is not None
        if self.lora_enabled:
            self.temporal.lstm = apply_lora_to_lstm(
                self.temporal.lstm, rank=lora_rank, alpha=lora_alpha
            )

    def forward(
        self,
        x: torch.Tensor,
        head: str = "car",
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        """
        Args:
            x: [B, seq_len, 6, 200] sequence of IMU windows
            head: domain head name
            hx: optional LSTM state (h0, c0)
        Returns:
            velocity: [B, 3]
            log_covariance: [B, 3]
            (hn, cn): updated LSTM state
        """
        B, S, C, T = x.shape

        # Encode each window through backbone
        x_flat = x.reshape(B * S, C, T)           # [B*S, 6, 200]
        features = self.backbone(x_flat)            # [B*S, 512]
        features = features.reshape(B, S, -1)       # [B, S, 512]

        # Temporal encoding
        hidden, (hn, cn) = self.temporal(features, hx)  # [B, 512]

        # Decode
        pred = self.decoder(hidden, head=head)  # [B, 6]
        velocity = pred[:, :3]
        log_cov = pred[:, 3:]

        return velocity, log_cov, (hn, cn)

    def nll_loss(
        self,
        velocity_pred: torch.Tensor,
        log_cov_pred: torch.Tensor,
        velocity_target: torch.Tensor,
    ) -> torch.Tensor:
        """Negative log-likelihood loss with learned diagonal covariance.

        L = 0.5 * sum( log_sigma_i + (v_i - v_hat_i)^2 / exp(log_sigma_i) )
        """
        precision = torch.exp(-log_cov_pred)  # 1 / sigma^2
        mse_weighted = precision * (velocity_pred - velocity_target) ** 2
        return 0.5 * (log_cov_pred + mse_weighted).mean()

    def merge_lora(self):
        """Merge LoRA weights into LSTM for export."""
        if self.lora_enabled and isinstance(self.temporal.lstm, LoRALSTM):
            self.temporal.lstm.merge_lora()
            self.temporal.lstm = self.temporal.lstm.lstm
            self.lora_enabled = False

    def trainable_params(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def total_params(self) -> int:
        return sum(p.numel() for p in self.parameters())

    @classmethod
    def load_from_checkpoint(cls, path: str, **kwargs) -> "TartanIMU":
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        if "model_config" in checkpoint:
            model = cls(**checkpoint["model_config"])
        else:
            model = cls(**kwargs)
        model.load_state_dict(checkpoint["model_state_dict"])
        return model

    def save_checkpoint(self, path: str, config: dict | None = None):
        torch.save({
            "model_state_dict": self.state_dict(),
            "model_config": config or {},
        }, path)
