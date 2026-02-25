"""LoRA (Low-Rank Adaptation) for LSTM weight matrices.

Applied to weight_ih and weight_hh of each LSTM layer.
W_new = W_frozen + (B @ A) * (alpha / r)
Only A and B are trained; ~1.1M params for rank-8 on 2-layer LSTM(512,512).
"""

import torch
import torch.nn as nn
import math


class LoRALinear(nn.Module):
    """LoRA adapter for a single weight matrix."""

    def __init__(self, in_features: int, out_features: int, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.rank = rank
        self.scaling = alpha / rank

        self.lora_A = nn.Parameter(torch.empty(rank, in_features))
        self.lora_B = nn.Parameter(torch.zeros(out_features, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))

    def forward(self) -> torch.Tensor:
        """Returns the LoRA delta: B @ A * scaling."""
        return (self.lora_B @ self.lora_A) * self.scaling


class LoRALSTM(nn.Module):
    """Wraps an nn.LSTM with LoRA adapters on weight_ih and weight_hh."""

    def __init__(self, lstm: nn.LSTM, rank: int = 8, alpha: float = 16.0):
        super().__init__()
        self.lstm = lstm

        # Freeze original LSTM parameters
        for param in self.lstm.parameters():
            param.requires_grad = False

        self.lora_adapters = nn.ModuleDict()
        for layer_idx in range(lstm.num_layers):
            weight_ih = getattr(lstm, f"weight_ih_l{layer_idx}")
            weight_hh = getattr(lstm, f"weight_hh_l{layer_idx}")

            self.lora_adapters[f"ih_l{layer_idx}"] = LoRALinear(
                weight_ih.shape[1], weight_ih.shape[0], rank, alpha
            )
            self.lora_adapters[f"hh_l{layer_idx}"] = LoRALinear(
                weight_hh.shape[1], weight_hh.shape[0], rank, alpha
            )

    def _apply_lora(self):
        """Add LoRA deltas to frozen weights (in-place for forward pass)."""
        for layer_idx in range(self.lstm.num_layers):
            weight_ih = getattr(self.lstm, f"weight_ih_l{layer_idx}")
            weight_hh = getattr(self.lstm, f"weight_hh_l{layer_idx}")

            ih_delta = self.lora_adapters[f"ih_l{layer_idx}"]()
            hh_delta = self.lora_adapters[f"hh_l{layer_idx}"]()

            weight_ih.data.add_(ih_delta.data)
            weight_hh.data.add_(hh_delta.data)

    def _remove_lora(self):
        """Remove LoRA deltas after forward pass."""
        for layer_idx in range(self.lstm.num_layers):
            weight_ih = getattr(self.lstm, f"weight_ih_l{layer_idx}")
            weight_hh = getattr(self.lstm, f"weight_hh_l{layer_idx}")

            ih_delta = self.lora_adapters[f"ih_l{layer_idx}"]()
            hh_delta = self.lora_adapters[f"hh_l{layer_idx}"]()

            weight_ih.data.sub_(ih_delta.data)
            weight_hh.data.sub_(hh_delta.data)

    def forward(
        self,
        x: torch.Tensor,
        hx: tuple[torch.Tensor, torch.Tensor] | None = None,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor]]:
        self._apply_lora()
        try:
            return self.lstm(x, hx)
        finally:
            self._remove_lora()

    def merge_lora(self):
        """Permanently merge LoRA weights into the LSTM (for export)."""
        for layer_idx in range(self.lstm.num_layers):
            weight_ih = getattr(self.lstm, f"weight_ih_l{layer_idx}")
            weight_hh = getattr(self.lstm, f"weight_hh_l{layer_idx}")

            ih_delta = self.lora_adapters[f"ih_l{layer_idx}"]()
            hh_delta = self.lora_adapters[f"hh_l{layer_idx}"]()

            weight_ih.data.add_(ih_delta.data)
            weight_hh.data.add_(hh_delta.data)

        # Remove adapters after merge
        self.lora_adapters = nn.ModuleDict()
        for param in self.lstm.parameters():
            param.requires_grad = True


def apply_lora_to_lstm(lstm: nn.LSTM, rank: int = 8, alpha: float = 16.0) -> LoRALSTM:
    """Convenience function to wrap an LSTM with LoRA adapters."""
    return LoRALSTM(lstm, rank=rank, alpha=alpha)
