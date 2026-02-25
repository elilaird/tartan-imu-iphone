"""CoreML conversion pipeline for TartanIMU.

Exports the model in stateful mode so LSTM hidden state persists
across 1 Hz calls on iPhone without Python overhead.
"""

import argparse

import torch
import torch.nn as nn

try:
    import coremltools as ct
except ImportError:
    ct = None

from tartanimu.model.tartanimu import TartanIMU
from tartanimu.model.heads import HEAD_NAMES


HEAD_MAP = {name: i for i, name in enumerate(HEAD_NAMES)}


class TartanIMUStateful(nn.Module):
    """Wraps model for CoreML stateful export.

    Accepts a single 200-sample window + hidden state,
    returns velocity, covariance, updated hidden state.
    iPhone calls this at 1 Hz (after accumulating samples).
    """

    def __init__(self, model: TartanIMU, head: str = "car"):
        super().__init__()
        self.backbone = model.backbone
        self.lstm = model.temporal.lstm
        self.head_net = model.decoder.heads[head]

    def forward(
        self,
        imu_window: torch.Tensor,   # [1, 6, 200]
        h0: torch.Tensor,           # [2, 1, 512]
        c0: torch.Tensor,           # [2, 1, 512]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        feat = self.backbone(imu_window)          # [1, 512]
        feat_seq = feat.unsqueeze(1)              # [1, 1, 512]
        out, (hn, cn) = self.lstm(feat_seq, (h0, c0))
        pred = self.head_net(out[:, -1, :])       # [1, 6]
        vel = pred[:, :3]                         # [1, 3]
        lcov = pred[:, 3:]                        # [1, 3]
        return vel, lcov, hn, cn


def export_coreml(
    checkpoint_path: str,
    head: str = "car",
    output_path: str | None = None,
    merge_lora: bool = True,
) -> "ct.models.MLModel":
    """Export TartanIMU to CoreML mlpackage.

    Args:
        checkpoint_path: path to PyTorch checkpoint
        head: domain head to export ("car", "human", "quadruped", "drone")
        output_path: output .mlpackage path (default: tartanimu_{head}.mlpackage)
        merge_lora: merge LoRA weights before export
    Returns:
        CoreML model
    """
    if ct is None:
        raise ImportError("coremltools is required for CoreML export: pip install coremltools")

    model = TartanIMU.load_from_checkpoint(checkpoint_path)
    if merge_lora:
        model.merge_lora()
    model.eval()

    wrapper = TartanIMUStateful(model, head=head)
    wrapper.eval()

    # Trace with example inputs
    imu_ex = torch.zeros(1, 6, 200)
    h_ex = torch.zeros(2, 1, 512)
    c_ex = torch.zeros(2, 1, 512)

    with torch.no_grad():
        traced = torch.jit.trace(wrapper, (imu_ex, h_ex, c_ex))

    # Convert with FP16 for iPhone Neural Engine
    mlmodel = ct.convert(
        traced,
        inputs=[
            ct.TensorType(name="imu_window", shape=(1, 6, 200), dtype=float),
            ct.TensorType(name="hidden", shape=(2, 1, 512), dtype=float),
            ct.TensorType(name="cell", shape=(2, 1, 512), dtype=float),
        ],
        outputs=[
            ct.TensorType(name="velocity"),
            ct.TensorType(name="log_covariance"),
            ct.TensorType(name="hidden_out"),
            ct.TensorType(name="cell_out"),
        ],
        compute_precision=ct.precision.FLOAT16,
        compute_units=ct.ComputeUnit.ALL,
        minimum_deployment_target=ct.target.iOS17,
    )

    if output_path is None:
        output_path = f"tartanimu_{head}.mlpackage"

    mlmodel.save(output_path)
    print(f"Exported {output_path}")
    return mlmodel


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export TartanIMU to CoreML")
    parser.add_argument("checkpoint", help="Path to PyTorch checkpoint")
    parser.add_argument("--head", default="car", choices=HEAD_NAMES)
    parser.add_argument("--output", default=None, help="Output .mlpackage path")
    parser.add_argument("--no-merge-lora", action="store_true")
    args = parser.parse_args()

    export_coreml(args.checkpoint, args.head, args.output, not args.no_merge_lora)
