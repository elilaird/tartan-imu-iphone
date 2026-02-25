"""Create an untrained TartanIMU checkpoint and export it to CoreML."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from tartanimu.model import TartanIMU
from tartanimu.export.to_coreml import export_coreml

CHECKPOINT_DIR = Path(__file__).resolve().parent.parent / "checkpoints"
EXPORT_DIR = Path(__file__).resolve().parent.parent / "export"


def main():
    CHECKPOINT_DIR.mkdir(exist_ok=True)
    EXPORT_DIR.mkdir(exist_ok=True)

    config = {
        "in_channels": 6,
        "window_size": 200,
        "hidden_size": 512,
        "lstm_layers": 2,
        "lstm_dropout": 0.1,
    }

    print("Creating untrained model...")
    model = TartanIMU(**config)
    print(f"  Total params:     {model.total_params():,}")
    print(f"  Trainable params: {model.trainable_params():,}")

    # Sanity check forward pass
    x = torch.randn(1, 1, 6, 200)
    with torch.no_grad():
        vel, lcov, _ = model(x, head="car")
    print(f"  Test forward pass: vel={vel[0].tolist()}, lcov={lcov[0].tolist()}")

    checkpoint_path = CHECKPOINT_DIR / "tartanimu_untrained.pt"
    model.save_checkpoint(str(checkpoint_path), config=config)
    print(f"  Saved checkpoint: {checkpoint_path}")

    print("\nExporting to CoreML...")
    for head in ["car", "human", "quadruped", "drone"]:
        output_path = EXPORT_DIR / f"tartanimu_{head}.mlpackage"
        export_coreml(str(checkpoint_path), head=head, output_path=str(output_path))

    print("\nDone.")


if __name__ == "__main__":
    main()
