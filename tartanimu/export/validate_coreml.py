"""Validate CoreML model parity against PyTorch.

Runs the same input through both models and asserts outputs
match within tolerance before shipping to device.
"""

import argparse
import sys

import numpy as np
import torch

try:
    import coremltools as ct
except ImportError:
    ct = None

from tartanimu.model.tartanimu import TartanIMU
from tartanimu.export.to_coreml import TartanIMUStateful


def validate_parity(
    checkpoint_path: str,
    mlpackage_path: str,
    head: str = "car",
    tolerance: float = 1e-3,
    n_steps: int = 5,
) -> bool:
    """Run parity check between PyTorch and CoreML models.

    Args:
        checkpoint_path: PyTorch checkpoint path
        mlpackage_path: CoreML .mlpackage path
        head: domain head
        tolerance: max absolute difference allowed
        n_steps: number of sequential inference steps to test
    Returns:
        True if all outputs match within tolerance
    """
    if ct is None:
        raise ImportError("coremltools is required: pip install coremltools")

    # Load PyTorch model
    pt_model = TartanIMU.load_from_checkpoint(checkpoint_path)
    pt_model.merge_lora()
    pt_model.eval()
    wrapper = TartanIMUStateful(pt_model, head=head)
    wrapper.eval()

    # Load CoreML model
    ml_model = ct.models.MLModel(mlpackage_path)

    # Initialize states
    h_pt = torch.zeros(2, 1, 512)
    c_pt = torch.zeros(2, 1, 512)

    all_passed = True

    for step in range(n_steps):
        # Random IMU input
        imu = torch.randn(1, 6, 200)

        # PyTorch forward
        with torch.no_grad():
            vel_pt, lcov_pt, h_pt, c_pt = wrapper(imu, h_pt, c_pt)

        # CoreML forward
        ml_input = {
            "imu_window": imu.numpy(),
            "hidden": h_pt.numpy() if step == 0 else ml_out["hidden_out"],
            "cell": c_pt.numpy() if step == 0 else ml_out["cell_out"],
        }
        # On first step, use zero-init for CoreML too
        if step == 0:
            ml_input["hidden"] = np.zeros((2, 1, 512), dtype=np.float32)
            ml_input["cell"] = np.zeros((2, 1, 512), dtype=np.float32)

        ml_out = ml_model.predict(ml_input)

        # Compare outputs
        vel_diff = np.abs(vel_pt.numpy() - ml_out["velocity"]).max()
        lcov_diff = np.abs(lcov_pt.numpy() - ml_out["log_covariance"]).max()
        h_diff = np.abs(h_pt.numpy() - ml_out["hidden_out"]).max()
        c_diff = np.abs(c_pt.numpy() - ml_out["cell_out"]).max()

        passed = all(d < tolerance for d in [vel_diff, lcov_diff, h_diff, c_diff])
        status = "PASS" if passed else "FAIL"
        all_passed = all_passed and passed

        print(f"Step {step}: {status}  "
              f"vel={vel_diff:.6f}  lcov={lcov_diff:.6f}  "
              f"h={h_diff:.6f}  c={c_diff:.6f}")

        # Use CoreML outputs as next state for CoreML path
        # Use PyTorch outputs as next state for PyTorch path

    print(f"\nOverall: {'PASS' if all_passed else 'FAIL'} (tolerance={tolerance})")
    return all_passed


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate CoreML vs PyTorch parity")
    parser.add_argument("checkpoint", help="PyTorch checkpoint path")
    parser.add_argument("mlpackage", help="CoreML .mlpackage path")
    parser.add_argument("--head", default="car")
    parser.add_argument("--tolerance", type=float, default=1e-3)
    parser.add_argument("--steps", type=int, default=5)
    args = parser.parse_args()

    ok = validate_parity(args.checkpoint, args.mlpackage, args.head, args.tolerance, args.steps)
    sys.exit(0 if ok else 1)
