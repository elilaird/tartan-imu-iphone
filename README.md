# TartanIMU — Usage Guide

## Prerequisites

- Python 3.11+ with conda (`lio-v` environment)
- PyTorch
- coremltools (`pip install coremltools`)
- Xcode 15+ (for iOS deployment)
- iPhone running iOS 17+

## 1. Training the Model

### Prepare data

Place your IMU recordings as `.npz` files in a data directory. Each file should contain:
- `imu`: `[N, 6]` array at 200 Hz (columns: acc_x, acc_y, acc_z, gyro_x, gyro_y, gyro_z)
- `velocity`: `[M, 3]` array at 1 Hz (body-frame velocity labels, where M = N / 200)

### Train

```python
import torch
from torch.utils.data import DataLoader
from tartanimu.model import TartanIMU
from tartanimu.data import IMUDataset, RotationAugmentation

# Load data
dataset = IMUDataset("path/to/data/", seq_len=10, window_size=200)
loader = DataLoader(dataset, batch_size=32, shuffle=True)

# Create model (with optional LoRA fine-tuning)
model = TartanIMU(lora_rank=8)
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=1e-3)

augment = RotationAugmentation()

# Training loop
for epoch in range(100):
    for imu_seq, vel_target in loader:
        imu_seq, vel_target = augment(imu_seq, vel_target)
        vel_pred, lcov_pred, _ = model(imu_seq, head="car")
        loss = model.nll_loss(vel_pred, lcov_pred, vel_target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# Save checkpoint
model.save_checkpoint("checkpoints/tartanimu_car.pt", config={
    "lora_rank": 8,
    "lora_alpha": 16.0,
})
```

## 2. Running Inference (PyTorch)

```python
import torch
from tartanimu.model import TartanIMU

model = TartanIMU.load_from_checkpoint("checkpoints/tartanimu_car.pt", lora_rank=8)
model.eval()

# Single sequence: 10 windows of 200 samples each
imu_input = torch.randn(1, 10, 6, 200)

with torch.no_grad():
    velocity, log_covariance, (hn, cn) = model(imu_input, head="car")

print(f"Velocity: {velocity}")          # [1, 3] m/s
print(f"Uncertainty: {log_covariance.exp().sqrt()}")  # [1, 3] std dev
```

### Streaming mode (window-by-window with state)

```python
h, c = None, None
for window in imu_windows:  # each [1, 1, 6, 200]
    with torch.no_grad():
        vel, lcov, (h, c) = model(window, head="car", hx=(h, c) if h is not None else None)
    print(f"Velocity: {vel[0].tolist()}")
```

## 3. Exporting to CoreML

```bash
conda activate lio-v
python -m tartanimu.export.to_coreml checkpoints/tartanimu_car.pt --head car
```

This produces `tartanimu_car.mlpackage`.

Options:
- `--head car|human|quadruped|drone` — which domain head to export
- `--output path.mlpackage` — custom output path
- `--no-merge-lora` — keep LoRA weights separate (not recommended for deployment)

### Validate export parity

```bash
python -m tartanimu.export.validate_coreml checkpoints/tartanimu_car.pt tartanimu_car.mlpackage
```

All steps should print `PASS` with differences < 1e-3.

### Optional: INT8 quantization

```bash
python -m tartanimu.export.quantize tartanimu_car.mlpackage --nbits 8
```

Produces `tartanimu_car_q8.mlpackage`.

## 4. Running Tests

```bash
conda activate lio-v
python -m pytest tests/ -v
```

## 5. Building the iOS App

### Setup

1. Open Xcode and create a new iOS App project:
   - Product Name: **TartanIMUBench**
   - Interface: **SwiftUI**
   - Language: **Swift**
   - Minimum Deployment: **iOS 17.0**

2. Replace the generated files with the Swift sources from `ios/TartanIMUBench/`:
   - `TartanIMUBenchApp.swift` — app entry point
   - `IMUCapture.swift` — IMU data capture
   - `LiveInference.swift` — CoreML inference runner
   - `TrajectoryView.swift` — UI dashboard
   - `BenchmarkRunner.swift` — throughput benchmarks

3. Drag `tartanimu_car.mlpackage` into the Xcode project navigator (check "Copy items if needed").

4. In the project's **Signing & Capabilities**:
   - Select your development team
   - Enable **Motion Usage** — add `NSMotionUsageDescription` to `Info.plist`:
     ```
     Key: Privacy - Motion Usage Description
     Value: TartanIMU needs motion data for real-time velocity estimation.
     ```

### Build & Run

1. Connect your iPhone via USB
2. Select your device as the build target
3. Press **Cmd+R** to build and run

> The app will not produce meaningful output in the Simulator since it requires real IMU hardware.

### Using the App

**Live Tab:**
- Tap **Start** to begin capturing IMU data and running inference
- The trajectory canvas shows dead-reckoned position (green = start, red = current)
- Metrics bar shows inference latency, FPS, IMU sample rate, and current velocity
- Tap **Export** to save the trajectory as CSV

**Benchmark Tab:**
- Tap **Run Benchmark** to test inference across compute unit configurations:
  - ANE + CPU (FP16)
  - GPU + CPU (FP16)
  - CPU only
  - All units (ANE + GPU + CPU)
- Results show mean latency, P99 latency, and throughput (FPS) for each configuration
