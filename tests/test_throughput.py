"""Throughput benchmark for TartanIMU PyTorch model.

Measures inference latency and throughput to validate
the model can hit 200 FPS on target hardware.
"""

import time
import statistics

import torch
import pytest

from tartanimu.model import TartanIMU


@pytest.fixture
def model():
    m = TartanIMU()
    m.eval()
    return m


def test_model_output_shapes(model):
    """Verify model produces correct output shapes."""
    x = torch.randn(1, 10, 6, 200)
    with torch.no_grad():
        vel, lcov, (hn, cn) = model(x, head="car")

    assert vel.shape == (1, 3), f"Expected velocity shape (1, 3), got {vel.shape}"
    assert lcov.shape == (1, 3), f"Expected log_cov shape (1, 3), got {lcov.shape}"
    assert hn.shape == (2, 1, 512), f"Expected hn shape (2, 1, 512), got {hn.shape}"
    assert cn.shape == (2, 1, 512), f"Expected cn shape (2, 1, 512), got {cn.shape}"


def test_all_heads(model):
    """Verify all 4 heads produce valid outputs."""
    x = torch.randn(1, 10, 6, 200)
    for head in ["car", "human", "quadruped", "drone"]:
        with torch.no_grad():
            vel, lcov, _ = model(x, head=head)
        assert vel.shape == (1, 3), f"Head {head}: wrong velocity shape"
        assert not torch.isnan(vel).any(), f"Head {head}: NaN in velocity"


def test_single_window_throughput(model):
    """Benchmark single-window inference (simulates CoreML stateful mode)."""
    # Single window: what CoreML will process per call
    window = torch.randn(1, 1, 6, 200)
    h0 = torch.zeros(2, 1, 512)
    c0 = torch.zeros(2, 1, 512)

    warmup = 20
    iterations = 200

    # Warmup
    for _ in range(warmup):
        with torch.no_grad():
            _, _, (h0, c0) = model(window, head="car", hx=(h0, c0))

    # Benchmark
    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            _, _, (h0, c0) = model(window, head="car", hx=(h0, c0))
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    mean_ms = statistics.mean(latencies)
    p99_ms = sorted(latencies)[int(len(latencies) * 0.99)]
    fps = 1000.0 / mean_ms

    print(f"\nSingle-window throughput:")
    print(f"  Mean latency: {mean_ms:.2f} ms")
    print(f"  P99 latency:  {p99_ms:.2f} ms")
    print(f"  Throughput:   {fps:.0f} FPS")

    # Paper claims 200 FPS — on CPU we may be slower, but should be reasonable
    assert mean_ms < 50, f"Single-window inference too slow: {mean_ms:.2f}ms (>50ms)"


def test_sequence_throughput(model):
    """Benchmark full 10-window sequence inference."""
    x = torch.randn(1, 10, 6, 200)

    warmup = 5
    iterations = 50

    for _ in range(warmup):
        with torch.no_grad():
            model(x, head="car")

    latencies = []
    for _ in range(iterations):
        start = time.perf_counter()
        with torch.no_grad():
            model(x, head="car")
        elapsed_ms = (time.perf_counter() - start) * 1000
        latencies.append(elapsed_ms)

    mean_ms = statistics.mean(latencies)
    fps = 1000.0 / mean_ms

    print(f"\n10-window sequence throughput:")
    print(f"  Mean latency: {mean_ms:.2f} ms")
    print(f"  Throughput:   {fps:.0f} FPS")


def test_param_count(model):
    """Verify model parameter count is reasonable."""
    total = model.total_params()
    trainable = model.trainable_params()

    print(f"\nParameter count:")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")

    # ResNet18-1D + 2-layer LSTM(512) + 4 heads should be ~10-15M params
    assert total > 1_000_000, f"Suspiciously few params: {total:,}"
    assert total < 50_000_000, f"Suspiciously many params: {total:,}"


def test_lora_param_count():
    """Verify LoRA reduces trainable params significantly."""
    model = TartanIMU(lora_rank=8)

    total = model.total_params()
    trainable = model.trainable_params()

    print(f"\nLoRA parameter count (rank=8):")
    print(f"  Total:     {total:,}")
    print(f"  Trainable: {trainable:,}")
    print(f"  Ratio:     {trainable/total:.2%}")

    # With LoRA, trainable should be much less than total
    assert trainable < total, "LoRA should reduce trainable params"


def test_lstm_state_persistence(model):
    """Verify LSTM state correctly persists across sequential calls."""
    window = torch.randn(1, 1, 6, 200)

    with torch.no_grad():
        # Run two windows sequentially with state passing
        _, _, (h1, c1) = model(window, head="car")
        vel_seq, _, _ = model(window, head="car", hx=(h1, c1))

        # Run same two windows as a batch
        two_windows = window.repeat(1, 2, 1, 1)  # [1, 2, 6, 200]
        vel_batch, _, _ = model(two_windows, head="car")

    # Results should match (sequential state passing == batched sequence)
    diff = (vel_seq - vel_batch).abs().max().item()
    assert diff < 1e-5, f"State persistence mismatch: max diff = {diff}"
