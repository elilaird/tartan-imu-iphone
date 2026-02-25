"""CoreML parity tests.

Verifies that the CoreML export wrapper produces identical outputs
to the full PyTorch model. These tests run without coremltools
(pure PyTorch verification of the stateful wrapper).
"""

import torch
import pytest

from tartanimu.model import TartanIMU
from tartanimu.export.to_coreml import TartanIMUStateful


@pytest.fixture
def model():
    m = TartanIMU()
    m.eval()
    return m


@pytest.fixture
def stateful_wrapper(model):
    return TartanIMUStateful(model, head="car")


def test_stateful_wrapper_output_shapes(stateful_wrapper):
    """Verify stateful wrapper produces correct output shapes."""
    imu = torch.randn(1, 6, 200)
    h0 = torch.zeros(2, 1, 512)
    c0 = torch.zeros(2, 1, 512)

    with torch.no_grad():
        vel, lcov, hn, cn = stateful_wrapper(imu, h0, c0)

    assert vel.shape == (1, 3)
    assert lcov.shape == (1, 3)
    assert hn.shape == (2, 1, 512)
    assert cn.shape == (2, 1, 512)


def test_stateful_vs_full_model(model, stateful_wrapper):
    """Verify stateful wrapper matches full model output."""
    imu_window = torch.randn(1, 6, 200)
    h0 = torch.zeros(2, 1, 512)
    c0 = torch.zeros(2, 1, 512)

    with torch.no_grad():
        # Full model: single window as sequence
        imu_seq = imu_window.unsqueeze(1)  # [1, 1, 6, 200]
        vel_full, lcov_full, (hn_full, cn_full) = model(imu_seq, head="car", hx=(h0, c0))

        # Stateful wrapper
        vel_wrap, lcov_wrap, hn_wrap, cn_wrap = stateful_wrapper(imu_window, h0, c0)

    torch.testing.assert_close(vel_full, vel_wrap, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lcov_full, lcov_wrap, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(hn_full, hn_wrap, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(cn_full, cn_wrap, atol=1e-6, rtol=1e-5)


def test_sequential_state_passing(model, stateful_wrapper):
    """Verify sequential calls through wrapper match batched sequence through model."""
    windows = [torch.randn(1, 6, 200) for _ in range(5)]
    h = torch.zeros(2, 1, 512)
    c = torch.zeros(2, 1, 512)

    # Sequential through wrapper
    with torch.no_grad():
        for w in windows:
            vel_w, lcov_w, h, c = stateful_wrapper(w, h, c)

    # Batched through full model
    with torch.no_grad():
        imu_seq = torch.stack([w.squeeze(0) for w in windows], dim=0).unsqueeze(0)  # [1, 5, 6, 200]
        vel_f, lcov_f, _ = model(imu_seq, head="car")

    torch.testing.assert_close(vel_w, vel_f, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lcov_w, lcov_f, atol=1e-5, rtol=1e-4)


def test_jit_trace(stateful_wrapper):
    """Verify the model can be JIT traced (prerequisite for CoreML export)."""
    imu = torch.randn(1, 6, 200)
    h0 = torch.zeros(2, 1, 512)
    c0 = torch.zeros(2, 1, 512)

    with torch.no_grad():
        traced = torch.jit.trace(stateful_wrapper, (imu, h0, c0))

        # Run traced model
        vel_t, lcov_t, hn_t, cn_t = traced(imu, h0, c0)

        # Run original
        vel_o, lcov_o, hn_o, cn_o = stateful_wrapper(imu, h0, c0)

    torch.testing.assert_close(vel_t, vel_o, atol=1e-6, rtol=1e-5)
    torch.testing.assert_close(lcov_t, lcov_o, atol=1e-6, rtol=1e-5)


def test_nll_loss(model):
    """Verify NLL loss computes without errors."""
    vel_pred = torch.randn(4, 3)
    lcov_pred = torch.randn(4, 3)
    vel_target = torch.randn(4, 3)

    loss = model.nll_loss(vel_pred, lcov_pred, vel_target)

    assert loss.shape == ()
    assert not torch.isnan(loss)
    assert loss.item() > 0  # loss should be positive for random inputs


def test_merge_lora():
    """Verify LoRA merge produces same outputs."""
    model = TartanIMU(lora_rank=8)
    model.eval()

    x = torch.randn(1, 10, 6, 200)

    with torch.no_grad():
        vel_before, lcov_before, _ = model(x, head="car")

    model.merge_lora()

    with torch.no_grad():
        vel_after, lcov_after, _ = model(x, head="car")

    torch.testing.assert_close(vel_before, vel_after, atol=1e-5, rtol=1e-4)
    torch.testing.assert_close(lcov_before, lcov_after, atol=1e-5, rtol=1e-4)
    assert not model.lora_enabled
