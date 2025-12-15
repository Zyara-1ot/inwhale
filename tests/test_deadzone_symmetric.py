import torch

from inwhale.core.uniform import DeadZoneSymmetricQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def test_bits_boundaries():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(bits=4, observer=obs, rounding=rnd, threshold_ratio=0.5)
    assert q.qmin == -(1 << (4 - 1))
    assert q.qmax == (1 << (4 - 1)) - 1


def test_threshold_computation():
    x = torch.tensor([0.001, -0.003, 0.006, -0.009, 0.02, -0.04])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    threshold_ratio = 0.5
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=threshold_ratio
    )

    q.quantize(x)

    # threshold = threshold_ratio * scale
    expected_threshold = threshold_ratio * q.scale
    assert torch.allclose(q.threshold, expected_threshold)


def test_small_values_quantized_to_zero():
    # Create very small values compared to the max
    x = torch.tensor([0.1, -0.1, 0.05, -0.08, 0.00001, -0.00002])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)

    # Very small values relative to max should be quantized to 0
    # Only the last two values should be zero (below threshold)
    assert qx[-2] == 0
    assert qx[-1] == 0


def test_large_values_outside_deadzone():
    x = torch.tensor([0.1, -0.15, 0.2, -0.25, 0.3, -0.4])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)

    # Values well outside threshold should not be zero
    non_zero_mask = (x.abs() >= 2 * q.threshold)
    assert torch.all(qx[non_zero_mask] != 0)


def test_quantized_values_within_range():
    x = torch.linspace(-1.0, 1.0, steps=51)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)

    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)


def test_dequantize_round_trip():
    x = torch.tensor([0.001, -0.003, 0.1, -0.15, 0.4, -0.8])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    # Small values should dequantize to near zero (or exact zero after threshold)
    small_mask = x.abs() < q.threshold
    assert torch.allclose(dx[small_mask], torch.zeros_like(dx[small_mask]), atol=1e-6)

    # Large values should reconstruct with bounded error
    large_mask = ~small_mask
    if large_mask.any():
        # Error should be bounded by threshold + one scale step
        error = (x[large_mask] - dx[large_mask]).abs()
        assert torch.all(error <= q.threshold + q.scale)


def test_zero_range_edge_case():
    x = torch.full((5,), 2.0)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)
    # Threshold may be scalar, convert for dequant
    if not isinstance(q.threshold, torch.Tensor):
        q.threshold = torch.tensor(q.threshold)
    dx = q.dequantize(qx)

    # When all values are identical, scale = 1.0 and threshold = 0.5
    assert float(q.scale) == 1.0
    assert float(q.threshold) == 0.5

    # Non-zero values should still be quantized
    assert torch.all(qx != 0)


def test_threshold_ratio_effect():
    x = torch.tensor([0.01, -0.02, 0.1, -0.2, 0.5, -1.0])
    obs1 = MinMaxObserver()
    obs2 = MinMaxObserver()
    rnd = NearestRounding()

    q_small_threshold = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs1, rounding=rnd, threshold_ratio=0.1
    )
    q_large_threshold = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs2, rounding=rnd, threshold_ratio=0.9
    )

    qx_small = q_small_threshold.quantize(x)
    qx_large = q_large_threshold.quantize(x)

    # Larger threshold ratio should result in more zeros
    zeros_small = torch.sum(qx_small == 0)
    zeros_large = torch.sum(qx_large == 0)
    assert zeros_large >= zeros_small


def test_sparsity_with_deadzone():
    # Create values with many small values and some large values
    x = torch.cat(
        [torch.randn(80) * 0.001, torch.randn(20) * 0.5]
    )  # 80 very small, 20 larger
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = DeadZoneSymmetricQuantizer(
        bits=8, observer=obs, rounding=rnd, threshold_ratio=0.5
    )

    qx = q.quantize(x)

    # Should have sparsity due to deadzone (many small values become zero)
    sparsity = torch.sum(qx == 0).float() / qx.numel()
    assert sparsity > 0.2  # At least 20% should be zero
