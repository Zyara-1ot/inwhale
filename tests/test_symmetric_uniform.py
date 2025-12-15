import torch

from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


def test_bits_boundaries():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q4 = SymmetricUniformQuantizer(bits=4, observer=obs, rounding=rnd)
    assert q4.qmin == -(1 << (4 - 1))
    assert q4.qmax == (1 << (4 - 1)) - 1


def test_quantize_dequantize_basic_close():
    x = torch.tensor([1.2, -0.7, 0.4, 2.3, -1.9])
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = SymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert q.scale is not None
    assert torch.allclose(dx, x, atol=float(q.scale))


def test_quantized_values_within_range():
    x = torch.linspace(-10, 10, steps=101)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = SymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd)

    qx = q.quantize(x)
    assert torch.all(qx >= q.qmin)
    assert torch.all(qx <= q.qmax)


def test_zero_input_scale_clamped_and_round_trip():
    x = torch.zeros(10)
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = SymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd)

    qx = q.quantize(x)
    dx = q.dequantize(qx)

    assert q.scale == torch.tensor(1e-8)
    assert torch.all(qx == 0)
    assert torch.all(dx == 0)


def test_observer_accumulates_across_calls():
    obs = MinMaxObserver()
    rnd = NearestRounding()
    q = SymmetricUniformQuantizer(bits=8, observer=obs, rounding=rnd)

    x1 = torch.tensor([-1.0, 1.0])
    q.quantize(x1)
    scale1 = q.scale

    x2 = torch.tensor([-2.0, 2.0])
    q.quantize(x2)
    scale2 = q.scale

    assert scale2 >= scale1
