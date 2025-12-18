import torch

from inwhale.rounding.bankers import BankersRounding


def test_half_to_even_positive_and_negative():
    br = BankersRounding()
    x = torch.tensor([0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, -3.5], dtype=torch.float32)
    out = br.round(x)
    expected = torch.tensor([0.0, 2.0, 2.0, 4.0, 0.0, -2.0, -2.0, -4.0], dtype=torch.float32)
    assert torch.equal(out, expected)


def test_non_halves_match_torch_round():
    br = BankersRounding()
    # values deliberately avoid .5 ties
    vals = [
        0.1, 0.2, 0.3, 0.4, 0.6, 0.7, 0.8, 0.9,
        1.1, 1.4, 1.6, 1.9,
        -0.1, -0.2, -0.3, -0.4, -0.6, -0.7, -0.8, -0.9,
        -1.1, -1.4, -1.6, -1.9,
    ]
    x = torch.tensor(vals, dtype=torch.float32)
    out = br.round(x)
    expected = torch.round(x)
    assert torch.equal(out, expected)


def test_integer_values_unchanged():
    br = BankersRounding()
    x = torch.tensor([0.0, 1.0, -2.0, 100.0], dtype=torch.float32)
    out = br.round(x)
    assert torch.equal(out, x)


def test_dtype_preserved():
    br = BankersRounding()
    for dtype in (torch.float32, torch.float64):
        x = torch.tensor([0.5, 1.2, -1.5, -2.2], dtype=dtype)
        out = br.round(x)
        assert out.dtype == dtype


def test_mixed_tensor_behaviour():
    br = BankersRounding()
    x = torch.tensor([
        -3.6, -3.5, -3.4,
        -2.6, -2.5, -2.4,
        -1.6, -1.5, -1.4,
        -0.6, -0.5, -0.4,
         0.4,  0.5,  0.6,
         1.4,  1.5,  1.6,
         2.4,  2.5,  2.6,
         3.4,  3.5,  3.6,
    ], dtype=torch.float32)

    out = br.round(x)
    expected = torch.tensor([
        -4.0, -4.0, -3.0,
        -3.0, -2.0, -2.0,
        -2.0, -2.0, -1.0,
        -1.0,  0.0,  0.0,
         0.0,  0.0,  1.0,
         1.0,  2.0,  2.0,
         2.0,  2.0,  3.0,
         3.0,  4.0,  4.0,
    ], dtype=torch.float32)

    assert torch.equal(out, expected)
