import torch
from .quantizer import BaseQuantizer


class SymmetricUniformQuantizer(BaseQuantizer):
    """
    Docstring for SymmetricUniformQuantizer

    The fundamental equation behind symmetric uniform quantization is:
    q = clamp(round(x / scale), qmin, qmax)
    and x' = q * scale
    where scale is computed based on the observed min and max values of the input tensor.

    In real networks, we quantize activations layer by layer and sometimes over many batches, before seeing all inputs.
    So we need a component whose ONLY job is to watch tensors go by and summarize their numeric range. This is the observer.

    And the MinMaxObserver answers one simple questions: What is the smallest and largest value I have ever seen?

    min_val, max_val = self.observer.get_range(), now we can compute the scale.

    Now calculating the scale is quite straightforward.
    Since we wish to _scale_ max_abs_value to qmax, we can just:

    scale = max_abs / qmax
    equivalently,
    max_abs = torch.max(min_val.abs(), max_val.abs())

    Finally, we need to ensure that scale is not zero (which would lead to division by zero during quantization).

    For which we did
    torch.clamp(self.scale, min=1e-8)

    Now one question still remains, why is rounding abstracted?
    Because, rounding is NOT JUST round(), it can change bias and the error distribution.

    There can be multiple strategies for rounding, and they can affect accumulated wrror, training stability, bias towards zero, etc.
    """

    def __init__(self, bits, observer, rounding):
        super().__init__(bits)

        self.observer = observer
        self.rounding = rounding
        self.scale = None

        self.qmin = -(1 << (bits - 1))
        self.qmax = (1 << (bits - 1)) - 1

    def _compute_scale(self, x):
        min_val, max_val = self.observer.get_range()

        max_abs = torch.max(min_val.abs(), max_val.abs())

        self.scale = max_abs / self.qmax
        self.scale = torch.clamp(self.scale, min=1e-8)

    def quantize(self, x):
        self.observer.observe(x)
        self._compute_scale(x)

        qx = x / self.scale
        qx = self.rounding.round(qx)
        qx = torch.clamp(qx, self.qmin, self.qmax)
        return qx

    def dequantize(self, qx):
        return qx * self.scale
