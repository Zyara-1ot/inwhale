import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.uniform import SymmetricUniformQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


x = torch.tensor([1.2, -0.7, 0.4, 2.3, -1.9])

observer = MinMaxObserver()
rounding = NearestRounding()
quant = SymmetricUniformQuantizer(bits=8, observer=observer, rounding=rounding)

"""
the observer sees:
min = -1.9
max = 2.3
max_abs = 2.3

for 8 bits,
qmax = 127
scale = max_abs / qmax = 2.3 / 127 = 0.01811 (approx)

now quantization happens:
x / scale = (approx) [66.2, -38.6, 22.1, 127.0, -104.9]
round() = [66., -39., 22., 127., -105.]

now for dequantization:
q * scale = (approx) [1.1953, -0.7063,  0.3984,  2.3000, -1.9016]

Error exists, but it's bounded, predictable and it shrinks with more bits.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)


print("Original:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())
