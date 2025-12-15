import warnings
import torch

warnings.filterwarnings(
    "ignore", category=UserWarning, module="torch._subclasses.functional_tensor"
)

from inwhale.core.uniform import DeadZoneSymmetricQuantizer
from inwhale.observers.minmax import MinMaxObserver
from inwhale.rounding.nearest import NearestRounding


x = torch.tensor([0.001, -0.003, 0.006, -0.009, 0.02, -0.04, 0.3, -1.9])

observer = MinMaxObserver()
rounding = NearestRounding()
quant = DeadZoneSymmetricQuantizer(
    bits=8,
    observer=observer,
    rounding=rounding,
    threshold_ratio=0.5,
)

"""
the observer sees:
min = -1.9
max = 1.2
max_abs = 1.9

for 8 bits:
qmax = 127
scale = max_abs / qmax
      = 1.9 / 127
      = 0.01496

dead-zone threshold:
threshold = threshold_ratio * scale
          = 0.5 * 0.01496
          = 0.00748

quantization logic:
- values |x| < threshold -> forced to zero
- values outside the dead-zone:
    (|x| - threshold) / scale -> round -> restore sign

this creates an explicit zero region, reducing noise around 0
and encouraging sparsity.
"""

qx = quant.quantize(x)
dx = quant.dequantize(qx)

print("Original:", x)
print("Quantized:", qx)
print("Dequantized:", dx)
print("Absolute error:", (x - dx).abs())
