from abc import ABC, abstractmethod


class BaseQuantizer(ABC):
    """
    Docstring for BaseQuantizer

    Base class for quantizers.

    Quantization maps high precision floating-point numbers (say 32 bits) to low precision integers (say 8 bits). This reduces:
    1. memory footprint (32 -> 8 bit = 75% reduction)
    2. computational requirements (integer math is faster and less power-hungry than floating point math)

    The challenge is doing this with minimal accuracy loss.
    """

    def __init__(self, bits: int):
        if bits < 0:
            raise ValueError("Bits must be positive.")
        self.bits = bits
        self.qmin = 0
        self.qmax = (
            1 << bits
        ) - 1  # hardware friendly way to determine the max value given the bits-value
        # say its 8 bits, its (1 << 8) - 1 = 256 - 1 = 255

    @classmethod
    @abstractmethod
    def quantize(self, x):
        pass

    @classmethod
    @abstractmethod
    def dequantize(self, x):
        pass
