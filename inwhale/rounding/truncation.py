import torch 

from .base import RoundingStrategy

class TruncationRounding(RoundingStrategy):
    """
    Truncation rounding simply discards the fractional part of the number.
    For positive numbers, this is equivalent to taking the floor.
    For negative numbers, this is equivalent to taking the ceiling.

    For example:
    2.7 -> 2.0
    -2.7 -> -2.0

    This method does not perform any rounding up; it always rounds towards zero.
    """
    def round(self, x):
        return torch.trunc(x)