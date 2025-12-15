import torch

from .base import RoundingStrategy


class NearestRounding(RoundingStrategy):
    def round(self, x):
        return torch.round(x)
