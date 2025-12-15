import torch
from .base import Observer


class MinMaxObserver(Observer):
    def __init__(self):
        super().__init__()
        self.min_val = None
        self.max_val = None

    def observe(self, x):
        min_x = x.min()
        max_x = x.max()

        if self.min_val is None or self.max_val is None:
            self.min_val = min_x
            self.max_val = max_x
        else:
            self.min_val = torch.min(self.min_val, min_x)
            self.max_val = torch.max(self.max_val, max_x)

    def get_range(self):
        if self.min_val is None:
            raise RuntimeError("No data observed yet.")
        return self.min_val, self.max_val
