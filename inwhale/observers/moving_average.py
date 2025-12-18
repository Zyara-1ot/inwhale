from .base import Observer


class MovingAverageObserver(Observer):
    def __init__(self, momentum=0.1):
        super().__init__()
        if not 0 < momentum <= 1:
            raise ValueError(f"Momentum must be in the range (0, 1], got {momentum}.")
        self.momentum = momentum
        self.min_val = None
        self.max_val = None

    def observe(self, x):
        if not hasattr(x, "min") or not hasattr(x, "max"):
            raise TypeError(
                f"Input must support min() and max() methods, got {type(x).__name__}"
            )
        min_x = x.min()
        max_x = x.max()

        if self.min_val is None or self.max_val is None:
            self.min_val = min_x
            self.max_val = max_x
        else:
            self.min_val = (1 - self.momentum) * self.min_val + self.momentum * min_x
            self.max_val = (1 - self.momentum) * self.max_val + self.momentum * max_x

    def get_range(self):
        if self.min_val is None:
            raise RuntimeError("No data observed yet.")
        return self.min_val, self.max_val
