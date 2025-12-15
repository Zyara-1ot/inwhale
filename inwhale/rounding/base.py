from abc import ABC, abstractmethod


class RoundingStrategy(ABC):
    @abstractmethod
    def round(cls, x):
        pass
