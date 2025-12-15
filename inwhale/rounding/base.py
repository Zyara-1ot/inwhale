from abc import ABC, abstractmethod


class RoundingStrategy(ABC):
    @classmethod
    @abstractmethod
    def round(cls, x):
        pass
