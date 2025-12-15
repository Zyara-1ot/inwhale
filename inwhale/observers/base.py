from abc import ABC, abstractmethod


class Observer(ABC):
    @classmethod
    @abstractmethod
    def observe(self, x):
        pass

    @classmethod
    @abstractmethod
    def get_range(self):
        pass
