from abc import ABC, abstractmethod


class Fi(ABC):
    @abstractmethod
    def __call__(self, state):
        pass
