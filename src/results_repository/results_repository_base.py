from abc import ABC, abstractmethod


class ResultsRepositoryBase(ABC):

    @abstractmethod
    def save(self, results_dict: dict) -> bool:
        raise NotImplementedError()

    @abstractmethod
    def get(self) -> dict:
        raise NotImplementedError()
