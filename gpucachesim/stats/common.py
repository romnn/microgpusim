import abc
from abc import abstractmethod


class Stats(abc.ABC):
    @abstractmethod
    def cycles(self) -> int:
        pass

    @abstractmethod
    def instructions(self) -> int:
        pass
