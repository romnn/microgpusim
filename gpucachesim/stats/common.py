import abc
from abc import abstractmethod


class Stats(abc.ABC):
    @abstractmethod
    def cycles(self) -> int:
        pass

    @abstractmethod
    def instructions(self) -> int:
        pass

    @abstractmethod
    def dram_reads(self) -> int:
        pass

    @abstractmethod
    def dram_writes(self) -> int:
        pass

    @abstractmethod
    def dram_accesses(self) -> int:
        pass

    @abstractmethod
    def l2_reads(self) -> int:
        pass

    @abstractmethod
    def l2_writes(self) -> int:
        pass

    @abstractmethod
    def l2_accesses(self) -> int:
        pass
