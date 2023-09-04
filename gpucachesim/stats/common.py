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
    def exec_time_sec(self) -> float:
        pass

    @abstractmethod
    def warp_instructions(self) -> float:
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

    @abstractmethod
    def l2_read_hits(self) -> int:
        pass

    @abstractmethod
    def l2_write_hits(self) -> int:
        pass

    @abstractmethod
    def l2_read_misses(self) -> int:
        pass

    @abstractmethod
    def l2_write_misses(self) -> int:
        pass
