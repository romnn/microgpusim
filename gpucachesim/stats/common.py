import abc
import typing
import pandas as pd
from abc import abstractmethod


class Stats(abc.ABC):
    @abstractmethod
    def cycles(self) -> int:
        pass

    @abstractmethod
    def instructions(self) -> int:
        pass

    @abstractmethod
    def num_blocks(self) -> int:
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


def compute_df_statistics(df: pd.DataFrame, group_by: typing.List[str] | None, agg=None):
    all_columns = set(df.columns.tolist())
    all_columns = all_columns.difference(group_by or [])
    all_columns = sorted(list(all_columns))

    if agg is None:
        agg = dict()

    if isinstance(group_by, list) and len(group_by) > 0:
        grouped = df.groupby(group_by)
    else:
        grouped = df.groupby(lambda _: True)

    df_mean = grouped.agg({**{c: "mean" for c in all_columns}, **agg})
    df_mean = df_mean.rename(columns={c: c + "_mean" for c in df_mean.columns})

    df_max = grouped.agg({**{c: "max" for c in all_columns}, **agg})
    df_max = df_max.rename(columns={c: c + "_max" for c in df_max.columns})

    df_min = grouped.agg({**{c: "min" for c in all_columns}, **agg})
    df_min = df_min.rename(columns={c: c + "_min" for c in df_min.columns})

    df_std = grouped.agg({**{c: "std" for c in all_columns}, **agg})
    df_std = df_std.rename(columns={c: c + "_std" for c in df_std.columns})

    return pd.concat([df_mean, df_max, df_min, df_std], axis=1)
