import abc
import typing
import pandas as pd
import numpy as np
import re
import typing
import os


BASELINE = dict(
    cores_per_cluster=1,
    num_clusters=28,
)


def function_name_from_signature(sig: str) -> str:
    pat = re.compile(r"^\s*(?:\w+\s+)?(\w+)(?:<([^>]+)>)?\s*(?:\(([^)]*)\))?\s*$")
    matches = re.findall(pat, sig)
    try:
        return matches[0][0]
    except IndexError as e:
        print(sig, matches)
        raise e


class BenchConfig(typing.TypedDict):
    name: str
    benchmark_idx: int
    uid: str

    path: os.PathLike
    executable: os.PathLike

    values: typing.Dict[str, typing.Any]


class Stats:
    bench_config: BenchConfig
    result_df: pd.DataFrame

    def __init__(self, result_df: pd.DataFrame) -> None:
        self.result_df = result_df

    def cores_per_cluster(self):
        return int(
            self.bench_config["values"].get(
                "cores_per_cluster", BASELINE["cores_per_cluster"]
            )
        )

    def num_clusters(self):
        return int(
            self.bench_config["values"].get("num_clusters", BASELINE["num_clusters"])
        )

    def total_cores(self):
        return self.cores_per_cluster() * self.num_clusters()

    # def exec_time_sec(self) -> float:
    #     return float(self.result_df["exec_time_sec"].mean())
    #
    # def cycles(self) -> float:
    #     return float(self.result_df["cycles"].mean())
    #
    # def num_blocks(self) -> float:
    #     return float(self.result_df["num_blocks"].mean())
    #
    # def instructions(self) -> float:
    #     return float(self.result_df["instructions"].mean())
    #
    # def warp_instructions(self) -> float:
    #     return float(self.result_df["warp_inst"].mean())
    #
    # def dram_reads(self) -> float:
    #     return float(self.result_df["dram_reads"].mean())
    #
    # def dram_writes(self) -> float:
    #     return float(self.result_df["dram_writes"].mean())
    #
    # def dram_accesses(self) -> float:
    #     return float(self.result_df["dram_accesses"].mean())
    #
    # def l2_reads(self) -> float:
    #     return float(self.result_df["l2_reads"].mean())
    #
    # def l1_reads(self) -> float:
    #     return float(self.result_df["l1_reads"].mean())
    #
    # def l2_writes(self) -> float:
    #     return float(self.result_df["l2_writes"].mean())
    #
    # def l2_accesses(self) -> float:
    #     return float(self.result_df["l2_accesses"].mean())
    #
    # def l2_read_hit_rate(self) -> float:
    #     return float(self.result_df["l2_read_hit_rate"].mean())
    #
    # def l2_write_hit_rate(self) -> float:
    #     return float(self.result_df["l2_write_hit_rate"].mean())
    #
    # def l2_read_miss_rate(self) -> float:
    #     return float(self.result_df["l2_read_miss_rate"].mean())
    #
    # def l2_write_miss_rate(self) -> float:
    #     return float(self.result_df["l2_write_miss_rate"].mean())
    #
    # def l2_read_hits(self) -> float:
    #     return float(self.result_df["l2_read_hits"].mean())
    #
    # def l2_write_hits(self) -> float:
    #     return float(self.result_df["l2_write_hits"].mean())
    #
    # def l2_read_misses(self) -> float:
    #     return float(self.result_df["l2_read_misses"].mean())
    #
    # def l2_write_misses(self) -> float:
    #     return float(self.result_df["l2_write_misses"].mean())
    #
    # def l2_hits(self) -> float:
    #     return float(self.result_df["l2_hits"].mean())
    #
    # def l2_misses(self) -> float:
    #     return float(self.result_df["l2_misses"].mean())

    # @abstractmethod
    # def cycles(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def instructions(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def num_blocks(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def exec_time_sec(self) -> float:
    #     pass
    #
    # @abstractmethod
    # def warp_instructions(self) -> float:
    #     pass
    #
    # @abstractmethod
    # def dram_reads(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def dram_writes(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def dram_accesses(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_reads(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_writes(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_accesses(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_read_hits(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_write_hits(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_read_misses(self) -> int:
    #     pass
    #
    # @abstractmethod
    # def l2_write_misses(self) -> int:
    #     pass

    def print_all_stats(self):
        print("instructions", self.instructions())
        print("warp instructions", self.warp_instructions())
        print("num blocks", self.num_blocks())
        print("cycles", self.cycles())
        print("exec time sec", self.exec_time_sec())
        print("dram reads", self.dram_reads())
        print("dram writes", self.dram_writes())
        print("dram accesses", self.dram_accesses())
        print("l2 reads", self.l2_reads())
        print("l2 writes", self.l2_writes())
        print("l2 accessses", self.l2_accesses())
        print("l2 read hits", self.l2_read_hits())
        print("l2 write hits", self.l2_write_hits())
        print("l2 read misses", self.l2_read_misses())
        print("l2 write misses", self.l2_write_misses())


# STAT_SUFFIXES = ["_mean", "_max", "_min", "_std"]
#
#
# def stat_cols(col):
#     return [col + suf for suf in STAT_SUFFIXES]
#


# def compute_df_statistics(df: pd.DataFrame, group_by: typing.List[str] | None, agg=None):
#     all_columns = set(df.columns.tolist())
#     all_columns = all_columns.difference(group_by or [])
#     all_columns = sorted(list(all_columns))
#
#     if agg is None:
#         agg = dict()
#
#     if isinstance(group_by, list) and len(group_by) > 0:
#         grouped = df.groupby(group_by)
#     else:
#         grouped = df.groupby(lambda _: True)
#
#     df_mean = grouped.agg({**{c: "mean" for c in all_columns}, **agg})
#     df_mean = df_mean.rename(columns={c: c + "_mean" for c in df_mean.columns})
#
#     df_max = grouped.agg({**{c: "max" for c in all_columns}, **agg})
#     df_max = df_max.rename(columns={c: c + "_max" for c in df_max.columns})
#
#     df_min = grouped.agg({**{c: "min" for c in all_columns}, **agg})
#     df_min = df_min.rename(columns={c: c + "_min" for c in df_min.columns})
#
#     def std(x):
#         return np.std(x)
#
#     df_std = grouped.agg({**{c: std for c in all_columns}, **agg})
#     df_std = df_std.rename(columns={c: c + "_std" for c in df_std.columns})
#
#     return pd.concat([df_mean, df_max, df_min, df_std], axis=1)
