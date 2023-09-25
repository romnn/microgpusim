import pandas as pd
from pathlib import Path
from typing import Sequence
import itertools

from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    AccelsimSimulateTargetConfig,
    AccelsimSimulateConfig,
)
import gpucachesim.stats.common as common
import gpucachesim.stats.stats as stats
from gpucachesim.stats.common import stat_cols, STAT_SUFFIXES


class Stats(stats.Stats):
    bench_config: BenchConfig[AccelsimSimulateTargetConfig]
    target_config: AccelsimSimulateConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig[AccelsimSimulateTargetConfig]) -> None:
        self.bench_config = bench_config
        self.target_config = self.bench_config["target_config"].value
        self.path = Path(self.target_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = self.bench_config["common"]["repetitions"]
        self.load_converted_stats()
        self.load_raw_stats()

        self.compute_result_df()

        # add the input configs

        try:
            # sanity checks
            assert self.raw_stats_df["num_issued_blocks"].sum() == self.num_blocks()
            assert self.raw_stats_df["gpu_tot_sim_cycle"].sum() == self.cycles()
            assert self.raw_stats_df["gpu_total_instructions"].sum() == self.instructions()
            assert self.raw_stats_df["total_dram_reads"].sum() == self.dram_reads()
            assert self.raw_stats_df["total_dram_writes"].sum() == self.dram_writes()
            assert self.raw_stats_df[["total_dram_writes", "total_dram_reads"]].sum().sum() == self.dram_accesses()

            # l2_cache_global_write_total == l2_cache_GLOBAL_ACC_W_TOTAL_ACCESS
            # l2_cache_global_read_total == l2_cache_GLOBAL_ACC_R_TOTAL_ACCESS
            # print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "MISS", "SECTOR_MISS"]))
            print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], stats.ACCESS_STATUSES))
            print(self._get_raw_l2_stats(["GLOBAL_ACC_W"], stats.ACCESS_STATUSES).sum().sum())
            print(self.raw_stats_df["l2_cache_global_write_total"])
            assert (
                self._get_raw_l2_stats(["GLOBAL_ACC_W"], ["HIT", "HIT_RESERVED", "MISS", "SECTOR_MISS"]).sum().sum()
                == self.raw_stats_df["l2_cache_global_write_total"].sum()
            )
            assert (
                self._get_raw_l2_stats(["GLOBAL_ACC_R"], ["HIT", "MISS", "SECTOR_MISS"]).sum().sum()
                == self.raw_stats_df["l2_cache_global_read_total"].sum()
            )
            assert self.raw_stats_df["l2_cache_global_write_total"].sum() == self.l2_writes()
            assert self.raw_stats_df["l2_cache_global_read_total"].sum() == self.l2_reads()

            assert self._get_raw_l2_read_stats(["HIT"]).sum().sum() == self.l2_read_hits()
            assert self._get_raw_l2_write_stats(["HIT"]).sum().sum() == self.l2_write_hits()
            assert self._get_raw_l2_read_stats(["MISS", "SECTOR_MISS"]).sum().sum() == self.l2_read_misses()
            assert self._get_raw_l2_write_stats(["MISS", "SECTOR_MISS"]).sum().sum() == self.l2_write_misses()

            assert (
                self.raw_stats_df["l2_cache_GLOBAL_ACC_W_HIT"] / self.raw_stats_df["l2_cache_global_write_total"]
            ).sum() == self.l2_write_hit_rate()
            assert (
                self.raw_stats_df["l2_cache_GLOBAL_ACC_R_HIT"] / self.raw_stats_df["l2_cache_global_read_total"]
            ).sum() == self.l2_read_hit_rate()

            # print(self.raw_stats_df[[c for c in self.raw_stats_df if "l1_data_cache" in c]])
            assert self.raw_stats_df["l1_data_cache_global_read_total"].sum() == self.l1_reads()
        except AssertionError as e:
            print(f"WARNING: {e}")

    def load_raw_stats(self) -> None:
        raw_stats_dfs = []
        for r in range(self.repetitions):
            raw_stats_df = pd.read_csv(
                self.path / f"raw.stats.{r}.csv",
                header=None,
                names=["kernel", "kernel_id", "stat", "value"],
            )
            raw_stats_df = raw_stats_df.pivot(
                index=["kernel", "kernel_id"],
                columns=["stat"],
            )["value"]
            # only keep the final kernel info
            raw_stats_df = raw_stats_df.loc["final_kernel", 0].reset_index()
            raw_stats_df.columns = ["stat", "value"]
            raw_stats_df = raw_stats_df.set_index("stat").T

            raw_stats_dfs.append(raw_stats_df)

        self.raw_stats_df = pd.concat(raw_stats_dfs)
        # self.raw_stats_df = common.compute_df_statistics(raw_stats_df, group_by=None)

    def _get_raw_l2_read_stats(self, status: Sequence[str]):
        return self._get_raw_l2_stats(stats.READ_ACCESS_KINDS, status)

    def _get_raw_l2_write_stats(self, status: Sequence[str]):
        return self._get_raw_l2_stats(stats.WRITE_ACCESS_KINDS, status)

    def _get_raw_l2_stats(self, kind: Sequence[str], status: Sequence[str]):
        cols = [f"l2_cache_{k.upper()}_{s.upper()}" for (k, s) in itertools.product(kind, status)]
        return self.raw_stats_df[cols]

    def _compute_warp_instructions(self):
        num_warps = self.result_df["num_blocks"] * stats.WARP_SIZE
        # print("this", self.raw_stats_df["warp_instruction_count"].values)
        # print(num_warps)
        self.result_df["warp_inst"] = self.raw_stats_df["warp_instruction_count"].values / num_warps
        # for s in STAT_SUFFIXES:
        #     self.result_df["warp_inst" + s] = self.raw_stats_df["warp_instruction_count" + s]
        #     self.result_df["warp_inst" + s] /= self.result_df["num_blocks" + s] * stats.WARP_SIZE

    # def _compute_num_blocks(self):
    #     self.result_df[stat_cols("num_blocks")] = self.raw_stats_df[stat_cols("num_issued_blocks")]

    # DONE

    # def exec_time_sec(self) -> float:
    #     return self.exec_time_sec_release["exec_time_mean"].sum()
    #
    # def cycles(self) -> float:
    #     cycles = self.sim_df["cycles_mean"].sum()
    #     assert self.raw_stats_df["gpu_tot_sim_cycle_mean"].sum() == cycles
    #     return cycles
    #
    # def num_warps(self) -> float:
    #     return self.num_blocks() * stats.WARP_SIZE
    #
    # def warp_instructions(self) -> float:
    #     num_instructions = self.raw_stats_df["warp_instruction_count_mean"].mean()
    #     num_warps = self.num_warps()
    #     return num_instructions / num_warps
    #
    # def instructions(self) -> float:
    #     instructions = self.sim_df["instructions_mean"].sum()
    #     assert self.raw_stats_df["gpu_total_instructions_mean"].sum() == instructions
    #     return instructions
    #
    # def num_blocks(self) -> float:
    #     return int(self.raw_stats_df["num_issued_blocks_mean"].sum())
    #
    # def dram_reads(self) -> float:
    #     dram_reads = self.dram_df["reads_mean"].sum()
    #     assert self.raw_stats_df["total_dram_reads_mean"].sum() == dram_reads
    #     return dram_reads
    #
    # def dram_writes(self) -> float:
    #     dram_writes = self.dram_df["writes_mean"].sum()
    #     assert self.raw_stats_df["total_dram_writes_mean"].sum() == dram_writes
    #     return dram_writes
    #
    # def dram_accesses(self) -> float:
    #     dram_accesses = self.dram_df[["reads_mean", "writes_mean"]].sum().sum()
    #     assert self.raw_stats_df[["total_dram_writes_mean", "total_dram_reads_mean"]].sum().sum() == dram_accesses
    #     return dram_accesses
    #
    # def get_raw_l2_read_stats(self, status: Sequence[str]):
    #     return self.get_raw_l2_stats(stats.READ_ACCESS_KINDS, status)
    #
    # def get_raw_l2_write_stats(self, status: Sequence[str]):
    #     return self.get_raw_l2_stats(stats.WRITE_ACCESS_KINDS, status)
    #
    # def get_raw_l2_stats(self, kind: Sequence[str], status: Sequence[str]):
    #     cols = [f"l2_cache_{k.upper()}_{s.upper()}_mean" for (k, s) in itertools.product(kind, status)]
    #     return self.raw_stats_df[cols]
    #
    # def l2_read_hits(self) -> float:
    #     read_hits = super().l2_read_hits()
    #     assert self.get_raw_l2_read_stats(["HIT"]).sum().sum() == read_hits
    #     return read_hits
    #
    # def l2_write_hits(self) -> float:
    #     write_hits = super().l2_write_hits()
    #     assert self.get_raw_l2_write_stats(["HIT"]).sum().sum() == write_hits
    #     return write_hits
    #
    # def l2_read_misses(self) -> float:
    #     read_misses = super().l2_read_misses()
    #     assert self.get_raw_l2_read_stats(["MISS"]).sum().sum() == read_misses
    #     return read_misses
    #
    # def l2_write_misses(self) -> float:
    #     write_misses = super().l2_write_misses()
    #     assert self.get_raw_l2_write_stats(["MISS"]).sum().sum() == write_misses
    #     return write_misses
