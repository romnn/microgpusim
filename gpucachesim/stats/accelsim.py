import pandas as pd
from pathlib import Path
from os import PathLike
from typing import Sequence
import json
import itertools

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common
import gpucachesim.stats.stats as stats


class Stats(stats.Stats):
    def __init__(self, config: GPUConfig, global_bench_config: BenchConfig) -> None:
        self.bench_config = global_bench_config["accelsim_simulate"]
        self.path = Path(self.bench_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = int(self.bench_config["repetitions"])
        self.load_converted_stats()
        self.load_raw_stats()

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
            # self.raw_stats_df = self.raw_stats_df.reset_index()
            # print(self.raw_stats_df.index)
            # only keep the final kernel info
            raw_stats_df = raw_stats_df.loc["final_kernel", 0].reset_index()
            raw_stats_df.columns = ["stat", "value"]
            raw_stats_df = raw_stats_df.set_index("stat").T
            # print(raw_stats_df)

            raw_stats_dfs.append(raw_stats_df)
            # self.raw_stats_df = self.raw_stats_df.T
            # self.raw_stats_df = self.raw_stats_df.T.reset_index(drop=True)
            # .reset_index()
            # print(self.raw_stats_df.index)
            # print(self.raw_stats_df.columns)
            # self.raw_stats_df = self.raw_stats_df[
            #     self.raw_stats_df["kernel"] == "final_kernel" & self.raw_stats_df["kernel_id"] == 0
            # ]
            # print(self.raw_stats_df)
        raw_stats_df = pd.concat(raw_stats_dfs)
        print(raw_stats_df)
        self.raw_stats_df = common.compute_df_statistics(raw_stats_df, group_by=None)
        print(self.raw_stats_df)

    # def _load_bench_config(self, bench_config: BenchConfig) -> None:
    #     self.bench_config = bench_config
    #
    #     with open(self.path / "exec_time.release.json", "rb") as f:
    #         # convert millis to seconds
    #         self.exec_time_sec_release = float(json.load(f)) * 1e-3
    #
    #     self.sim_df = pd.read_csv(
    #         self.path / "stats.sim.csv",
    #         header=0,
    #     )
    #     self.accesses_df = pd.read_csv(self.path / "stats.accesses.csv", header=None, names=["access", "count"])
    #     self.dram_df = pd.read_csv(
    #         self.path / "stats.dram.csv",
    #         header=0,
    #     )
    #     self.dram_banks_df = pd.read_csv(
    #         self.path / "stats.dram.banks.csv",
    #         header=0,
    #     )
    #     self.instructions_df = pd.read_csv(
    #         self.path / "stats.instructions.csv",
    #         header=None,
    #         names=["memory_space", "write", "count"],
    #     )
    #     self.l1_inst_stats = stats.parse_cache_stats(self.path / "stats.cache.l1i.csv")
    #     self.l1_tex_stats = stats.parse_cache_stats(self.path / "stats.cache.l1t.csv")
    #     self.l1_data_stats = stats.parse_cache_stats(self.path / "stats.cache.l1d.csv")
    #     self.l1_const_stats = stats.parse_cache_stats(self.path / "stats.cache.l1c.csv")
    #
    #     self.l2_data_stats = stats.parse_cache_stats(self.path / "stats.cache.l2d.csv")
    #
    #     self.raw_stats_df = pd.read_csv(
    #         self.path / "raw.stats.csv",
    #         header=None,
    #         names=["kernel", "kernel_id", "stat", "value"],
    #     )
    #     self.raw_stats_df = self.raw_stats_df.pivot(
    #         index=["kernel", "kernel_id"],
    #         columns=["stat"],
    #     )["value"]
    #     # self.raw_stats_df = self.raw_stats_df.reset_index()
    #     # print(self.raw_stats_df.index)
    #     # only keep the final kernel info
    #     self.raw_stats_df = self.raw_stats_df.loc["final_kernel", 0].reset_index()
    #     self.raw_stats_df.columns = ["stat", "value"]
    #     self.raw_stats_df = self.raw_stats_df.set_index("stat").T
    #     # self.raw_stats_df = self.raw_stats_df.T
    #     # self.raw_stats_df = self.raw_stats_df.T.reset_index(drop=True)
    #     # .reset_index()
    #     # print(self.raw_stats_df.index)
    #     # print(self.raw_stats_df.columns)
    #     # self.raw_stats_df = self.raw_stats_df[
    #     #     self.raw_stats_df["kernel"] == "final_kernel" & self.raw_stats_df["kernel_id"] == 0
    #     # ]
    #     # print(self.raw_stats_df)

    def exec_time_sec(self) -> float:
        return self.exec_time_sec_release["exec_time_mean"].sum()

    def cycles(self) -> float:
        cycles = self.sim_df["cycles_mean"].sum()
        assert self.raw_stats_df["gpu_tot_sim_cycle_mean"].sum() == cycles
        return cycles

    def num_warps(self) -> float:
        return self.num_blocks() * stats.WARP_SIZE

    def warp_instructions(self) -> float:
        num_instructions = self.raw_stats_df["warp_instruction_count_mean"].mean()
        num_warps = self.num_warps()
        return num_instructions / num_warps

    def instructions(self) -> float:
        instructions = self.sim_df["instructions_mean"].sum()
        assert self.raw_stats_df["gpu_total_instructions_mean"].sum() == instructions
        return instructions

    def num_blocks(self) -> float:
        return int(self.raw_stats_df["num_issued_blocks_mean"].sum())

    def dram_reads(self) -> float:
        dram_reads = self.dram_df["reads_mean"].sum()
        assert self.raw_stats_df["total_dram_reads_mean"].sum() == dram_reads
        return dram_reads

    def dram_writes(self) -> float:
        dram_writes = self.dram_df["writes_mean"].sum()
        assert self.raw_stats_df["total_dram_writes_mean"].sum() == dram_writes
        return dram_writes

    def dram_accesses(self) -> float:
        dram_accesses = self.dram_df[["reads_mean", "writes_mean"]].sum().sum()
        assert self.raw_stats_df[["total_dram_writes_mean", "total_dram_reads_mean"]].sum().sum() == dram_accesses
        return dram_accesses

    # def l2_reads(self) -> float:
    #     # print(self.l2_data_stats[self.l2_data_stats["count"] != 0])
    #     hit_mask = self.l2_data_stats["status"] == "HIT"
    #     miss_mask = self.l2_data_stats["status"] == "MISS"
    #     read_mask = self.l2_data_stats["is_write"] == False
    #     reads = self.l2_data_stats[(hit_mask ^ miss_mask) & read_mask]
    #     return int(reads["count"].sum())
    #
    # def l2_writes(self) -> float:
    #     hit_mask = self.l2_data_stats["status"] == "HIT"
    #     miss_mask = self.l2_data_stats["status"] == "MISS"
    #     write_mask = self.l2_data_stats["is_write"] == True
    #     reads = self.l2_data_stats[(hit_mask ^ miss_mask) & write_mask]
    #     return int(reads["count"].sum())
    #
    # def l2_accesses(self) -> float:
    #     hit_mask = self.l2_data_stats["status"] == "HIT"
    #     miss_mask = self.l2_data_stats["status"] == "MISS"
    #     accesses = self.l2_data_stats[hit_mask ^ miss_mask]
    #     return int(accesses["count"].sum())

    def get_raw_l2_read_stats(self, status: Sequence[str]):
        return self.get_raw_l2_stats(stats.READ_ACCESS_KINDS, status)

    def get_raw_l2_write_stats(self, status: Sequence[str]):
        return self.get_raw_l2_stats(stats.WRITE_ACCESS_KINDS, status)

    def get_raw_l2_stats(self, kind: Sequence[str], status: Sequence[str]):
        cols = [f"l2_cache_{k.upper()}_{s.upper()}_mean" for (k, s) in itertools.product(kind, status)]
        return self.raw_stats_df[cols]

    def l2_read_hits(self) -> float:
        read_hits = super().l2_read_hits()
        assert self.get_raw_l2_read_stats(["HIT"]).sum().sum() == read_hits
        return read_hits
        # hit_mask = self.l2_data_stats["status"] == "HIT"
        # read_mask = self.l2_data_stats["is_write"] == False
        # read_hits = self.l2_data_stats[hit_mask & read_mask]
        #
        # assert self.get_raw_l2_read_stats(["HIT"]).sum().sum() == read_hits["count"].sum()
        # return int(read_hits["count"].sum())

    def l2_write_hits(self) -> float:
        write_hits = super().l2_write_hits()
        assert self.get_raw_l2_write_stats(["HIT"]).sum().sum() == write_hits
        return write_hits
        # hit_mask = self.l2_data_stats["status"] == "HIT"
        # write_mask = self.l2_data_stats["is_write"] == True
        # write_hits = self.l2_data_stats[hit_mask & write_mask]
        #
        # return int(write_hits["count"].sum())

    def l2_read_misses(self) -> float:
        read_misses = super().l2_read_misses()
        assert self.get_raw_l2_read_stats(["MISS"]).sum().sum() == read_misses
        return read_misses
        # miss_mask = self.l2_data_stats["status"] == "MISS"
        # read_mask = self.l2_data_stats["is_write"] == False
        # read_misses = self.l2_data_stats[miss_mask & read_mask]
        #
        # assert self.get_raw_l2_read_stats(["MISS"]).sum().sum() == read_misses["count"].sum()
        # return int(read_misses["count"].sum())

    def l2_write_misses(self) -> float:
        write_misses = super().l2_write_misses()
        assert self.get_raw_l2_write_stats(["MISS"]).sum().sum() == write_misses
        return write_misses
        # miss_mask = self.l2_data_stats["status"] == "MISS"
        # write_mask = self.l2_data_stats["is_write"] == True
        # write_misses = self.l2_data_stats[miss_mask & write_mask]
        #
        # assert self.get_raw_l2_write_stats(["MISS"]).sum().sum() == write_misses["count"].sum()
        # return int(write_misses["count"].sum())
