import pandas as pd
from pathlib import Path
from os import PathLike
from typing import Sequence
import json
import itertools

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common

# ("GLOBAL_ACC_R", "global_read"),
#         ("LOCAL_ACC_R", "local_read"),
#         ("CONST_ACC_R", "constant_read"),
#         ("TEXTURE_ACC_R", "texture_read"),
#         ("GLOBAL_ACC_W", "global_write"),
#         ("LOCAL_ACC_W", "local_write"),
#         ("L1_WRBK_ACC", "l1_writeback"),
#         ("L2_WRBK_ACC", "l2_writeback"),
#         ("INST_ACC_R", "inst_read"),
#         ("L1_WR_ALLOC_R", "l1_write_alloc_read"),
#         ("L2_WR_ALLOC_R", "l2_write_alloc_read"),

WARP_SIZE = 32

READ_ACCESS_KINDS = [
    "GLOBAL_ACC_R",
    "LOCAL_ACC_R",
    "CONST_ACC_R",
    "TEXTURE_ACC_R",
    "INST_ACC_R",
    "L1_WR_ALLOC_R",
    "L2_WR_ALLOC_R",
]
WRITE_ACCESS_KINDS = ["GLOBAL_ACC_W", "LOCAL_ACC_W", "L1_WRBK_ACC", "L2_WRBK_ACC"]


def access_is_write(access_type: str) -> bool:
    if access_type.upper() in READ_ACCESS_KINDS:
        return False
    if access_type.upper() in WRITE_ACCESS_KINDS:
        return True
    raise ValueError(f"bad access type: {access_type}")

    # match access_type.upper():
    #     case "GLOBAL_ACC_R" | "LOCAL_ACC_R" | "CONST_ACC_R" | "TEXTURE_ACC_R" | "INST_ACC_R" | "L1_WR_ALLOC_R" | "L2_WR_ALLOC_R":
    #         return False
    #     case "GLOBAL_ACC_W" | "LOCAL_ACC_W" | "L1_WRBK_ACC" | "L2_WRBK_ACC":
    #         return True
    #     case other:
    #         raise ValueError(f"bad access type: {other}")


def parse_cache_stats(path: PathLike):
    stats = pd.read_csv(
        path,
        header=None,
        names=["cache_id", "access_type", "status", "count"],
    )
    stats["is_write"] = stats["access_type"].apply(access_is_write)
    return stats


class Stats(common.Stats):
    bench_config: BenchConfig
    config: GPUConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["accelsim_simulate"]["stats_dir"])
        self.use_duration = False
        self.config = config
        self._load_bench_config(bench_config)

    def _load_bench_config(self, bench_config: BenchConfig) -> None:
        self.bench_config = bench_config

        print(self.path / "exec_time.release.json", "rb")
        with open(self.path / "exec_time.release.json", "rb") as f:
            # convert millis to seconds
            self.exec_time_sec_release = float(json.load(f)) * 1e-3

        self.sim_df = pd.read_csv(
            self.path / "stats.sim.csv",
            header=0,
        )
        self.accesses_df = pd.read_csv(self.path / "stats.accesses.csv", header=None, names=["access", "count"])
        self.dram_df = pd.read_csv(
            self.path / "stats.dram.csv",
            header=0,
        )
        self.dram_banks_df = pd.read_csv(
            self.path / "stats.dram.banks.csv",
            header=0,
        )
        self.instructions_df = pd.read_csv(
            self.path / "stats.instructions.csv",
            header=None,
            names=["memory_space", "write", "count"],
        )
        self.l1_inst_stats = parse_cache_stats(self.path / "stats.cache.l1i.csv")
        self.l1_tex_stats = parse_cache_stats(self.path / "stats.cache.l1t.csv")
        self.l1_data_stats = parse_cache_stats(self.path / "stats.cache.l1d.csv")
        self.l1_const_stats = parse_cache_stats(self.path / "stats.cache.l1c.csv")

        self.l2_data_stats = parse_cache_stats(self.path / "stats.cache.l2d.csv")

        self.raw_stats_df = pd.read_csv(
            self.path / "raw.stats.csv",
            header=None,
            names=["kernel", "kernel_id", "stat", "value"],
        )
        self.raw_stats_df = self.raw_stats_df.pivot(
            index=["kernel", "kernel_id"],
            columns=["stat"],
        )["value"]
        # self.raw_stats_df = self.raw_stats_df.reset_index()
        # print(self.raw_stats_df.index)
        # only keep the final kernel info
        self.raw_stats_df = self.raw_stats_df.loc["final_kernel", 0].reset_index()
        self.raw_stats_df.columns = ["stat", "value"]
        self.raw_stats_df = self.raw_stats_df.set_index("stat").T
        # self.raw_stats_df = self.raw_stats_df.T
        # self.raw_stats_df = self.raw_stats_df.T.reset_index(drop=True)
        # .reset_index()
        # print(self.raw_stats_df.index)
        # print(self.raw_stats_df.columns)
        # self.raw_stats_df = self.raw_stats_df[
        #     self.raw_stats_df["kernel"] == "final_kernel" & self.raw_stats_df["kernel_id"] == 0
        # ]
        # print(self.raw_stats_df)

    def exec_time_sec(self) -> float:
        return self.exec_time_sec_release

    def cycles(self) -> int:
        assert self.raw_stats_df["gpu_tot_sim_cycle"].sum() == self.sim_df["cycles"].sum()
        return self.sim_df["cycles"].sum()

    def warp_instructions(self) -> float:
        return self.raw_stats_df["warp_instruction_count"].mean() / float(WARP_SIZE)

    def instructions(self) -> int:
        assert self.raw_stats_df["gpu_total_instructions"].sum() == self.sim_df["instructions"].sum()
        return self.sim_df["instructions"].sum()

    def dram_reads(self) -> int:
        assert self.raw_stats_df["total_dram_reads"].sum() == self.dram_df["reads"].sum()
        return int(self.dram_df["reads"].sum())

    def dram_writes(self) -> int:
        assert self.raw_stats_df["total_dram_writes"].sum() == self.dram_df["writes"].sum()
        return int(self.dram_df["writes"].sum())

    def dram_accesses(self) -> int:
        assert (
            self.raw_stats_df[["total_dram_writes", "total_dram_reads"]].sum().sum()
            == self.dram_df[["reads", "writes"]].sum().sum()
        )
        return int(self.dram_df[["reads", "writes"]].sum().sum())

    def l2_reads(self) -> int:
        # print(self.l2_data_stats[self.l2_data_stats["count"] != 0])
        hit_mask = self.l2_data_stats["status"] == "HIT"
        miss_mask = self.l2_data_stats["status"] == "MISS"
        read_mask = self.l2_data_stats["is_write"] == False
        reads = self.l2_data_stats[(hit_mask ^ miss_mask) & read_mask]
        return int(reads["count"].sum())

    def l2_writes(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        miss_mask = self.l2_data_stats["status"] == "MISS"
        write_mask = self.l2_data_stats["is_write"] == True
        reads = self.l2_data_stats[(hit_mask ^ miss_mask) & write_mask]
        return int(reads["count"].sum())

    def l2_accesses(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        miss_mask = self.l2_data_stats["status"] == "MISS"
        accesses = self.l2_data_stats[hit_mask ^ miss_mask]
        return int(accesses["count"].sum())

    def get_raw_l2_read_stats(self, status: Sequence[str]):
        return self.get_raw_l2_stats(READ_ACCESS_KINDS, status)

    def get_raw_l2_write_stats(self, status: Sequence[str]):
        return self.get_raw_l2_stats(WRITE_ACCESS_KINDS, status)

    def get_raw_l2_stats(self, kind: Sequence[str], status: Sequence[str]):
        cols = [f"l2_cache_{k.upper()}_{s.upper()}" for (k, s) in itertools.product(kind, status)]
        return self.raw_stats_df[cols]

    def l2_read_hits(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        read_mask = self.l2_data_stats["is_write"] == False
        read_hits = self.l2_data_stats[hit_mask & read_mask]

        assert self.get_raw_l2_read_stats(["HIT"]).sum().sum() == read_hits["count"].sum()
        return int(read_hits["count"].sum())

    def l2_write_hits(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        write_mask = self.l2_data_stats["is_write"] == True
        write_hits = self.l2_data_stats[hit_mask & write_mask]

        assert self.get_raw_l2_write_stats(["HIT"]).sum().sum() == write_hits["count"].sum()
        return int(write_hits["count"].sum())

    def l2_read_misses(self) -> int:
        miss_mask = self.l2_data_stats["status"] == "MISS"
        read_mask = self.l2_data_stats["is_write"] == False
        read_misses = self.l2_data_stats[miss_mask & read_mask]

        assert self.get_raw_l2_read_stats(["MISS"]).sum().sum() == read_misses["count"].sum()
        return int(read_misses["count"].sum())

    def l2_write_misses(self) -> int:
        miss_mask = self.l2_data_stats["status"] == "MISS"
        write_mask = self.l2_data_stats["is_write"] == True
        write_misses = self.l2_data_stats[miss_mask & write_mask]

        assert self.get_raw_l2_write_stats(["MISS"]).sum().sum() == write_misses["count"].sum()
        return int(write_misses["count"].sum())
