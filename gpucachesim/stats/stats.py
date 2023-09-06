import pandas as pd
import json
from os import PathLike
from pathlib import Path

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common

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


def parse_cache_stats(path: PathLike):
    stats = pd.read_csv(
        path,
        header=None,
        names=["cache_id", "access_type", "status", "count"],
    )
    stats["is_write"] = stats["access_type"].apply(access_is_write)
    return stats


class Stats(common.Stats):
    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.path = Path(bench_config["simulate"]["stats_dir"])
        self.use_duration = False
        self.config = config
        self._load_bench_config(bench_config)

    def _load_bench_config(self, bench_config: BenchConfig) -> None:
        self.bench_config = bench_config

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

    def exec_time_sec(self) -> float:
        return self.exec_time_sec_release

    def cycles(self) -> int:
        return self.sim_df["cycles"].sum()

    def warp_instructions(self) -> float:
        return 0

    def instructions(self) -> int:
        return self.sim_df["instructions"].sum()

    def dram_reads(self) -> int:
        return int(self.dram_df["reads"].sum())

    def dram_writes(self) -> int:
        return int(self.dram_df["writes"].sum())

    def dram_accesses(self) -> int:
        return int(self.dram_df[["reads", "writes"]].sum().sum())

    def l2_reads(self) -> int:
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

    def l2_read_hits(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        read_mask = self.l2_data_stats["is_write"] == False
        read_hits = self.l2_data_stats[hit_mask & read_mask]

        return int(read_hits["count"].sum())

    def l2_write_hits(self) -> int:
        hit_mask = self.l2_data_stats["status"] == "HIT"
        write_mask = self.l2_data_stats["is_write"] == True
        write_hits = self.l2_data_stats[hit_mask & write_mask]

        return int(write_hits["count"].sum())

    def l2_read_misses(self) -> int:
        miss_mask = self.l2_data_stats["status"] == "MISS"
        read_mask = self.l2_data_stats["is_write"] == False
        read_misses = self.l2_data_stats[miss_mask & read_mask]

        return int(read_misses["count"].sum())

    def l2_write_misses(self) -> int:
        miss_mask = self.l2_data_stats["status"] == "MISS"
        write_mask = self.l2_data_stats["is_write"] == True
        write_misses = self.l2_data_stats[miss_mask & write_mask]

        return int(write_misses["count"].sum())
