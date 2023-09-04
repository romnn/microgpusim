import pandas as pd
from pathlib import Path
from os import PathLike

from gpucachesim.benchmarks import GPUConfig, BenchConfig
import gpucachesim.stats.common as common


def access_is_write(access_type: str) -> bool:
    match access_type.upper():
        case "GLOBAL_ACC_R" | "LOCAL_ACC_R" | "CONST_ACC_R" | "TEXTURE_ACC_R" | "INST_ACC_R" | "L1_WR_ALLOC_R" | "L2_WR_ALLOC_R":
            return False
        case "GLOBAL_ACC_W" | "LOCAL_ACC_W" | "L1_WRBK_ACC" | "L2_WRBK_ACC":
            return True
        case other:
            raise ValueError(f"bad access type: {other}")


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
        self.load_bench_config(bench_config)

    def load_bench_config(self, bench_config: BenchConfig) -> None:
        self.bench_config = bench_config
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

    def cycles(self) -> int:
        # return self.metrics["gpu_tot_sim_cycle"].sum()
        return self.sim_df["cycles"].sum()

    def instructions(self) -> int:
        return self.sim_df["instructions"].sum()

    def dram_reads(self) -> int:
        return int(self.dram_df["reads"].sum())

    def dram_writes(self) -> int:
        return int(self.dram_df["writes"].sum())

    def dram_accesses(self) -> int:
        # total = int(self.df["total_dram_reads"].sum())
        # total += int(self.df["total_dram_writes"].sum())
        # return total
        total = int(self.dram_df["reads"].sum())
        total += int(self.dram_df["writes"].sum())
        return total

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
