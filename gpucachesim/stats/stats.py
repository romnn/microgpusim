import pandas as pd
import json
from os import PathLike
from pathlib import Path

from gpucachesim.benchmarks import GPUConfig, BenchConfig, SimulateConfig
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
    bench_config: SimulateConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig) -> None:
        self.bench_config = bench_config["simulate"]
        self.path = Path(self.bench_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = int(self.bench_config["repetitions"])
        self.load_converted_stats()

    def load_converted_stats(self) -> None:
        exec_time_sec_release_dfs = []
        sim_dfs = []
        accesses_dfs = []
        dram_dfs = []
        dram_banks_dfs = []
        instructions_dfs = []
        l1_inst_stats_dfs = []
        l1_tex_stats_dfs = []
        l1_data_stats_dfs = []
        l1_const_stats_dfs = []
        l2_data_stats_dfs = []

        for r in range(self.repetitions):
            with open(self.path / f"exec_time.release.{r}.json", "rb") as f:
                # convert millis to seconds
                exec_time = float(json.load(f)) * 1e-3
                # exec_time_sec_release_dfs.append(pd.DataFrame.from_records([dict(run=r, exec_time=exec_time)]))
                exec_time_sec_release_dfs.append(pd.DataFrame.from_records([dict(exec_time=exec_time)]))

            sim_df = pd.read_csv(
                self.path / f"stats.sim.{r}.csv",
                header=0,
            )
            # sim_df["run"] = r
            sim_dfs.append(sim_df)

            accesses_df = pd.read_csv(
                self.path / f"stats.accesses.{r}.csv",
                header=None,
                names=["access", "count"],
            )
            # accesses_df["run"] = r
            accesses_dfs.append(accesses_df)

            dram_df = pd.read_csv(
                self.path / f"stats.dram.{r}.csv",
                header=0,
            )
            # dram_df["run"] = r
            dram_dfs.append(dram_df)

            dram_banks_df = pd.read_csv(
                self.path / f"stats.dram.banks.{r}.csv",
                header=0,
            )
            dram_banks_dfs.append(dram_banks_df)

            instructions_df = pd.read_csv(
                self.path / f"stats.instructions.{r}.csv",
                header=None,
                names=["memory_space", "write", "count"],
            )
            instructions_dfs.append(instructions_df)

            l1_inst_stats_df = parse_cache_stats(self.path / f"stats.cache.l1i.{r}.csv")
            l1_inst_stats_dfs.append(l1_inst_stats_df)
            l1_tex_stats_df = parse_cache_stats(self.path / f"stats.cache.l1t.{r}.csv")
            l1_tex_stats_dfs.append(l1_tex_stats_df)
            l1_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l1d.{r}.csv")
            l1_data_stats_dfs.append(l1_data_stats_df)
            l1_const_stats_df = parse_cache_stats(self.path / f"stats.cache.l1c.{r}.csv")
            l1_const_stats_dfs.append(l1_const_stats_df)
            l2_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l2d.{r}.csv")
            l2_data_stats_dfs.append(l2_data_stats_df)

        exec_time_sec_release = pd.concat(exec_time_sec_release_dfs)
        # print(exec_time_sec_release)
        self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=None)
        # print(self.exec_time_sec_release)
        # self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=INDEX_COLS)

        sim_df = pd.concat(sim_dfs)
        # print(sim_df)
        self.sim_df = common.compute_df_statistics(sim_df, group_by=None)
        # print(self.sim_df)

        accesses_df = pd.concat(accesses_dfs)
        # print(accesses_df)
        self.accesses_df = common.compute_df_statistics(accesses_df, group_by=["access"])
        # print(self.accesses_df)

        dram_df = pd.concat(dram_dfs)
        # print(dram_df)
        self.dram_df = common.compute_df_statistics(dram_df, group_by=["chip_id", "bank_id"])
        # print(self.dram_df)

        dram_banks_df = pd.concat(dram_banks_dfs)
        # print(dram_banks_df)
        self.dram_banks_df = common.compute_df_statistics(dram_banks_df, group_by=["core_id", "chip_id", "bank_id"])
        # print(self.dram_banks_df)

        instructions_df = pd.concat(instructions_dfs)
        # print(instructions_df)
        self.instructions_df = common.compute_df_statistics(instructions_df, group_by=["memory_space", "write"])
        # print(self.instructions_df)

        def load_cache_stats(dfs):
            df = pd.concat(dfs)
            # print(df)
            df = common.compute_df_statistics(df, group_by=["cache_id", "access_type", "status", "is_write"])
            # print(df)
            return df

        self.l1_inst_stats_df = load_cache_stats(l1_inst_stats_dfs)
        self.l1_tex_stats_df = load_cache_stats(l1_tex_stats_dfs)
        self.l1_data_stats_df = load_cache_stats(l1_data_stats_dfs)
        self.l1_const_stats_df = load_cache_stats(l1_const_stats_dfs)
        self.l2_data_stats_df = load_cache_stats(l2_data_stats_dfs)

    def exec_time_sec(self) -> float:
        return self.exec_time_sec_release["exec_time_mean"].sum()

    def cycles(self) -> float:
        return self.sim_df["cycles_mean"].sum()

    def warp_instructions(self) -> float:
        return 0

    def instructions(self) -> float:
        return self.sim_df["instructions_mean"].sum()

    def num_blocks(self) -> float:
        return self.sim_df["num_blocks_mean"].sum()

    def dram_reads(self) -> float:
        return self.dram_df["reads_mean"].sum()

    def dram_writes(self) -> float:
        return self.dram_df["writes_mean"].sum()

    def dram_accesses(self) -> float:
        return self.dram_df[["reads_mean", "writes_mean"]].sum().sum()

    def l2_reads(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        hit_mask = df["status"] == "HIT"
        miss_mask = df["status"] == "MISS"
        read_mask = df["is_write"] == False
        reads = df[(hit_mask ^ miss_mask) & read_mask]
        return reads["count_mean"].sum()

    def l2_writes(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        hit_mask = df["status"] == "HIT"
        miss_mask = df["status"] == "MISS"
        write_mask = df["is_write"] == True
        reads = df[(hit_mask ^ miss_mask) & write_mask]
        return reads["count_mean"].sum()

    def l2_accesses(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        hit_mask = df["status"] == "HIT"
        miss_mask = df["status"] == "MISS"
        accesses = df[hit_mask ^ miss_mask]
        return accesses["count_mean"].sum()

    def l2_read_hits(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        hit_mask = df["status"] == "HIT"
        read_mask = df["is_write"] == False
        read_hits = df[hit_mask & read_mask]
        return read_hits["count_mean"].sum()

    def l2_write_hits(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        hit_mask = df["status"] == "HIT"
        write_mask = df["is_write"] == True
        write_hits = df[hit_mask & write_mask]
        return write_hits["count_mean"].sum()

    def l2_read_misses(self) -> float:
        df = self.l2_data_stats_df.reset_index()
        miss_mask = df["status"] == "MISS"
        read_mask = df["is_write"] == False
        read_misses = df[miss_mask & read_mask]
        return read_misses["count_mean"].sum()

    def l2_write_misses(self) -> int:
        df = self.l2_data_stats_df.reset_index()
        miss_mask = df["status"] == "MISS"
        write_mask = df["is_write"] == True
        write_misses = df[miss_mask & write_mask]
        return write_misses["count_mean"].sum()
