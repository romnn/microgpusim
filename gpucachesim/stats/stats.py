import pandas as pd
import numpy as np
import cxxfilt
import json
from os import PathLike
from pathlib import Path
from typing import Sequence
from pprint import pprint

from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    SimulateConfig,
    SimulateTargetConfig,
    INDEX_COLS,
)
import gpucachesim.stats.common as common


def parse_cache_stats(path: PathLike):
    df = pd.read_csv(
        path,
        header=0,
    )
    return df


class Stats(common.Stats):
    bench_config: BenchConfig[SimulateTargetConfig]
    target_config: SimulateConfig

    def __init__(self, config: GPUConfig, bench_config: BenchConfig[SimulateTargetConfig]) -> None:
        self.bench_config = bench_config
        self.target_config = self.bench_config["target_config"].value
        self.path = Path(self.target_config["stats_dir"])
        self.use_duration = False
        self.config = config
        self.repetitions = int(self.bench_config["common"]["repetitions"])
        self.load_converted_stats()
        self.compute_result_df()

    def load_converted_stats(self) -> None:
        # exec_time_sec_release_dfs = []
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
            sim_df = pd.read_csv(
                self.path / f"stats.sim.{r}.csv",
                header=0,
            )
            sim_df["run"] = r
            sim_dfs.append(sim_df)

            accesses_df = pd.read_csv(
                self.path / f"stats.accesses.{r}.csv",
                header=0,
            )
            accesses_df["run"] = r
            accesses_dfs.append(accesses_df)

            dram_banks_df = pd.read_csv(
                self.path / f"stats.dram.banks.{r}.csv",
                header=0,
            )
            dram_banks_df["run"] = r
            dram_banks_dfs.append(dram_banks_df)

            try:
                instructions_df = pd.read_csv(
                    self.path / f"stats.instructions.{r}.csv",
                    header=0,
                )
            except pd.errors.EmptyDataError:
                print(self.path / f"stats.instructions.{r}.csv")
                raise
            instructions_df["run"] = r
            instructions_dfs.append(instructions_df)

            l1_inst_stats_df = parse_cache_stats(self.path / f"stats.cache.l1i.{r}.csv")
            l1_inst_stats_df["run"] = r
            l1_inst_stats_dfs.append(l1_inst_stats_df)

            l1_tex_stats_df = parse_cache_stats(self.path / f"stats.cache.l1t.{r}.csv")
            l1_tex_stats_df["run"] = r
            l1_tex_stats_dfs.append(l1_tex_stats_df)

            l1_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l1d.{r}.csv")
            l1_data_stats_df["run"] = r
            l1_data_stats_dfs.append(l1_data_stats_df)

            l1_const_stats_df = parse_cache_stats(self.path / f"stats.cache.l1c.{r}.csv")
            l1_const_stats_df["run"] = r
            l1_const_stats_dfs.append(l1_const_stats_df)

            l2_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l2d.{r}.csv")
            l2_data_stats_df["run"] = r
            l2_data_stats_dfs.append(l2_data_stats_df)

        self.sim_df = pd.concat(sim_dfs)
        self.accesses_df = pd.concat(accesses_dfs)
        self.dram_banks_df = pd.concat(dram_banks_dfs)
        self.instructions_df = pd.concat(instructions_dfs)
        self.l1_inst_stats_df = pd.concat(l1_inst_stats_dfs)
        self.l1_tex_stats_df = pd.concat(l1_tex_stats_dfs)
        self.l1_data_stats_df = pd.concat(l1_data_stats_dfs)
        self.l1_const_stats_df = pd.concat(l1_const_stats_dfs)
        self.l2_data_stats_df = pd.concat(l2_data_stats_dfs)

    def compute_result_df(self):
        self.result_df = pd.DataFrame()
        self._compute_cycles()
        self._compute_instructions()
        self._compute_num_blocks()
        self._compute_total_cores()
        self._compute_mean_blocks_per_sm()
        self._compute_warp_instructions()
        self._compute_exec_time_sec()
        self._compute_is_release_build()

        # DRAM
        self._compute_dram_reads()
        self._compute_dram_writes()
        self._compute_dram_accesses()

        # L2 accesses
        self._compute_l2_reads()
        self._compute_l2_writes()
        self._compute_l2_accesses()
        self._compute_l2_read_hits()
        self._compute_l2_write_hits()
        self._compute_l2_read_misses()
        self._compute_l2_write_misses()
        self._compute_l2_hits()
        self._compute_l2_misses()

        # L2 rates
        self._compute_l2_read_hit_rate()
        self._compute_l2_write_hit_rate()
        self._compute_l2_read_miss_rate()
        self._compute_l2_write_miss_rate()
        self._compute_l2_hit_rate()
        self._compute_l2_miss_rate()

        # L1 accesses
        self._compute_l1_accesses()
        self._compute_l1_reads()
        self._compute_l1_writes()
        self._compute_l1_hits()
        self._compute_l1_misses()

        # L1 rates
        self._compute_l1_hit_rate()
        self._compute_l1_global_hit_rate()
        self._compute_l1_local_hit_rate()
        self._compute_l1_miss_rate()

        # fix the index
        self.result_df = self.result_df.reset_index()
        self.result_df["stream_id"] = np.nan
        self.result_df["context_id"] = np.nan
        self.result_df["device"] = np.nan
        # self.result_df["kernel_name_mangled"] = self.result_df["kernel_name_mangled"].bfill()
        self.result_df["kernel_function_signature"] = self.result_df["kernel_name_mangled"].apply(
            lambda name: np.nan if pd.isnull(name) else cxxfilt.demangle(name)
        )
        self.result_df["kernel_name"] = self.result_df["kernel_function_signature"].apply(
            lambda sig: np.nan if pd.isnull(sig) else common.function_name_from_signature(sig)
        )

    def _compute_l2_read_hit_rate(self):
        self.result_df["l2_read_hit_rate"] = self.result_df["l2_read_hits"] / self.result_df["l2_reads"]

    def _compute_l2_read_miss_rate(self):
        self.result_df["l2_read_miss_rate"] = 1.0 - self.result_df["l2_read_hit_rate"].fillna(0.0)

    def _compute_l2_write_hit_rate(self):
        self.result_df["l2_write_hit_rate"] = self.result_df["l2_write_hits"] / self.result_df["l2_writes"]

    def _compute_l2_write_miss_rate(self):
        self.result_df["l2_write_miss_rate"] = 1.0 - self.result_df["l2_write_hit_rate"].fillna(0.0)

    def _compute_l2_hit_rate(self):
        # print(self.result_df[["l2_hits", "l2_reads", "l2_writes", "l2_accesses"]].fillna(0.0))
        hits = self.result_df["l2_hits"].fillna(0.0)
        accesses = self.result_df["l2_accesses"].fillna(0.0)
        # print((hits / accesses).fillna(0.0))
        self.result_df["l2_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l2_miss_rate(self):
        self.result_df["l2_miss_rate"] = np.nan

    def _compute_l1_reads(self):
        df = self.l1_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        read_mask = df["is_write"] == False
        reads = df[(hit_mask | miss_mask) & read_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_reads"] = grouped["num_accesses"].sum()

    def _compute_l1_writes(self):
        df = self.l1_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_W"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        write_mask = df["is_write"] == True
        reads = df[(hit_mask | miss_mask) & write_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_writes"] = grouped["num_accesses"].sum()

    def _compute_l1_accesses(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        accesses = df[(hit_mask | miss_mask) & (global_read)]
        grouped = accesses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_accesses"] = grouped["num_accesses"].sum()

    def _compute_l1_hits(self):
        df = self.l1_data_stats_df
        hit_mask = df["access_status"] == "HIT"
        hits = df[hit_mask]
        grouped = hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_hits"] = grouped["num_accesses"].sum()

    def _compute_l1_misses(self):
        df = self.l1_data_stats_df
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        misses = df[miss_mask]
        grouped = misses.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["l1_misses"] = grouped["num_accesses"].sum()

    def _compute_l1_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()

        self.result_df["l1_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_global_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()

        self.result_df["l1_global_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_local_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["LOCAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False, sort=False)["num_accesses"].sum()

        self.result_df["l1_local_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_miss_rate(self):
        self.result_df["l1_miss_rate"] = 1.0 - self.result_df["l1_hit_rate"].fillna(0.0)

    def _compute_is_release_build(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["is_release_build"] = grouped["is_release_build"].first()

    def _compute_exec_time_sec(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["exec_time_sec"] = grouped["elapsed_millis"].mean()
        self.result_df["exec_time_sec"] /= 1000.0

    def _compute_cycles(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["cycles"] = grouped["cycles"].mean()

    def _compute_instructions(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["instructions"] = grouped["instructions"].mean()

    def _compute_warp_instructions(self):
        # do not have that yet
        self.result_df["warp_inst"] = 0.0

    def _compute_num_blocks(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["num_blocks"] = grouped["num_blocks"].mean()

    def _compute_total_cores(self):
        self.result_df["cores_per_cluster"] = self.cores_per_cluster()
        self.result_df["num_clusters"] = self.num_clusters()
        self.result_df["total_cores"] = self.total_cores()

    def _compute_mean_blocks_per_sm(self):
        blocks_per_sm = self.result_df["num_blocks"] / self.result_df["total_cores"]
        self.result_df["mean_blocks_per_sm"] = blocks_per_sm

    def _compute_dram_reads(self):
        df = self.dram_banks_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        # is_read = df["is_write"] == False
        reads = df[is_global_read & ~no_kernel]
        grouped = reads.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["dram_reads"] = grouped["num_accesses"].sum()

    def _compute_dram_writes(self):
        df = self.dram_banks_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global_write = df["access_kind"].isin(["GLOBAL_ACC_W"])
        # is_write = df["is_write"] == True
        reads = df[is_global_write & ~no_kernel]
        grouped = reads.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["dram_writes"] = grouped["num_accesses"].sum()

    def _compute_dram_accesses(self):
        df = self.dram_banks_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W", "GLOBAL_ACC_R"])
        reads = df[is_global & ~no_kernel]
        grouped = reads.groupby(INDEX_COLS, dropna=False, sort=False)
        self.result_df["dram_accesses"] = grouped["num_accesses"].sum().sum()

    def _compute_l2_reads(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_R"])
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_miss = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        is_read = df["is_write"] == False
        reads = df[(is_hit | is_miss) & is_read & is_global & ~no_kernel]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_reads"] = grouped["num_accesses"].sum()

    def _compute_l2_writes(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W"])
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_miss = df["access_status"].isin(["SECTOR_MISS", "MISS"])
        is_write = df["is_write"] == True
        writes = df[(is_hit | is_miss) & is_write & is_global & ~no_kernel]
        grouped = writes.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_writes"] = grouped["num_accesses"].sum()

    def _compute_l2_accesses(self):
        df = self.l2_data_stats_df
        # l2 accesses are only read in nvprof
        # global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W", "GLOBAL_ACC_R"])
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_miss = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        accesses = df[(is_hit | is_miss) & is_global & ~no_kernel]
        # print(accesses)
        grouped = accesses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_accesses"] = grouped["num_accesses"].sum()

    def _compute_l2_read_hits(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_R"])
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_read = df["is_write"] == False
        read_hits = df[is_hit & is_read & is_global & ~no_kernel]
        grouped = read_hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_read_hits"] = grouped["num_accesses"].sum()

    def _compute_l2_write_hits(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W"])
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_write = df["is_write"] == True
        write_hits = df[is_hit & is_write & is_global & ~no_kernel]
        grouped = write_hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_write_hits"] = grouped["num_accesses"].sum()

    def _compute_l2_read_misses(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_R"])
        is_miss = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        is_read = df["is_write"] == False
        read_misses = df[is_miss & is_read & is_global & ~no_kernel]
        grouped = read_misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_read_misses"] = grouped["num_accesses"].sum()

    def _compute_l2_write_misses(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W"])
        is_miss = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        is_write = df["is_write"] == True
        write_misses = df[is_miss & is_write & is_global & ~no_kernel]
        grouped = write_misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_write_misses"] = grouped["num_accesses"].sum()

    def _compute_l2_hits(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_hit = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W", "GLOBAL_ACC_R"])
        hits = df[is_hit & is_global & ~no_kernel]
        grouped = hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_hits"] = grouped["num_accesses"].sum()

    def _compute_l2_misses(self):
        df = self.l2_data_stats_df
        no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        is_miss = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        is_global = df["access_kind"].isin(["GLOBAL_ACC_W", "GLOBAL_ACC_R"])
        misses = df[is_miss & is_global & ~no_kernel]
        grouped = misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_misses"] = grouped["num_accesses"].sum()


class ExecDrivenStats(Stats):
    bench_config: BenchConfig[SimulateTargetConfig]
    target_config: SimulateConfig
