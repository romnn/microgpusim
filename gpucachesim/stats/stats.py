import pandas as pd
import numpy as np
import cxxfilt
import json
from os import PathLike
from pathlib import Path
from typing import Sequence
from pprint import pprint

from gpucachesim.benchmarks import (
    WRITE_ACCESS_KINDS,
    GPUConfig,
    BenchConfig,
    SimulateConfig,
    SimulateTargetConfig,
    INDEX_COLS,
    SPECIAL_DTYPES,
    ACCESS_KINDS,
    ACCESS_STATUSES,
)
import gpucachesim.stats.common as common


def parse_cache_stats(path: PathLike):
    df = pd.read_csv(
        path,
        header=0,
    )
    return df


def fix_dtypes(df):
    dtypes = {
        **{col: "float64" for col in df.columns},
        **SPECIAL_DTYPES,
    }
    dtypes = {col: dtype for col, dtype in dtypes.items() if col in df}

    def map_dtype(dtype):
        if dtype == "object":
            return "string"
        return dtype

    dtypes = {col: map_dtype(dtype) for col, dtype in dtypes.items()}
    return df.astype(dtypes)


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

    def load_converted_stats(self, all=False) -> None:
        # exec_time_sec_release_dfs = []
        sim_dfs = []
        accesses_dfs = []
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

            if all:
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

            if all:
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

            if all:
                l1_inst_stats_df = parse_cache_stats(self.path / f"stats.cache.l1i.{r}.csv")
                l1_inst_stats_df["run"] = r
                l1_inst_stats_dfs.append(l1_inst_stats_df)

            if all:
                l1_tex_stats_df = parse_cache_stats(self.path / f"stats.cache.l1t.{r}.csv")
                l1_tex_stats_df["run"] = r
                l1_tex_stats_dfs.append(l1_tex_stats_df)

            l1_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l1d.{r}.csv")
            l1_data_stats_df["run"] = r
            l1_data_stats_dfs.append(l1_data_stats_df)

            if all:
                l1_const_stats_df = parse_cache_stats(self.path / f"stats.cache.l1c.{r}.csv")
                l1_const_stats_df["run"] = r
                l1_const_stats_dfs.append(l1_const_stats_df)

            l2_data_stats_df = parse_cache_stats(self.path / f"stats.cache.l2d.{r}.csv")
            l2_data_stats_df["run"] = r
            l2_data_stats_dfs.append(l2_data_stats_df)

        self.sim_df = fix_dtypes(pd.concat(sim_dfs))
        self.dram_banks_df = fix_dtypes(pd.concat(dram_banks_dfs))

        if len(accesses_dfs) > 0:
            self.accesses_df = fix_dtypes(pd.concat(accesses_dfs))

        if len(instructions_dfs) > 0:
            self.instructions_df = fix_dtypes(pd.concat(instructions_dfs))

        if len(l1_inst_stats_dfs) > 0:
            self.l1_inst_stats_df = fix_dtypes(pd.concat(l1_inst_stats_dfs))

        if len(l1_tex_stats_dfs) > 0:
            self.l1_tex_stats_df = fix_dtypes(pd.concat(l1_tex_stats_dfs))

        if len(l1_const_stats_dfs) > 0:
            self.l1_const_stats_df = fix_dtypes(pd.concat(l1_const_stats_dfs))

        self.l1_data_stats_df = fix_dtypes(pd.concat(l1_data_stats_dfs))
        self.l2_data_stats_df = fix_dtypes(pd.concat(l2_data_stats_dfs))

        if False:
            print(self.l1_data_stats_df)
            print(self.l1_data_stats_df.columns)
            print(self.l1_data_stats_df.index)

            # product = pd.DataFrame(
            #     {
            #         "access_status": ACCESS_STATUSES,
            #         "access_kind": ACCESS_KINDS,
            #     }
            # )
            product = pd.MultiIndex.from_product(
                [ACCESS_STATUSES, ACCESS_KINDS], names=["access_status", "access_kind"]
            )
            product = product.to_frame().reset_index(drop=True)
            product["is_write"] = product["access_kind"].isin(WRITE_ACCESS_KINDS)

            def test(df):
                print("aggregate")
                print(df)
                #     print(df.index)
                #     print(df.columns)
                return df

            self.l1_data_stats_df = self.l1_data_stats_df.groupby(INDEX_COLS).apply(test)

            raise ValueError("todo")
        # # no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        # global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        # hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        # miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])

        # for df in [
        #     self.sim_df,
        #     self.accesses_df,
        #     self.dram_banks_df,
        #     self.instructions_df,
        #     self.l1_inst_stats_df,
        #     self.l1_tex_stats_df,
        #     self.l1_data_stats_df,
        #     self.l1_const_stats_df,
        #     self.l2_data_stats_df,
        # ]:
        #     dtypes = {
        #         **{col: "float64" for col in df.columns},
        #         **SPECIAL_DTYPES,
        #     }
        #     dtypes = {col: dtype for col, dtype in dtypes.items() if col in df}
        #     # df = df.astype(dtypes)
        #     df.astype(dtypes, copy=False)

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
        # no_kernel = df["kernel_name"].isna() & df["kernel_name_mangled"].isna()
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])

        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]

        # print("")
        # print("HIT KERNEL NAME MANGLED")
        # print(hits.index)
        #
        # print("ACCESS KERNEL NAME MANGLED")
        # print(accesses.index)

        grouped = hits.groupby(INDEX_COLS, dropna=False, sort=False)
        hits = grouped["num_accesses"].sum()
        grouped = accesses.groupby(INDEX_COLS, dropna=False, sort=False)
        accesses = grouped["num_accesses"].sum()

        # for index_col in self.result_df.index.names:
        #     print(
        #         index_col,
        #         self.result_df.index.get_level_values(index_col).dtype,
        #         self.result_df.index.get_level_values(index_col),
        #     )

        # self.result_df.index._sort_levels_monotonic(raise_if_incomparable=True)
        # hits._sort_levels_monotonic(raise_if_incomparable=True)
        # accesses._sort_levels_monotonic(raise_if_incomparable=True)

        # print("")
        # print("HIT KERNEL NAME MANGLED")
        # print(hits.index)
        #
        # print("ACCESS KERNEL NAME MANGLED")
        # print(accesses.index)

        # print("==== hits")
        # print(hits)
        # print(hits.index)
        # print(hits.dtype)
        # print(hits.to_numpy())

        # print("==== accesses")
        # print(accesses)
        # print(accesses.index)
        # print(accesses.dtype)
        # print(accesses.to_numpy())

        # print("==== result")
        # res = (hits / accesses).fillna(0.0)
        # print(res)
        # print(res.dtype)
        # print(res.index)
        # for index_col in res.index.names:
        #     print(index_col, res.index.get_level_values(index_col).dtype)
        #
        # print("==== have")
        # print(self.result_df.index)
        # for index_col in self.result_df.index.names:
        #     print(index_col, self.result_df.index.get_level_values(index_col).dtype)

        # print("")

        # print("\nHAVE:")
        # print(self.result_df.index)
        # for row in self.result_df.index:
        #     print(row)
        # print(self.result_df.index.to_frame().reset_index(drop=True).drop_duplicates())
        #
        # print("\nNEW:")
        # print("hits {} accesses {}".format(len(hits), len(accesses)))
        # assert hits.index == accesses.index
        l1_hit_rate = (hits / accesses).fillna(0.0)
        # if len(hits) != 0 and len(accesses) != 0:
        #     l1_hit_rate = (hits / accesses).fillna(0.0)
        # else:
        #     l1_hit_rate = 0.0
        # assert self.result_df.index.names == l1_hit_rate.index.names
        # print(l1_hit_rate.index)
        # for row in l1_hit_rate.index:
        #     print(row)
        # l1_hit_rate.index = self.result_df.index
        # print(l1_hit_rate.index.to_frame().reset_index(drop=True).drop_duplicates())

        self.result_df["l1_hit_rate"] = l1_hit_rate

    def _compute_l1_global_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]

        grouped = hits.groupby(INDEX_COLS, dropna=False, sort=False)
        hits = grouped["num_accesses"].sum()
        grouped = accesses.groupby(INDEX_COLS, dropna=False, sort=False)
        accesses = grouped["num_accesses"].sum()

        self.result_df["l1_global_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_local_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["LOCAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask | miss_mask)]

        grouped = hits.groupby(INDEX_COLS, dropna=False, sort=False)
        hits = grouped["num_accesses"].sum()
        grouped = accesses.groupby(INDEX_COLS, dropna=False, sort=False)
        accesses = grouped["num_accesses"].sum()

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
