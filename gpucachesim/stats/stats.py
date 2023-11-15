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


# def access_is_write(access_type: str) -> bool:
#     if access_type.upper() in READ_ACCESS_KINDS:
#         return False
#     if access_type.upper() in WRITE_ACCESS_KINDS:
#         return True
#     raise ValueError(f"bad access type: {access_type}")


def parse_cache_stats(path: PathLike):
    df = pd.read_csv(
        path,
        header=0,
        # header=None,
        # names=["cache_id", "access_type", "status", "count"],
    )
    # print(df)
    # df["is_write"] = df["access_kind"].apply(access_is_write)
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

        # # add the input configs
        # # print(self.bench_config["values"])
        # values = pd.DataFrame.from_records([self.bench_config["values"]])
        # values.columns = [self.bench_config["name"] + "_" + c for c in values.columns]
        # values["simulator"] = "gpucachesim"
        # values = self.result_df.merge(values, how="cross")
        # # values = pd.concat([values, self.result_df])  # , axis="columns")
        # # self.result_df.join()
        # print(values.T)

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
            # with open(self.path / f"exec_time.release.{r}.json", "rb") as f:
            #     # convert millis to seconds
            #     exec_time = float(json.load(f)) * 1e-3
            #     # exec_time_sec_release_dfs.append(pd.DataFrame.from_records([dict(run=r, exec_time=exec_time)]))
            #     exec_time_sec_release_dfs.append(pd.DataFrame.from_records([dict(run=r, exec_time=exec_time)]))

            sim_df = pd.read_csv(
                self.path / f"stats.sim.{r}.csv",
                header=0,
            )
            sim_df["run"] = r
            sim_dfs.append(sim_df)

            accesses_df = pd.read_csv(
                self.path / f"stats.accesses.{r}.csv",
                header=0,
                # header=None,
                # names=["access", "count"],
            )
            accesses_df["run"] = r
            accesses_dfs.append(accesses_df)

            # dram_df = pd.read_csv(
            #     self.path / f"stats.dram.{r}.csv",
            #     header=0,
            # )
            # dram_df["run"] = r
            # dram_dfs.append(dram_df)
            # print(dram_dfs)

            dram_banks_df = pd.read_csv(
                self.path / f"stats.dram.banks.{r}.csv",
                header=0,
            )
            dram_banks_df["run"] = r
            dram_banks_dfs.append(dram_banks_df)
            # print(dram_banks_dfs)

            try:
                instructions_df = pd.read_csv(
                    self.path / f"stats.instructions.{r}.csv",
                    header=0,
                    # header=None,
                    # names=["memory_space", "write", "count"],
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

        # self.exec_time_sec_release = pd.concat(exec_time_sec_release_dfs)
        # print(exec_time_sec_release)
        # self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=None)
        # self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=INDEX_COLS)
        # print(self.exec_time_sec_release)

        self.sim_df = pd.concat(sim_dfs)
        # print(sim_df)
        # self.sim_df = common.compute_df_statistics(sim_df, group_by=None)
        # print(self.sim_df)

        self.accesses_df = pd.concat(accesses_dfs)
        # print(accesses_df)
        # self.accesses_df = common.compute_df_statistics(accesses_df, group_by=["access"])
        # print(self.accesses_df)

        # self.dram_df = pd.concat(dram_dfs)
        # print(dram_df)
        # self.dram_df = common.compute_df_statistics(dram_df, group_by=["chip_id", "bank_id"])
        # print(self.dram_df)

        self.dram_banks_df = pd.concat(dram_banks_dfs)
        # print(dram_banks_df)
        # self.dram_banks_df = common.compute_df_statistics(dram_banks_df, group_by=["core_id", "chip_id", "bank_id"])
        # print(self.dram_banks_df)

        self.instructions_df = pd.concat(instructions_dfs)
        # print(instructions_df)
        # self.instructions_df = common.compute_df_statistics(instructions_df, group_by=["memory_space", "write"])
        # print(self.instructions_df)

        # def load_cache_stats(dfs):
        #     df = pd.concat(dfs)
        #     # print(df)
        #     # df = common.compute_df_statistics(df, group_by=["cache_id", "access_type", "status", "is_write"])
        #     # print(df)
        #     return df

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
        self.result_df["kernel_name_mangled"] = self.result_df["kernel_name_mangled"].bfill()
        self.result_df["kernel_function_signature"] = self.result_df["kernel_name_mangled"].apply(
            lambda name: np.nan if pd.isnull(name) else cxxfilt.demangle(name)
        )
        self.result_df["kernel_name"] = self.result_df["kernel_function_signature"].apply(
            lambda sig: np.nan if pd.isnull(sig) else common.function_name_from_signature(sig)
        )

    def _compute_l2_read_hit_rate(self):
        # df = self.l2_data_stats_df
        # hit_mask = df["access_status"] == "HIT"
        # miss_mask = df["access_status"] == "MISS"
        # write_mask = df["is_write"] == False
        # write_hits = df[hit_mask & write_mask]
        # grouped = write_hits.groupby(INDEX_COLS, dropna=False)
        #
        # total_writes = df[(hit_mask ^ miss_mask) & write_mask].groupby(INDEX_COLS, dropna=False)
        # self.result_df["l2_read_hit_rate"] = grouped["num_accesses"].sum() / total_writes["num_accesses"].sum()
        self.result_df["l2_read_hit_rate"] = self.result_df["l2_read_hits"] / self.result_df["l2_reads"]

    def _compute_l2_read_miss_rate(self):
        self.result_df["l2_read_miss_rate"] = 1.0 - self.result_df["l2_read_hit_rate"]

    def _compute_l2_write_hit_rate(self):
        # df = self.l2_data_stats_df
        # hit_mask = df["access_status"] == "HIT"
        # miss_mask = df["access_status"] == "MISS"
        # write_mask = df["is_write"] == True
        # write_hits = df[hit_mask & write_mask]
        # grouped = write_hits.groupby(INDEX_COLS, dropna=False)
        #
        # total_writes = df[(hit_mask ^ miss_mask) & write_mask].groupby(INDEX_COLS, dropna=False)
        # self.result_df["l2_write_hit_rate"] = grouped["num_accesses"].sum() / total_writes["num_accesses"].sum()
        self.result_df["l2_write_hit_rate"] = self.result_df["l2_write_hits"] / self.result_df["l2_writes"]

        # "100*float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"])/"+\
        #     "float(sim[\"\s+L2_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])",
        # grouped = self.df.groupby(INDEX_COLS, dropna=False)
        # self.result_df["l2_write_hit_rate"] = grouped["l2_tex_write_hit_rate"].mean()
        # self.result_df["l2_write_hit_rate"] /= 100.0
        # self.result_df["is_release_build"] = grouped["is_release_build"].first()

    def _compute_l2_write_miss_rate(self):
        self.result_df["l2_write_miss_rate"] = 1.0 - self.result_df["l2_write_hit_rate"]

    def _compute_l2_hit_rate(self):
        # print(self.result_df[["l2_accesses", "l2_hits"]].T)
        hits = self.result_df["l2_hits"].fillna(0.0)
        accesses = self.result_df["l2_accesses"].fillna(0.0)
        # print((hits / accesses).T)
        self.result_df["l2_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l2_miss_rate(self):
        self.result_df["l2_miss_rate"] = np.nan

    def _compute_l1_reads(self):
        df = self.l1_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        read_mask = df["is_write"] == False
        reads = df[(hit_mask ^ miss_mask) & read_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_reads"] = grouped["num_accesses"].sum()

    def _compute_l1_writes(self):
        df = self.l1_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_W"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        write_mask = df["is_write"] == True
        reads = df[(hit_mask ^ miss_mask) & write_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_writes"] = grouped["num_accesses"].sum()

    def _compute_l1_accesses(self):
        df = self.l1_data_stats_df
        # global_write = df["access_kind"] == "GLOBAL_ACC_W"
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT", "HIT_RESERVED"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        accesses = df[(hit_mask ^ miss_mask) & (global_read)]
        # print(accesses)
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
        # print(misses)
        grouped = misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l1_misses"] = grouped["num_accesses"].sum()

    def _compute_l1_hit_rate(self):
        # read_hits = self.result_df["l1_data_cache_GLOBAL_ACC_R_HIT"]
        # write_hits = self.result_df["l1_data_cache_GLOBAL_ACC_W_HIT"]
        # total_writes = self.result_df["l1_data_cache_global_write_total"]
        # total_reads = self.result_df["l1_data_cache_global_read_total"]
        # self.result_df["l1_hit_rate"] = (read_hits + write_hits) / (total_writes + total_reads)

        # print(self.result_df[["l1_accesses", "l1_hits"]].T)
        # hits = self.result_df["l1_hits"].fillna(0.0)
        # accesses = self.result_df["l1_accesses"].fillna(0.0)
        # self.result_df["l1_hit_rate"] = (hits / accesses).fillna(0.0)

        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask ^ miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()

        self.result_df["l1_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_global_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask ^ miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()

        self.result_df["l1_global_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_local_hit_rate(self):
        df = self.l1_data_stats_df
        global_read = df["access_kind"].isin(["LOCAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        hits = df[global_read & hit_mask]
        accesses = df[global_read & (hit_mask ^ miss_mask)]
        hits = hits.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()
        accesses = accesses.groupby(INDEX_COLS, dropna=False)["num_accesses"].sum()

        self.result_df["l1_local_hit_rate"] = (hits / accesses).fillna(0.0)

    def _compute_l1_miss_rate(self):
        self.result_df["l1_miss_rate"] = 1.0 - self.result_df["l1_hit_rate"]

    def _compute_is_release_build(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["is_release_build"] = grouped["is_release_build"].first()

    def _compute_exec_time_sec(self):
        # exec_time: pd.DataFrame = self.sim_df.groupby(INDEX_COLS, dropna=False)["elapsed_millis"].mean()
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["exec_time_sec"] = grouped["elapsed_millis"].mean()
        self.result_df["exec_time_sec"] /= 1000.0
        # self.result_df[stat_cols("exec_time_sec")] = self.exec_time_sec_release[stat_cols("exec_time")]

    def _compute_cycles(self):
        # self.result_df["cycles"] = self.sim_df["cycles"]
        # print(INDEX_COLS)
        # print(self.sim_df.fillna(0.0)[INDEX_COLS])
        # print(self.sim_df.fillna(0.0).groupby(INDEX_COLS, dropna=False).sum())
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["cycles"] = grouped["cycles"].mean()
        # self.result_df[stat_cols("cycles")] = self.sim_df[stat_cols("cycles")]

    def _compute_instructions(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["instructions"] = grouped["instructions"].mean()
        # self.result_df[stat_cols("instructions")] = self.sim_df[stat_cols("instructions")]

    # def _num_warps_df(self):
    #     self.result_df[stat_cols("num_warps")] = self.result_df[stat_cols("num_blocks")]
    #     return self.num_blocks() * stats.WARP_SIZE
    #     pass

    def _compute_warp_instructions(self):
        # do not have that yet
        self.result_df["warp_inst"] = 0.0
        # self.result_df[stat_cols("warp_inst")] = 0.0

    def _compute_num_blocks(self):
        grouped = self.sim_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["num_blocks"] = grouped["num_blocks"].mean()
        # self.result_df[stat_cols("num_blocks")] = self.sim_df[stat_cols("num_blocks")]

    def _compute_total_cores(self):
        self.result_df["cores_per_cluster"] = self.cores_per_cluster()
        self.result_df["num_clusters"] = self.num_clusters()
        self.result_df["total_cores"] = self.total_cores()

    def _compute_mean_blocks_per_sm(self):
        blocks_per_sm = self.result_df["num_blocks"] / self.result_df["total_cores"]
        self.result_df["mean_blocks_per_sm"] = blocks_per_sm

    def _compute_dram_reads(self):
        # dram_reads = self.dram_df[stat_cols("reads")].sum(axis=0).values
        # self.result_df[stat_cols("dram_reads")] = dram_reads
        # print(self.dram_df.groupby(INDEX_COLS + ["dram)["reads"].mean())
        df = self.dram_banks_df
        # print(df)
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_R"])
        # hit_mask = df["access_status"].isin(["HIT"])
        # miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        # read_mask = df["is_write"] == False
        reads = df[global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)

        # grouped = self.dram_banks_df.groupby(INDEX_COLS + ["access_kind"], dropna=False)
        # GLOBAL_ACC_R
        # print(grouped.sum())
        self.result_df["dram_reads"] = grouped["num_accesses"].sum()

    def _compute_dram_writes(self):
        df = self.dram_banks_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_W"])
        # hit_mask = df["access_status"].isin(["HIT"])
        # miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        # read_mask = df["is_write"] == False
        reads = df[global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["dram_writes"] = grouped["num_accesses"].sum()

        # dram_writes = self.dram_df[stat_cols("writes")].sum(axis=0)
        # self.result_df[stat_cols("dram_writes")] = dram_writes.values
        # grouped = self.dram_banks_df.groupby(INDEX_COLS, dropna=False)
        # print(grouped.sum())
        # self.result_df["dram_writes"] = grouped["writes"].sum()

    def _compute_dram_accesses(self):
        # print(self.dram_df.groupby(INDEX_COLS, dropna=False)[["reads", "writes"]].sum().sum(axis=1))
        df = self.dram_banks_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_W", "GLOBAL_ACC_R"])
        # hit_mask = df["access_status"].isin(["HIT"])
        # miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        # read_mask = df["is_write"] == False
        reads = df[global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["dram_accesses"] = grouped["num_accesses"].sum().sum()

        # grouped = self.dram_banks_df.groupby(INDEX_COLS, dropna=False)
        # print(grouped.sum())
        # reads_and_writes = grouped[["reads", "writes"]].sum()
        # self.result_df["dram_accesses"] = reads_and_writes.sum(axis=1)

    def _compute_l2_reads(self):
        df = self.l2_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_R"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        read_mask = df["is_write"] == False
        reads = df[(hit_mask ^ miss_mask) & read_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        # print(grouped["num_accesses"].sum())
        self.result_df["l2_reads"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # hit_mask = df["status"] == "HIT"
        # miss_mask = df["status"] == "MISS"
        # read_mask = df["is_write"] == False
        # reads = df[(hit_mask ^ miss_mask) & read_mask]
        # reads = reads[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_reads")] = reads.values

    def _compute_l2_writes(self):
        df = self.l2_data_stats_df
        global_mask = df["access_kind"].isin(["GLOBAL_ACC_W"])
        hit_mask = df["access_status"].isin(["HIT"])
        miss_mask = df["access_status"].isin(["SECTOR_MISS", "MISS"])
        write_mask = df["is_write"] == True
        reads = df[(hit_mask ^ miss_mask) & write_mask & global_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_writes"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # hit_mask = df["status"] == "HIT"
        # miss_mask = df["status"] == "MISS"
        # write_mask = df["is_write"] == True
        # writes = df[(hit_mask ^ miss_mask) & write_mask]
        # writes = writes[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_writes")] = writes.values

    def _compute_l2_accesses(self):
        df = self.l2_data_stats_df
        # l2 accesses are only read in nvprof
        # global_read = df["access_kind"].isin(["GLOBAL_ACC_R"])
        mask = df["access_status"].isin(["MISS", "SECTOR_MISS", "HIT"])
        accesses = df[mask]
        grouped = accesses.groupby(INDEX_COLS, dropna=False)
        # print(
        #     accesses.groupby(
        #         INDEX_COLS + ["allocation_id", "access_kind", "access_status"],
        #         dropna=False,
        #     )["num_accesses"].sum()
        # )
        self.result_df["l2_accesses"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # hit_mask = df["status"] == "HIT"
        # miss_mask = df["status"] == "MISS"
        # accesses = df[hit_mask ^ miss_mask]
        # accesses = accesses[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_accesses")] = accesses.values

    def _compute_l2_read_hits(self):
        df = self.l2_data_stats_df
        hit_mask = df["access_status"] == "HIT"
        read_mask = df["is_write"] == False
        read_hits = df[hit_mask & read_mask]
        grouped = read_hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_read_hits"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # hit_mask = df["status"] == "HIT"
        # read_mask = df["is_write"] == False
        # read_hits = df[hit_mask & read_mask]
        # read_hits = read_hits[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_read_hits")] = read_hits.values

    def _compute_l2_write_hits(self):
        df = self.l2_data_stats_df
        hit_mask = df["access_status"] == "HIT"
        write_mask = df["is_write"] == True
        write_hits = df[hit_mask & write_mask]
        grouped = write_hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_write_hits"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # hit_mask = df["status"] == "HIT"
        # write_mask = df["is_write"] == True
        # write_hits = df[hit_mask & write_mask]
        # write_hits = write_hits[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_write_hits")] = write_hits.values

    def _compute_l2_read_misses(self):
        df = self.l2_data_stats_df
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        read_mask = df["is_write"] == False
        read_misses = df[miss_mask & read_mask]
        grouped = read_misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_read_misses"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # miss_mask = df["status"] == "MISS"
        # read_mask = df["is_write"] == False
        # read_misses = df[miss_mask & read_mask]
        # read_misses = read_misses[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_read_misses")] = read_misses.values

    def _compute_l2_write_misses(self):
        df = self.l2_data_stats_df
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        write_mask = df["is_write"] == True
        write_misses = df[miss_mask & write_mask]
        grouped = write_misses.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_write_misses"] = grouped["num_accesses"].sum()

        # df = self.l2_data_stats_df.reset_index()
        # miss_mask = df["status"] == "MISS"
        # write_mask = df["is_write"] == True
        # write_misses = df[miss_mask & write_mask]
        # write_misses = write_misses[stat_cols("count")].sum(axis=0)
        # self.result_df[stat_cols("l2_write_misses")] = write_misses.values

    def _compute_l2_hits(self):
        df = self.l2_data_stats_df
        hit_mask = df["access_status"] == "HIT"
        hits = df[hit_mask]
        # print(hits.groupby(["access_kind", "access_status"], dropna=False)["num_accesses"].sum())
        grouped = hits.groupby(INDEX_COLS, dropna=False)
        # print(grouped.sum().reset_index())
        self.result_df["l2_hits"] = grouped["num_accesses"].sum()
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_hits" + s] = self.result_df["l2_read_hits" + s] + self.result_df["l2_write_hits" + s]

    def _compute_l2_misses(self):
        df = self.l2_data_stats_df
        miss_mask = df["access_status"].isin(["MISS", "SECTOR_MISS"])
        misses = df[miss_mask]
        # print(misses.groupby(["access_kind", "access_status"], dropna=False)["num_accesses"].sum())
        grouped = misses.groupby(INDEX_COLS, dropna=False)
        # print(grouped.sum().reset_index())
        self.result_df["l2_misses"] = grouped["num_accesses"].sum()

        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_misses" + s] = (
        #         self.result_df["l2_read_misses" + s] + self.result_df["l2_write_misses" + s]
        # )

    # def exec_time_sec(self) -> float:
    #     return self.exec_time_sec_release["exec_time_mean"].sum()
    #
    # def cycles(self) -> float:
    #     return self.sim_df["cycles_mean"].sum()
    #
    # def warp_instructions(self) -> float:
    #     return 0
    #
    # def instructions(self) -> float:
    #     return self.sim_df["instructions_mean"].sum()
    #
    # def num_blocks(self) -> float:
    #     return self.sim_df["num_blocks_mean"].sum()
    #
    # def dram_reads(self) -> float:
    #     return self.dram_df["reads_mean"].sum()
    #
    # def dram_writes(self) -> float:
    #     return self.dram_df["writes_mean"].sum()
    #
    # def dram_accesses(self) -> float:
    #     return self.dram_df[["reads_mean", "writes_mean"]].sum().sum()
    #
    # def l2_reads(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     hit_mask = df["status"] == "HIT"
    #     miss_mask = df["status"] == "MISS"
    #     read_mask = df["is_write"] == False
    #     reads = df[(hit_mask ^ miss_mask) & read_mask]
    #     return reads["count_mean"].sum()
    #
    # def l2_writes(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     hit_mask = df["status"] == "HIT"
    #     miss_mask = df["status"] == "MISS"
    #     write_mask = df["is_write"] == True
    #     reads = df[(hit_mask ^ miss_mask) & write_mask]
    #     return reads["count_mean"].sum()
    #
    # def l2_accesses(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     hit_mask = df["status"] == "HIT"
    #     miss_mask = df["status"] == "MISS"
    #     accesses = df[hit_mask ^ miss_mask]
    #     return accesses["count_mean"].sum()
    #
    # def l2_read_hits(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     hit_mask = df["status"] == "HIT"
    #     read_mask = df["is_write"] == False
    #     read_hits = df[hit_mask & read_mask]
    #     return read_hits["count_mean"].sum()
    #
    # def l2_write_hits(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     hit_mask = df["status"] == "HIT"
    #     write_mask = df["is_write"] == True
    #     write_hits = df[hit_mask & write_mask]
    #     return write_hits["count_mean"].sum()
    #
    # def l2_read_misses(self) -> float:
    #     df = self.l2_data_stats_df.reset_index()
    #     miss_mask = df["status"] == "MISS"
    #     read_mask = df["is_write"] == False
    #     read_misses = df[miss_mask & read_mask]
    #     return read_misses["count_mean"].sum()
    #
    # def l2_write_misses(self) -> int:
    #     df = self.l2_data_stats_df.reset_index()
    #     miss_mask = df["status"] == "MISS"
    #     write_mask = df["is_write"] == True
    #     write_misses = df[miss_mask & write_mask]
    #     return write_misses["count_mean"].sum()


class ExecDrivenStats(Stats):
    bench_config: BenchConfig[SimulateTargetConfig]
    target_config: SimulateConfig

    # def __init__(self, config: GPUConfig, bench_config: BenchConfig[SimulateTargetConfig]) -> None:
    #     self.bench_config = bench_config
    #     self.target_config = self.bench_config["target_config"].value
    #     self.path = Path(self.target_config["stats_dir"])
    #     self.use_duration = False
    #     self.config = config
    #     self.repetitions = int(self.bench_config["common"]["repetitions"])
    #     self.load_converted_stats()
    #     self.compute_result_df()
