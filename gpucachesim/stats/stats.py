import pandas as pd
import json
from os import PathLike
from pathlib import Path
from typing import Sequence

from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    SimulateConfig,
    SimulateTargetConfig,
)
import gpucachesim.stats.common as common
from gpucachesim.stats.common import stat_cols, STAT_SUFFIXES

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

INDEX_COLS = [
    "kernel_name",
    "kernel_name_mangled",
    "kernel_launch_id",
    "run",
]

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
                exec_time_sec_release_dfs.append(pd.DataFrame.from_records([dict(run=r, exec_time=exec_time)]))

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

            dram_df = pd.read_csv(
                self.path / f"stats.dram.{r}.csv",
                header=0,
            )
            dram_df["run"] = r
            dram_dfs.append(dram_df)

            dram_banks_df = pd.read_csv(
                self.path / f"stats.dram.banks.{r}.csv",
                header=0,
            )
            dram_banks_df["run"] = r
            dram_banks_dfs.append(dram_banks_df)

            instructions_df = pd.read_csv(
                self.path / f"stats.instructions.{r}.csv",
                header=0,
                # header=None,
                # names=["memory_space", "write", "count"],
            )
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

        self.exec_time_sec_release = pd.concat(exec_time_sec_release_dfs)
        # print(exec_time_sec_release)
        # self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=None)
        print(self.exec_time_sec_release)
        # self.exec_time_sec_release = common.compute_df_statistics(exec_time_sec_release, group_by=INDEX_COLS)

        self.sim_df = pd.concat(sim_dfs)
        # print(sim_df)
        # self.sim_df = common.compute_df_statistics(sim_df, group_by=None)
        print(self.sim_df)

        self.accesses_df = pd.concat(accesses_dfs)
        # print(accesses_df)
        # self.accesses_df = common.compute_df_statistics(accesses_df, group_by=["access"])
        print(self.accesses_df)

        self.dram_df = pd.concat(dram_dfs)
        # print(dram_df)
        # self.dram_df = common.compute_df_statistics(dram_df, group_by=["chip_id", "bank_id"])
        print(self.dram_df)

        self.dram_banks_df = pd.concat(dram_banks_dfs)
        # print(dram_banks_df)
        # self.dram_banks_df = common.compute_df_statistics(dram_banks_df, group_by=["core_id", "chip_id", "bank_id"])
        print(self.dram_banks_df)

        self.instructions_df = pd.concat(instructions_dfs)
        # print(instructions_df)
        # self.instructions_df = common.compute_df_statistics(instructions_df, group_by=["memory_space", "write"])
        print(self.instructions_df)

        def load_cache_stats(dfs):
            df = pd.concat(dfs)
            # print(df)
            # df = common.compute_df_statistics(df, group_by=["cache_id", "access_type", "status", "is_write"])
            # print(df)
            return df

        self.l1_inst_stats_df = load_cache_stats(l1_inst_stats_dfs)
        self.l1_tex_stats_df = load_cache_stats(l1_tex_stats_dfs)
        self.l1_data_stats_df = load_cache_stats(l1_data_stats_dfs)
        self.l1_const_stats_df = load_cache_stats(l1_const_stats_dfs)
        self.l2_data_stats_df = load_cache_stats(l2_data_stats_dfs)

    def compute_result_df(self):
        self.result_df = pd.DataFrame()
        self._compute_cycles()
        self._compute_instructions()
        self._compute_num_blocks()
        self._compute_warp_instructions()
        self._compute_exec_time_sec()
        self._compute_is_release_build()

        self._compute_dram_reads()
        self._compute_dram_writes()
        self._compute_dram_accesses()

        self._compute_l2_reads()
        self._compute_l2_writes()
        self._compute_l2_accesses()
        self._compute_l2_read_hits()
        self._compute_l2_write_hits()
        self._compute_l2_read_misses()
        self._compute_l2_write_misses()
        self._compute_l2_hits()
        self._compute_l2_misses()

        # if False:
        # self._compute_l2_read_hit_rate()
        # self._compute_l2_write_hit_rate()
        # self._compute_l2_read_miss_rate()
        # self._compute_l2_write_miss_rate()

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

    def _compute_dram_reads(self):
        # dram_reads = self.dram_df[stat_cols("reads")].sum(axis=0).values
        # self.result_df[stat_cols("dram_reads")] = dram_reads
        # print(self.dram_df.groupby(INDEX_COLS + ["dram)["reads"].mean())
        grouped = self.dram_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["dram_reads"] = grouped["reads"].sum()

    def _compute_dram_writes(self):
        # dram_writes = self.dram_df[stat_cols("writes")].sum(axis=0)
        # self.result_df[stat_cols("dram_writes")] = dram_writes.values
        grouped = self.dram_df.groupby(INDEX_COLS, dropna=False)
        self.result_df["dram_writes"] = grouped["writes"].sum()

    def _compute_dram_accesses(self):
        # print(self.dram_df.groupby(INDEX_COLS, dropna=False)[["reads", "writes"]].sum().sum(axis=1))
        grouped = self.dram_df.groupby(INDEX_COLS, dropna=False)
        reads_and_writes = grouped[["reads", "writes"]].sum()
        self.result_df["dram_accesses"] = reads_and_writes.sum(axis=1)
        # for s in STAT_SUFFIXES:
        #     self.result_df["dram_accesses" + s] = self.result_df["dram_reads" + s] + self.result_df["dram_writes" + s]

    def _compute_l2_reads(self):
        df = self.l2_data_stats_df
        hit_mask = df["access_status"] == "HIT"
        miss_mask = df["access_status"] == "MISS"
        read_mask = df["is_write"] == False
        reads = df[(hit_mask ^ miss_mask) & read_mask]
        grouped = reads.groupby(INDEX_COLS, dropna=False)
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
        hit_mask = df["access_status"] == "HIT"
        miss_mask = df["access_status"] == "MISS"
        write_mask = df["is_write"] == True
        reads = df[(hit_mask ^ miss_mask) & write_mask]
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
        hit_mask = df["access_status"] == "HIT"
        miss_mask = df["access_status"] == "MISS"
        accesses = df[hit_mask ^ miss_mask]
        grouped = accesses.groupby(INDEX_COLS, dropna=False)
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
        miss_mask = df["access_status"] == "MISS"
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
        miss_mask = df["access_status"] == "MISS"
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
        grouped = hits.groupby(INDEX_COLS, dropna=False)
        self.result_df["l2_hits"] = grouped["num_accesses"].sum()
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_hits" + s] = self.result_df["l2_read_hits" + s] + self.result_df["l2_write_hits" + s]

    def _compute_l2_misses(self):
        df = self.l2_data_stats_df
        hit_mask = df["access_status"] == "MISS"
        hits = df[hit_mask]
        grouped = hits.groupby(INDEX_COLS, dropna=False)
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
