from os import PathLike, walk
from pathlib import Path
import cxxfilt
import typing
import numpy as np
import json
import re
import pandas as pd
from pprint import pprint

from gpucachesim.benchmarks import (
    GPUConfig,
    BenchConfig,
    ProfileConfig,
    ProfileTargetConfig,
)
import gpucachesim.stats.common as common
from gpucachesim.stats.common import stat_cols, STAT_SUFFIXES

NVPROF_INDEX_COLS = [
    "Stream",
    "Context",
    "Device",
    "Kernel",
    "Correlation_ID",
    "run",
]

NSIGHT_INDEX_COLS = [
    "Stream",
    "Context",
    "Process Name",
    "Host Name",
    "device__attribute_display_name",
    "device__attribute_device_index",
    "Kernel Name",
    "ID",
    "run",
]

NVPROF_NUMERIC_METRIC_COLUMNS = [
    "elapsed_cycles_sm",
    "issue_slots",
    "unique_warps_launched",
    "pcie_total_data_transmitted",
    "pcie_total_data_received",
    "inst_issued",
    "inst_executed",
    "inst_fp_16",
    "inst_fp_32",
    "inst_fp_64",
    "inst_integer",
    "inst_control",
    "inst_compute_ld_st",
    "inst_misc",
    "ldst_issued",
    "ldst_executed",
    "atomic_transactions",
    "l2_atomic_transactions",
    "l2_tex_read_transactions",
    "l2_tex_write_transactions",
    "ecc_transactions",
    "dram_read_transactions",
    "dram_write_transactions",
    "shared_store_transactions",
    "shared_load_transactions",
    "local_load_transactions",
    "local_store_transactions",
    "gld_transactions",
    "gst_transactions",
    "sysmem_read_transactions",
    "sysmem_write_transactions",
    "l2_read_transactions",
    "l2_write_transactions",
    "tex_cache_transactions",
    "flop_count_hp",
    "flop_count_hp_add",
    "flop_count_hp_mul",
    "flop_count_hp_fma",
    "flop_count_dp",
    "flop_count_dp_add",
    "flop_count_dp_fma",
    "flop_count_dp_mul",
    "flop_count_sp",
    "flop_count_sp_add",
    "flop_count_sp_fma",
    "flop_count_sp_mul",
    "flop_count_sp_special",
    "inst_executed_global_loads",
    "inst_executed_local_loads",
    "inst_executed_shared_loads",
    "inst_executed_surface_loads",
    "inst_executed_global_stores",
    "inst_executed_local_stores",
    "inst_executed_shared_stores",
    "inst_executed_surface_stores",
    "inst_executed_global_atomics",
    "inst_executed_global_reductions",
    "inst_executed_surface_atomics",
    "l2_global_load_bytes",
    "l2_local_load_bytes",
    "l2_surface_load_bytes",
    "l2_local_global_store_bytes",
    "l2_global_reduction_bytes",
    "l2_global_atomic_store_bytes",
    "l2_surface_store_bytes",
    "l2_surface_reduction_bytes",
    "l2_surface_atomic_store_bytes",
    "sysmem_read_bytes",
    "sysmem_write_bytes",
    "dram_write_bytes",
    "dram_read_bytes",
    "global_load_requests",
    "local_load_requests",
    "surface_load_requests",
    "global_store_requests",
    "local_store_requests",
    "surface_store_requests",
    "global_atomic_requests",
    "global_reduction_requests",
    "surface_atomic_requests",
    "surface_reduction_requests",
    "texture_load_requests",
    "inst_per_warp",
    "ipc",
    "issued_ipc",
    "issue_slot_utilization",
    "eligible_warps_per_cycle",
    "inst_replay_overhead",
    "local_memory_overhead",
    "tex_cache_hit_rate",
    "l2_tex_read_hit_rate",
    "l2_tex_write_hit_rate",
    "l2_tex_hit_rate",
    "global_hit_rate",
    "local_hit_rate",
    "stall_inst_fetch",
    "stall_exec_dependency",
    "stall_memory_dependency",
    "stall_texture",
    "stall_sync",
    "stall_other",
    "stall_constant_memory_dependency",
    "stall_pipe_busy",
    "stall_memory_throttle",
    "stall_not_selected",
    "atomic_transactions_per_request",
    "shared_store_transactions_per_request",
    "shared_load_transactions_per_request",
    "local_load_transactions_per_request",
    "local_store_transactions_per_request",
    "gld_transactions_per_request",
    "gst_transactions_per_request",
    "sm_efficiency",
    "shared_efficiency",
    "flop_hp_efficiency",
    "flop_sp_efficiency",
    "flop_dp_efficiency",
    "gld_efficiency",
    "gst_efficiency",
    "branch_efficiency",
    "warp_execution_efficiency",
    "warp_nonpred_execution_efficiency",
    "dram_read_throughput",
    "dram_write_throughput",
    "l2_atomic_throughput",
    "ecc_throughput",
    "tex_cache_throughput",
    "l2_tex_read_throughput",
    "l2_tex_write_throughput",
    "l2_read_throughput",
    "l2_write_throughput",
    "sysmem_read_throughput",
    "sysmem_write_throughput",
    "local_load_throughput",
    "local_store_throughput",
    "shared_load_throughput",
    "shared_store_throughput",
    "gld_throughput",
    "gst_throughput",
    "gld_requested_throughput",
    "gst_requested_throughput",
]


def normalize_nvprof_device_name(name):
    # Strip off device numbers, e.g. (0), (1)
    # that some profiler versions add to the end of device name
    return re.sub(r" \(\d+\)$", "", name)


class NsightStats(common.Stats):
    bench_config: BenchConfig[ProfileTargetConfig]
    target_config: ProfileConfig

    def __init__(
        self, config: GPUConfig, bench_config: BenchConfig[ProfileTargetConfig]
    ) -> None:
        self.bench_config = bench_config
        self.target_config = self.bench_config["target_config"].value
        self.path = Path(self.target_config["profile_dir"])
        self.repetitions = self.bench_config["common"]["repetitions"]
        self.use_duration = False
        self.config = config

        dfs = []
        for r in range(self.repetitions):
            with open(self.path / f"profile.nsight.metrics.{r}.json", "rb") as f:
                metrics = json.load(f)
                # pprint(metrics)
                df = pd.DataFrame.from_records(
                    [
                        {k: None if v is None else v["value"] for k, v in m.items()}
                        for m in metrics
                    ]
                )
                df = df.drop(columns=["Kernel Time"])
                df["run"] = r
                # _units = pd.DataFrame.from_records([{k: v["unit"] for k, v in m.items()} for m in metrics])
                dfs.append(df)

        self.df = pd.concat(dfs)

        self.compute_native_result_df_nsight()

    def compute_native_result_df_nsight(self):
        self.result_df = pd.DataFrame()
        self._compute_cycles()
        self._compute_num_blocks()
        self._compute_exec_time_sec()
        self._compute_instructions()
        self._compute_warp_instructions()

        # DRAM
        self._compute_dram_reads()
        self._compute_dram_writes()
        self._compute_dram_accesses()

        # L2 rates
        self._compute_l2_read_hit_rate()
        self._compute_l2_write_hit_rate()
        self._compute_l2_read_miss_rate()
        self._compute_l2_write_miss_rate()
        self._compute_l2_hit_rate()
        self._compute_l2_miss_rate()

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

        # L1 rates
        self._compute_l1_hit_rate()
        self._compute_l1_miss_rate()

        # L1 accesses
        self._compute_l1_accesses()
        self._compute_l1_reads()
        self._compute_l1_writes()
        self._compute_l1_hits()
        self._compute_l1_misses()

        # fix the index
        self.result_df = self.result_df.reset_index()
        self.result_df = self.result_df.rename(
            columns={
                "Stream": "stream_id",
                "Context": "context_id",
                "device__attribute_display_name": "device",
                "Kernel Name": "kernel_name",
                "ID": "kernel_launch_id",
            }
        )
        self.result_df["kernel_name_mangled"] = np.nan
        self.result_df["kernel_function_signature"] = np.nan
        assert "run" in self.result_df.columns

        # map sorted correlation ids to increasing launch ids
        launch_ids = sorted(self.result_df["kernel_launch_id"].unique().tolist())
        new_launch_ids = {old: new for new, old in enumerate(launch_ids)}
        self.result_df["kernel_launch_id"] = self.result_df["kernel_launch_id"].apply(
            lambda id: new_launch_ids[id]
        )

    def _compute_cycles(self):
        # print(self.df.dropna(axis=1).groupby(NSIGHT_INDEX_COLS, dropna=False).mean().T)
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        sm_count = self.config.num_total_cores
        # print(grouped[["sm__cycles_active.avg", "sm__cycles_elapsed.sum"]].mean())
        # print(grouped["sm__cycles_active.avg", "gpc__cycles_elapsed.avg"].mean().columns.tolist())
        # sm__elapsed_cycles_sum
        # sm__elapsed_cycles_avg
        self.result_df["cycles"] = grouped["sm__active_cycles_avg"].sum()
        # self.result_df["cycles"] /= sm_count

    def _kernel_durations_us(self):
        # convert ns to us
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)

        return grouped["gpu__time_duration"].sum() * 1e-3

    def _compute_exec_time_sec(self):
        # print(self._kernel_durations_us())
        # print(self.result_df)
        self.result_df["exec_time_sec"] = self._kernel_durations_us().values * float(
            1e-6
        )

    def _compute_num_blocks(self):
        self.result_df["num_blocks"] = np.nan

    def _compute_instructions(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        # there is also sm__inst_executed.sum_inst
        # sm__sass_thread_inst_executed.sum_inst
        # self.result_df["instructions"] = grouped["smsp__inst_executed_sum"].sum()
        # self.result_df["instructions"] = grouped["sm__inst_executed_sum"].sum()
        # self.result_df["instructions"] = grouped["smsp__thread_inst_executed_sum"].sum()
        # self.result_df["instructions"] = grouped["smsp__thread_inst_executed_sum"].sum()
        self.result_df["instructions"] = grouped[
            "smsp__thread_inst_executed_not_pred_off_sum"
        ].sum()

    def _compute_warp_instructions(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["warp_inst"] = grouped["smsp__inst_executed_per_warp"].sum()

    def _compute_dram_reads(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        # print(self.df[NSIGHT_INDEX_COLS + ["dram__read_sectors"]])
        self.result_df["dram_reads"] = grouped["dram__read_sectors"].sum()

    def _compute_dram_writes(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["dram_writes"] = grouped["dram__write_sectors"].sum()

    def _compute_dram_accesses(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        accesses = grouped[["dram__read_sectors", "dram__write_sectors"]]
        self.result_df["dram_accesses"] = accesses.sum().astype(float).sum(axis=1)

    def _compute_l2_reads(self):
        # lts__request_tex_write_sectors
        # lts__request_tex_read_sectors
        # lts__request_tex_sectors
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        # print(grouped["lts__request_tex_read_sectors"].sum())
        self.result_df["l2_reads"] = grouped["lts__request_tex_read_sectors"].sum()

    def _compute_l2_writes(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l2_writes"] = grouped["lts__request_tex_write_sectors"].sum()

    def _compute_l2_accesses(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        accesses = (
            grouped[["lts__request_tex_read_sectors", "lts__request_tex_write_sectors"]]
            .sum()
            .astype(float)
            .sum(axis=1)
        )
        # self.df["lts__request_tex_sectors"].fillna(0.0)
        # print(self.df["lts__request_tex_sectors"].fillna(0.0))
        # print(accesses.fillna(0.0).sum(axis=1))
        # print(self.df["lts__request_tex_sectors"].fillna(0.0).sum())
        # print(accesses.fillna(0.0).sum(axis=1).sum())
        # assert (self.df["lts__request_tex_sectors"].fillna(0.0) == accesses.fillna(0.0).sum(axis=1)).all()
        total = grouped["lts__request_tex_sectors"].sum()
        # less than 5 percent rel error, otherwise we should investigate
        # print(total.shape)
        # print(accesses.shape)
        # print((total - accesses).abs())
        # print((total - accesses).abs() / total.abs())
        assert (((total - accesses).abs() / total.abs()) < 0.05).all()

        self.result_df["l2_accesses"] = grouped["lts__request_tex_sectors"].sum()

    def _compute_l2_read_hit_rate(self):
        self.result_df["l2_read_hit_rate"] = np.nan

    def _compute_l2_write_hit_rate(self):
        self.result_df["l2_write_hit_rate"] = np.nan

    def _compute_l2_read_miss_rate(self):
        self.result_df["l2_read_miss_rate"] = np.nan

    def _compute_l2_write_miss_rate(self):
        self.result_df["l2_write_miss_rate"] = np.nan

    def _compute_l2_read_hits(self):
        self.result_df["l2_read_hits"] = np.nan

    def _compute_l2_write_hits(self):
        self.result_df["l2_write_hits"] = np.nan

    def _compute_l2_read_misses(self):
        self.result_df["l2_read_misses"] = np.nan

    def _compute_l2_write_misses(self):
        self.result_df["l2_write_misses"] = np.nan

    def _compute_l2_hits(self):
        self.result_df["l2_hits"] = np.nan

    def _compute_l2_misses(self):
        # grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)

        # tex__m_rd_sectors_miss_global_ld_cached
        # tex__m_rd_sectors_miss_global_ld_cached_pct
        # tex__m_rd_sectors_miss_global_ld_uncached
        # tex__m_rd_sectors_miss_global_ld_uncached_pct
        # tex__m_rd_sectors_miss_local_ld_cached
        # tex__m_rd_sectors_miss_local_ld_cached_pct
        # tex__m_rd_sectors_miss_local_ld_uncached
        # tex__m_rd_sectors_miss_local_ld_uncached_pct
        # tex__m_rd_sectors_miss_surface_ld
        # tex__m_rd_sectors_miss_surface_ld_pct
        # lts__request_total_sectors_hitrate_pct
        # tex__hitrate_pct
        self.result_df["l2_misses"] = np.nan

    def _compute_l1_accesses(self):
        self.result_df["l1_accesses"] = np.nan

    def _compute_l1_reads(self):
        self.result_df["l1_reads"] = np.nan

    def _compute_l1_writes(self):
        self.result_df["l1_writes"] = np.nan

    def _compute_l1_hit_rate(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_hit_rate"] = grouped["tex__hitrate_pct"].mean()
        self.result_df["l1_hit_rate"] /= 100.0

    def _compute_l1_hits(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_hits"] = (
            grouped[
                [
                    "tex__t_sectors_hit_global_ld_cached",
                    "tex__t_sectors_hit_local_ld_cached",
                ]
            ]
            .sum()
            .astype(float)
            .sum(axis=1)
        )

    def _compute_l1_read_hit_rate(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_read_hit_rate"] = (
            grouped[
                [
                    "tex__m_rd_sectors_miss_global_ld_cached_pct",
                    "tex__m_rd_sectors_miss_global_ld_uncached_pct",
                    "tex__m_rd_sectors_miss_local_ld_cached_pct",
                    "tex__m_rd_sectors_miss_local_ld_uncached_pct",
                    "tex__m_rd_sectors_miss_surface_ld_pct",
                ]
            ]
            .mean()
            .astype(float)
            .mean(axis=1)
        )
        self.result_df["l1_read_hit_rate"] /= 100.0

    def _compute_l1_miss_rate(self):
        self.result_df["l1_miss_rate"] = 1.0 - self.result_df["l1_hit_rate"]

    def _compute_l1_misses(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_misses"] = (
            grouped[
                [
                    # "tex__t_sectors_miss_global_ld_cached",
                    # "tex__t_sectors_miss_global_ld_uncached",
                    # "tex__t_sectors_miss_local_ld_cached",
                    # "tex__t_sectors_miss_local_ld_uncached",
                    # "tex__t_sectors_miss_surface_ld",
                    # l1 MISS writes
                    "tex__m_wr_sectors_atom_red",
                    "tex__m_wr_sectors_global_atom",
                    "tex__m_wr_sectors_global_nonatom",
                    "tex__m_wr_sectors_global_red",
                    "tex__m_wr_sectors_local_st",
                    "tex__m_wr_sectors_surface_atom",
                    "tex__m_wr_sectors_surface_nonatom",
                    "tex__m_wr_sectors_surface_red",
                    # l1 MISS reads
                    "tex__m_rd_sectors_miss_surface_ld",
                    "tex__m_rd_sectors_miss_local_ld_uncached",
                    "tex__m_rd_sectors_miss_local_ld_cached",
                    "tex__m_rd_sectors_miss_global_ld_uncached",
                    "tex__m_rd_sectors_miss_global_ld_cached",
                ]
            ]
            .sum()
            .astype(float)
            .sum(axis=1)
        )

    def _compute_l1_read_misses(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_read_misses"] = (
            grouped[
                [
                    "tex__m_rd_sectors_miss_global_ld_cached",
                    "tex__m_rd_sectors_miss_global_ld_uncached",
                    "tex__m_rd_sectors_miss_local_ld_cached",
                    "tex__m_rd_sectors_miss_local_ld_uncached",
                    "tex__m_rd_sectors_miss_surface_ld",
                ]
            ]
            .sum()
            .astype(float)
            .sum(axis=1)
        )

    def _compute_l1_write_misses(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        self.result_df["l1_write_misses"] = (
            grouped[
                [
                    "tex__m_wr_sectors_atom_red",
                    "tex__m_wr_sectors_global_atom",
                    "tex__m_wr_sectors_global_nonatom",
                    "tex__m_wr_sectors_global_red",
                    "tex__m_wr_sectors_local_st",
                    "tex__m_wr_sectors_surface_atom",
                    "tex__m_wr_sectors_surface_nonatom",
                    "tex__m_wr_sectors_surface_red",
                ]
            ]
            .sum()
            .astype(float)
            .sum(axis=1)
        )

    def _compute_l2_hit_rate(self):
        grouped = self.df.groupby(NSIGHT_INDEX_COLS, dropna=False)
        # lts__request_total_sectors_hitrate_pct
        # lts__t_sectors_srcunit_tex_op_read_lookup_hit.sum_sector
        self.result_df["l2_hit_rate"] = grouped[
            "lts__request_total_sectors_hitrate_pct"
        ].mean()
        self.result_df["l2_hit_rate"] /= 100.0

    def _compute_l2_miss_rate(self):
        self.result_df["l2_miss_rate"] = 1.0 - self.result_df["l2_hit_rate"]


class NvprofStats(common.Stats):
    bench_config: BenchConfig[ProfileTargetConfig]
    target_config: ProfileConfig

    def __init__(
        self, config: GPUConfig, bench_config: BenchConfig[ProfileTargetConfig]
    ) -> None:
        self.bench_config = bench_config
        self.target_config = self.bench_config["target_config"].value
        self.path = Path(self.target_config["profile_dir"])
        self.repetitions = self.bench_config["common"]["repetitions"]
        self.use_duration = False
        self.config = config

        dfs = []
        command_dfs = []
        for r in range(self.repetitions):
            with open(self.path / f"profile.nvprof.commands.{r}.json", "rb") as f:
                commands = json.load(f)
                commands_df = pd.DataFrame.from_records(
                    [{k: v["value"] for k, v in c.items()} for c in commands]
                )
                # _units = pd.DataFrame.from_records([{k: v["unit"] for k, v in c.items()} for c in commands])

                # name refers to kernels now
                commands_df = commands_df.rename(columns={"Name": "Kernel"})

                # commands_df["Device"] = commands_df["Device"].apply(normalize_device_name)
                commands_df["run"] = r
                command_dfs.append(commands_df)

            with open(self.path / f"profile.nvprof.metrics.{r}.json", "rb") as f:
                metrics = json.load(f)
                df = pd.DataFrame.from_records(
                    [{k: v["value"] for k, v in m.items()} for m in metrics]
                )
                # df["Device"] = df["Device"].apply(normalize_device_name)
                df["run"] = r

                # _units = pd.DataFrame.from_records([{k: v["unit"] for k, v in m.items()} for m in metrics])
                dfs.append(df)

        self.df = pd.concat(dfs)

        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        if False:
            print(
                grouped[
                    [
                        "l2_read_transactions",
                        "l2_tex_read_transactions",
                        "l2_tex_read_hit_rate",
                    ]
                ]
                .mean()
                .reset_index()
            )
            print(
                grouped[
                    [
                        "l2_write_transactions",
                        "l2_tex_write_transactions",
                        "l2_tex_write_hit_rate",
                    ]
                ]
                .mean()
                .reset_index()
            )
            print(
                grouped[
                    [
                        "tex_cache_transactions",
                        "tex_cache_hit_rate",
                    ]
                ]
                .mean()
                .reset_index()
            )

        # nvprof_key = "l2_read_transactions"
        # if nvprof_key in self.df:
        #     grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        #     self.result_df["l2_reads"] = grouped[nvprof_key].sum()

        self.commands_df = pd.concat(command_dfs)

        self.compute_native_result_df_nvprof()

        # print(grouped[["dram_write_transactions", "dram_read_transactions"]].sum().head())
        if False:
            print(self.result_df[["l1_accesses", "l1_hit_rate"]].head(n=20))
            print(
                self.result_df[
                    [
                        "l2_accesses",
                        "l2_hits",
                        "l2_misses",
                        "l2_read_hit_rate",
                        "l2_write_hit_rate",
                    ]
                ].head(n=20)
            )
            print(
                self.result_df[["dram_accesses", "dram_reads", "dram_writes"]].head(
                    n=20
                )
            )

        # print(commands_df.select_dtypes(include=["object"]).columns)

    def compute_native_result_df_nvprof(self):
        self.result_df = pd.DataFrame()
        self._compute_cycles()
        self._compute_num_blocks()
        self._compute_num_warps()
        self._compute_exec_time_sec()
        self._compute_instructions()
        self._compute_warp_instructions()

        # DRAM
        self._compute_dram_reads()
        self._compute_dram_writes()
        self._compute_dram_accesses()

        # L2 rates
        self._compute_l2_read_hit_rate()
        self._compute_l2_write_hit_rate()
        self._compute_l2_read_miss_rate()
        self._compute_l2_write_miss_rate()
        self._compute_l2_hit_rate()
        self._compute_l2_miss_rate()

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

        # L1 accesses
        self._compute_l1_accesses()
        self._compute_l1_reads()
        self._compute_l1_writes()

        # L1 rates
        self._compute_l1_hit_rate()
        self._compute_l1_global_hit_rate()
        self._compute_l1_local_hit_rate()
        self._compute_l1_hits()
        self._compute_l1_miss_rate()
        self._compute_l1_misses()

        # chart_name="L1D Hit Rate",
        # plotfile="l1hitrate",
        # hw_eval="np.average(hw[\"tex_cache_hit_rate\"])",
        # hw_error=None,
        # sim_eval="float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])" +\
        #          "/(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])" +\
        #          "+float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"]) + 1) * 100",

        # CorrelStat(chart_name="L1D Hit Rate (global_hit_rate match)",
        # plotfile="l1hitrate.global",
        # hw_eval="np.average(hw[\"global_hit_rate\"])",
        # hw_error=None,
        # sim_eval="(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[HIT\]\s*=\s*(.*)\"])" +\
        #         " + float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[HIT\]\s*=\s*(.*)\"]))" +\
        #          "/(float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_W\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"])" +\
        #          "+float(sim[\"\s+Total_core_cache_stats_breakdown\[GLOBAL_ACC_R\]\[TOTAL_ACCESS\]\s*=\s*(.*)\"]) + 1) * 100",

        # fix the index
        self.result_df = self.result_df.reset_index()
        self.result_df = self.result_df.rename(
            columns={
                "Stream": "stream_id",
                "Context": "context_id",
                "Device": "device",
                "Kernel": "kernel_name_mangled",
                "Correlation_ID": "kernel_launch_id",
            }
        )
        self.result_df["kernel_function_signature"] = self.result_df[
            "kernel_name_mangled"
        ].apply(lambda name: cxxfilt.demangle(name))
        self.result_df["kernel_name"] = np.nan

        # map sorted correlation ids to increasing launch ids
        launch_ids = sorted(self.result_df["kernel_launch_id"].unique().tolist())
        new_launch_ids = {old: new for new, old in enumerate(launch_ids)}
        self.result_df["kernel_launch_id"] = self.result_df["kernel_launch_id"].apply(
            lambda id: new_launch_ids[id]
        )

    def _compute_exec_time_sec(self):
        # print(self._kernel_durations_us())
        # print(self.result_df)
        self.result_df["exec_time_sec"] = self._kernel_durations_us().values * float(
            1e-6
        )
        # self.result_df["exec_time_sec"] = self._kernel_durations_us() * float(1e-6)
        # self.result_df[stat_cols("exec_time_sec")] = self._duration_us() * float(1e-6)

    # def exec_time_sec(self) -> float:
    #     return self.result_df["exec_time_sec_mean"].sum()
    #     # return self._duration_us().sum() * float(1e-6)

    def _compute_cycles(self):
        if self.use_duration:
            # clock speed is mhz, so *1e6
            # duration is us, so *1e-6
            # unit conversions cancel each other out
            durations = self._kernel_durations_us()
            clock_speed = float(self.config.core_clock_speed)
            self.result_df["cycles"] = durations * clock_speed
            # self.result_df["cycles_mean"] = durations["mean"] * clock_speed
            # self.result_df["cycles_min"] = durations["min"] * clock_speed
            # self.result_df["cycles_max"] = durations["max"] * clock_speed
            # self.result_df["cycles_std"] = durations["std"] * clock_speed
        else:
            grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
            sm_count = self.config.num_total_cores
            self.result_df["cycles"] = grouped["elapsed_cycles_sm"].sum()
            self.result_df["cycles"] /= sm_count

    def _compute_num_blocks(self):
        self.result_df["num_blocks"] = np.nan

    def _compute_num_warps(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["num_warps"] = grouped["unique_warps_launched"].sum()

    # def num_blocks(self) -> float:
    #     return self.result_df["num_blocks_mean"].sum()

    def _compute_instructions(self):
        inst_cols = [
            "inst_fp_16",
            "inst_fp_32",
            "inst_fp_64",
            "inst_integer",
            "inst_control",
            "inst_compute_ld_st",
            "inst_misc",
        ]

        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        # print(grouped[inst_cols].sum().astype(float))
        # print(grouped[inst_cols].sum().astype(float).sum(axis=1))
        self.result_df["instructions"] = (
            grouped[inst_cols].sum().astype(float).sum(axis=1)
        )
        # self.result_df["instructions"] = grouped[inst_cols].sum().astype(float).sum(axis=1)
        # self.result_df[stat_cols("instructions")] = self.df[
        #         [col + "_mean" for col in inst_cols]].sum().sum()
        # self.result_df["instructions_mean"] = self.df[[col + "_mean" for col in inst_cols]].sum().sum()
        # self.result_df["instructions_min"] = self.df[[col + "_min" for col in inst_cols]].sum().sum()
        # self.result_df["instructions_max"] = self.df[[col + "_max" for col in inst_cols]].sum().sum()
        # self.result_df["instructions_std"] = self.df[[col + "_std" for col in inst_cols]].sum().sum()

    # def instructions(self) -> float:
    #     return self.result_df["instructions_mean"].sum()
    # inst_cols = [
    #     "inst_fp_16_mean",
    #     "inst_fp_32_mean",
    #     "inst_fp_64_mean",
    #     "inst_integer_mean",
    #     "inst_control_mean",
    #     "inst_compute_ld_st_mean",
    #     "inst_misc_mean",
    # ]
    # total_instructions = self.df[inst_cols].sum().sum()
    # # print(total_instructions)
    # # per_run_total_instructions = self.df[["run"] + inst_cols].astype(int)
    # # print(per_run_total_instructions)
    # # per_run_total_instructions = per_run_total_instructions.groupby("run")[inst_cols].sum()
    # # per_run_total_instructions = per_run_total_instructions.sum(axis=1)
    # return int(total_instructions)

    def _compute_warp_instructions(self):
        nvprof_key = "inst_per_warp"
        if nvprof_key in self.df:
            # print(self.df[stat_cols(nvprof_key)])
            grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
            self.result_df["warp_inst"] = grouped[nvprof_key].sum()
            # self.result_df[stat_cols("warp_inst")] = self.df[stat_cols(nvprof_key)]
        else:
            raise ValueError("missing nsight warp instructions")

    # def warp_instructions(self) -> float:
    #     return self.result_df["warp_inst_mean"].sum()
    #     # nvprof_key = "inst_per_warp_mean"
    #     # if nvprof_key in self.df:
    #     #     return float(self.df[nvprof_key].mean())
    #     # else:
    #     #     raise ValueError("nsight warp instructions")

    def _compute_dram_reads(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        # print(grouped[[nvprof_key]].mean().head())
        # print(grouped[[nvprof_key]].sum().head())
        assert (
            grouped["dram_read_transactions"].mean()
            == grouped["dram_read_transactions"].sum()
        ).all()
        self.result_df["dram_reads"] = grouped["dram_read_transactions"].mean()

    # def dram_reads(self) -> float:
    #     return self.result_df["dram_reads_mean"].sum()
    #     # nvprof_key = "dram_read_transactions_mean"
    #     # if nvprof_key in self.df:
    #     #     return int(self.df[nvprof_key].sum())
    #     # else:
    #     #     nsight_key = "dram__sectors_read.sum_sector"
    #     #     return int(self.df[nsight_key].sum())

    def _compute_dram_writes(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS)
        self.result_df["dram_writes"] = grouped["dram_write_transactions"].sum()

    # def dram_writes(self) -> float:
    #     return self.result_df["dram_writes_mean"].sum()
    #     # nvprof_key = "dram_write_transactions_mean"
    #     # if nvprof_key in self.df:
    #     #     return int(self.df[nvprof_key].sum())
    #     # else:
    #     #     nsight_key = "dram__sectors_write.sum_sector"
    #     #     return int(self.df[nsight_key].sum())

    def _compute_dram_accesses(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        reads_and_writes = grouped[
            ["dram_read_transactions", "dram_write_transactions"]
        ]
        self.result_df["dram_accesses"] = (
            reads_and_writes.sum().astype(float).sum(axis=1)
        )

        # for s in STAT_SUFFIXES:
        #     if "dram_read_transactions_mean" in self.df:
        #         self.result_df["dram_accesses" + s] = self.df[
        #             ["dram_read_transactions" + s, "dram_write_transactions" + s]
        #         ].sum(axis=1)
        #     else:
        #         self.result_df["dram_accesses" + s] = self.df[
        #             [
        #                 "dram__sectors_read.sum_sector" + s,
        #                 "dram__sectors_write.sum_sector" + s,
        #             ]
        # ].sum(axis=1)

    # def dram_accesses(self) -> float:
    #     return self.result_df["dram_accesses_mean"].sum()
    #     # nvprof_keys = ["dram_read_transactions_mean", "dram_write_transactions_mean"]
    #     # if set(nvprof_keys).issubset(self.df):
    #     #     return self.df[nvprof_keys].sum().sum()
    #     # else:
    #     #     nsight_keys = [
    #     #         "dram__sectors_read.sum_sector",
    #     #         "dram__sectors_write.sum_sector",
    #     #     ]
    #     #     return self.df[nsight_keys].sum().sum()

    def _compute_l2_reads(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l2_reads"] = grouped["l2_tex_read_transactions"].sum()

    # def l2_reads(self) -> float:
    #     return self.result_df["l2_reads_mean"].sum()
    #     # nvprof_key = "l2_tex_read_transactions_mean"
    #     # nvprof_key = "l2_read_transactions_mean"
    #     # if nvprof_key in self.df:
    #     #     return self.df[nvprof_key].sum()
    #     # else:
    #     #     return self.df["lts__t_sectors_srcunit_tex_op_read.sum_sector"].sum()

    def _compute_l2_writes(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l2_writes"] = grouped["l2_tex_write_transactions"].sum()

    # def l2_writes(self) -> float:
    #     return self.result_df["l2_writes_mean"].sum()
    #     # nvprof_key = "l2_tex_write_transactions_mean"
    #     # nvprof_key = "l2_write_transactions_mean"
    #     # if nvprof_key in self.df:
    #     #     return self.df[nvprof_key].sum()
    #     # else:
    #     #     nsight_key = "lts__t_sectors_srcunit_tex_op_write.sum_sector"
    #     #     return self.df[nsight_key].sum()

    def _compute_l2_accesses(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        reads_and_writes = grouped[["l2_read_transactions", "l2_write_transactions"]]
        self.result_df["l2_accesses"] = reads_and_writes.sum().astype(float).sum(axis=1)
        # self.result_df["l2_accesses"] = grouped["tex_cache_transactions"].sum()

    # def l2_accesses(self) -> float:
    #     return self.result_df["l2_accesses_mean"].sum()
    #     # # nvprof_read_key = "l2_tex_read_transactions"
    #     # # nvprof_write_key = "l2_tex_write_transactions"
    #     #
    #     # nvprof_keys = ["l2_read_transactions_mean", "l2_write_transactions_mean"]
    #     # if set(nvprof_keys).issubset(self.df):
    #     #     # if nvprof_keys in self.df:
    #     #     return self.df[nvprof_keys].sum().sum()
    #     # else:
    #     #     nsight_keys = [
    #     #         "lts__t_sectors_srcunit_tex_op_write.sum_sector",
    #     #         "lts__t_sectors_srcunit_tex_op_read.sum_sector",
    #     #     ]
    #     #     return self.df[nsight_keys].sum().sum()

    def _compute_l2_read_hit_rate(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l2_read_hit_rate"] = grouped["l2_tex_read_hit_rate"].mean()
        self.result_df["l2_read_hit_rate"] /= 100.0
        # self.result_df[stat_cols("l2_read_hit_rate")] = self.df[stat_cols("l2_tex_read_hit_rate")] / 100.0

    # def l2_read_hit_rate(self) -> float:
    #     return float(self.result_df["l2_read_hit_rate_mean"].mean())
    #     # nvprof_key = "l2_tex_read_hit_rate_mean"
    #     # # nvprof_key = "l2_read_hit_rate"
    #     # return float(self.df[nvprof_key].mean()) / 100.0

    def _compute_l2_write_hit_rate(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l2_write_hit_rate"] = grouped["l2_tex_write_hit_rate"].mean()
        self.result_df["l2_write_hit_rate"] /= 100.0
        # self.result_df[stat_cols("l2_write_hit_rate")] = self.df[stat_cols("l2_tex_write_hit_rate")] / 100.0
        # self.result_df[stat_cols("l2_write_hit_rate")] = self.df[stat_cols("l2_tex_write_hit_rate")] / 100.0
        # nvprof_key = "l2_tex_write_hit_rate_mean"
        # # nvprof_key = "l2_write_hit_rate"
        # return float(self.df[nvprof_key].mean()) / 100.0

    # def l2_write_hit_rate(self) -> float:
    #     return float(self.result_df["l2_write_hit_rate_mean"].mean())
    #     # nvprof_key = "l2_tex_write_hit_rate_mean"
    #     # # nvprof_key = "l2_write_hit_rate"
    #     # return float(self.df[nvprof_key].mean()) / 100.0

    def _compute_l2_read_miss_rate(self):
        self.result_df["l2_read_miss_rate"] = 1.0 - self.result_df["l2_read_hit_rate"]
        # self.result_df[stat_cols("l2_read_miss_rate")] = 1.0 - self.result_df[stat_cols("l2_read_hit_rate")]

    # def l2_read_miss_rate(self) -> float:
    #     return float(self.result_df["l2_read_miss_rate_mean"].mean())
    #     # return 1 - self.l2_read_hit_rate()

    def _compute_l2_write_miss_rate(self):
        self.result_df["l2_write_miss_rate"] = 1.0 - self.result_df["l2_write_hit_rate"]
        # self.result_df[stat_cols("l2_write_miss_rate")] = 1.0 - self.result_df[stat_cols("l2_write_hit_rate")]

    def _compute_l2_hit_rate(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l2_hit_rate"] = grouped["l2_tex_hit_rate"].mean()
        self.result_df["l2_hit_rate"] /= 100.0

    def _compute_l2_miss_rate(self):
        self.result_df["l2_miss_rate"] = 1.0 - self.result_df["l2_hit_rate"]

    # def l2_write_miss_rate(self) -> float:
    #     return float(self.result_df["l2_write_miss_rate_mean"].mean())
    #     # return 1 - self.l2_write_hit_rate()

    def _compute_l2_read_hits(self):
        self.result_df["l2_read_hits"] = (
            self.result_df["l2_reads"] * self.result_df["l2_read_hit_rate"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_read_hits" + s] = self.result_df["l2_reads" + s] * self.result_df["l2_read_hit_rate" + s]
        # return int(float(self.l2_reads()) * self.l2_read_hit_rate())

    # def l2_read_hits(self) -> float:
    #     return self.result_df["l2_read_hits_mean"].sum()
    #     # return int(float(self.l2_reads()) * self.l2_read_hit_rate())

    def _compute_l2_write_hits(self):
        self.result_df["l2_write_hits"] = (
            self.result_df["l2_writes"] * self.result_df["l2_write_hit_rate"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_write_hits" + s] = (
        #         self.result_df["l2_writes" + s] * self.result_df["l2_write_hit_rate" + s]
        #     )
        # return int(float(self.l2_writes()) * self.l2_write_hit_rate())

    # def l2_write_hits(self) -> float:
    #     return self.result_df["l2_write_hits_mean"].sum()
    #     # return int(float(self.l2_writes()) * self.l2_write_hit_rate())

    def _compute_l2_read_misses(self):
        self.result_df["l2_read_misses"] = (
            self.result_df["l2_reads"] * self.result_df["l2_read_miss_rate"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_read_misses" + s] = (
        #         self.result_df["l2_reads" + s] * self.result_df["l2_read_miss_rate" + s]
        #     )

    # def l2_read_misses(self) -> float:
    #     return self.result_df["l2_read_misses_mean"].sum()
    # return int(float(self.l2_reads()) * self._l2_read_miss_rate())

    def _compute_l2_write_misses(self):
        self.result_df["l2_write_misses"] = (
            self.result_df["l2_writes"] * self.result_df["l2_write_miss_rate"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_write_misses" + s] = (
        #         self.result_df["l2_writes" + s] * self.result_df["l2_write_miss_rate" + s]
        #     )

    # def l2_write_misses(self) -> float:
    #     return self.result_df["l2_write_misses_mean"].sum()
    #     # return int(float(self.l2_writes()) * self._l2_write_miss_rate())

    def _compute_l2_hits(self):
        self.result_df["l2_hits"] = (
            self.result_df["l2_read_hits"] + self.result_df["l2_write_hits"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_hits" + s] = self.result_df["l2_read_hits" + s] + self.result_df["l2_write_hits" + s]
        # return self.l2_read_hits() + self.l2_write_hits()

    # def l2_hits(self) -> float:
    #     return self.result_df["l2_hits"].sum()
    #     # return self.l2_read_hits() + self.l2_write_hits()

    def _compute_l2_misses(self):
        self.result_df["l2_misses"] = (
            self.result_df["l2_read_misses"] + self.result_df["l2_write_misses"]
        )
        # for s in STAT_SUFFIXES:
        #     self.result_df["l2_misses" + s] = (
        #         self.result_df["l2_read_misses" + s] + self.result_df["l2_write_misses" + s]
        #     )

    # def l2_misses(self) -> float:
    #     return self.result_df["l2_misses"].sum()
    #     # return self.l2_read_misses() + self.l2_write_misses()

    def _compute_l1_accesses(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        # note: tex_cache_transaction are only READ transactions
        self.result_df["l1_accesses"] = grouped["tex_cache_transactions"].sum()

    def _compute_l1_reads(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l1_accesses"] = grouped["tex_cache_transactions"].sum()
        # self.result_df["l1_reads"] = grouped["gld_transactions"].sum()

    def _compute_l1_writes(self):
        self.result_df["l1_writes"] = np.nan

    def _compute_l1_global_hit_rate(self):
        # global_hit_rate: (GLOBAL_ACC_R[HIT]+GLOBAL_ACC_W[HIT]) / (GLOBAL_ACC_R[TOTAL]+GLOBAL_ACC_W[TOTAL])
        # tex_cache_hit_ratek: GLOBAL_ACC_R[HIT]/(GLOBAL_ACC_R[TOTAL]+GLOBAL_ACC_W[TOTAL])
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l1_global_hit_rate"] = grouped["global_hit_rate"].mean()
        self.result_df["l1_global_hit_rate"] /= 100.0

    def _compute_l1_local_hit_rate(self):
        # global_hit_rate: (GLOBAL_ACC_R[HIT]+GLOBAL_ACC_W[HIT]) / (GLOBAL_ACC_R[TOTAL]+GLOBAL_ACC_W[TOTAL])
        # tex_cache_hit_ratek: GLOBAL_ACC_R[HIT]/(GLOBAL_ACC_R[TOTAL]+GLOBAL_ACC_W[TOTAL])
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l1_local_hit_rate"] = grouped["local_hit_rate"].mean()
        self.result_df["l1_local_hit_rate"] /= 100.0

    def _compute_l1_hit_rate(self):
        grouped = self.df.groupby(NVPROF_INDEX_COLS, dropna=False)
        self.result_df["l1_hit_rate"] = grouped["tex_cache_hit_rate"].mean()
        self.result_df["l1_hit_rate"] /= 100.0

    def _compute_l1_hits(self):
        self.result_df["l1_hits"] = np.nan

    def _compute_l1_miss_rate(self):
        self.result_df["l1_miss_rate"] = 1.0 - self.result_df["l1_hit_rate"]

    def _compute_l1_misses(self):
        self.result_df["l1_misses"] = np.nan

    def _kernel_launches_df(self) -> pd.DataFrame:
        # print(self.commands_df.index)
        # commands = self.commands_df.reset_index()
        commands = self.commands_df
        kernel_launches = commands[
            ~commands["Kernel"].str.contains(r"\[CUDA memcpy|memset.*\]")
        ]
        if isinstance(kernel_launches, pd.Series):
            return kernel_launches.to_frame()
        return kernel_launches

    def _kernel_durations_us(self) -> pd.DataFrame:
        # duration already in us
        kernel_launches = self._kernel_launches_df()
        grouped = kernel_launches.groupby(NVPROF_INDEX_COLS, dropna=False)
        return grouped["Duration"].sum()

        # duration_df["mean"] = kernel_launches[nvprof_key + "_mean"]
        # duration_df["mean"] = kernel_launches[nvprof_key + "_mean"]
        # duration_df["min"] = kernel_launches[nvprof_key + "_min"]
        # duration_df["max"] = kernel_launches[nvprof_key + "_max"]
        # duration_df["std"] = kernel_launches[nvprof_key + "_std"]

    def executed_instructions(self):
        nvprof_key = "inst_issued_mean"
        nvprof_key = "inst_executed_mean"
        return self.df[nvprof_key].sum() * self.config.num_total_cores

        # print("inst_issued", self.df["inst_issued"].sum() * self.config.num_total_cores)
        # print(
        #     "inst_executed",
        #     self.df["inst_executed"].sum() * self.config.num_total_cores,
        # )
        #
        # total_instructions = (
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        #     .sum()
        # )
        # print("total", total_instructions)
        #
        # sub_instructions = (
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             # "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        #     .sum()
        # )
        #
        # print(
        #     self.df[
        #         [
        #             "inst_fp_16",
        #             "inst_fp_32",
        #             "inst_fp_64",
        #             "inst_integer",
        #             # "inst_control",
        #             "inst_compute_ld_st",
        #             "inst_misc",
        #         ]
        #     ]
        #     .astype(int)
        #     .sum()
        # )
        # print("sub", sub_instructions)

        # inst_fp_16	{'value': 0, 'unit': None}
        # inst_fp_32	{'value': 100, 'unit': None}
        # inst_fp_64	{'value': 0, 'unit': None}
        # inst_integer	{'value': 4896, 'unit': None}
        # inst_bit_convert	{'value': '0', 'unit': None}
        # inst_control	{'value': '1024', 'unit': None}
        # inst_compute_ld_st	{'value': '300', 'unit': None}
        # inst_misc	{'value': '4196', 'unit': None}
